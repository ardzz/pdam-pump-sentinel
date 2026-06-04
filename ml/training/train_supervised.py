from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

from ml.datasets.skab_loader import SENSOR_COLUMNS, load_skab_csv
from ml.datasets.skab_manifest import SkabSplitManifest, load_skab_split_manifest
from ml.evaluation.metrics import evaluate_split, point_classification_metrics
from ml.features.enriched import build_enriched_window_features
from ml.features.windowing import WindowedSensorDataset
from ml.training.train_pca import _METRIC_PROTOCOL
from ml.utils.provenance import collect_provenance

_MODEL_TYPES = {'lightgbm', 'xgboost'}
_FEATURE_MODES = {'enriched'}
_SPECTRAL_N_BANDS = 4
_ENRICHED_DOMAIN_FEATURE_COUNT = 6
_SUPERVISED_METRIC_PROTOCOL = _METRIC_PROTOCOL | {
    'threshold_calibration': 'supervised validation probability threshold selected by maximum F1',
    'score_semantics': 'positive-class predicted probability',
}


@dataclass(frozen=True)
class SupervisedTrainingConfig:
    input_path: Path
    output_dir: Path
    validation_input_path: Path | None = None
    split_manifest_path: Path | None = None
    window_size: int = 60
    stride: int = 1
    threshold_quantile: float | None = None
    scaler: str | None = 'standard'
    feature_mode: str = 'enriched'
    model_type: str = 'lightgbm'
    n_estimators: int = 300
    learning_rate: float = 0.05
    early_stopping_rounds: int = 30
    seed: int = 42
    log_mlflow: bool = False
    register_model: bool = False
    registered_model_name: str = 'PumpAD'
    alias: str | None = None


@dataclass(frozen=True)
class SupervisedTrainingResult:
    output_dir: Path
    artifact_paths: dict[str, Path]
    params: dict[str, Any]
    metrics: dict[str, Any]
    thresholds: dict[str, float]
    sensor_columns: tuple[str, ...]
    input_example: Any | None = None
    output_example: Any | None = None


def train_supervised_from_skab(config: SupervisedTrainingConfig) -> SupervisedTrainingResult:
    normalized_config = _normalize_config(config)
    mlflow_run_context = _start_mlflow_run_if_requested(normalized_config)
    if mlflow_run_context is None:
        result, _model = _fit_and_write_artifacts(normalized_config)
        return result

    with mlflow_run_context:
        result, model = _fit_and_write_artifacts(normalized_config)
        _log_supervised_training_run_safely(result, model, normalized_config)

    return result


def main(argv: Sequence[str] | None = None) -> SupervisedTrainingResult:
    config = _parse_args(argv)
    result = train_supervised_from_skab(config)
    print(json.dumps(_result_payload(result), sort_keys=True))
    return result


def _fit_and_write_artifacts(config: SupervisedTrainingConfig) -> tuple[SupervisedTrainingResult, Any]:
    if config.split_manifest_path is not None:
        return _fit_and_write_artifacts_split(config)

    training_windows = _load_windows(config.input_path, config)
    validation_path = config.validation_input_path or config.input_path
    validation_windows = training_windows if validation_path == config.input_path else _load_windows(validation_path, config)
    return _fit_and_write_artifacts_common(
        config=config,
        train_windows=training_windows,
        validation_windows=validation_windows,
        test_windows=None,
        provenance_input_files=_non_split_input_files(config),
    )


def _fit_and_write_artifacts_split(config: SupervisedTrainingConfig) -> tuple[SupervisedTrainingResult, Any]:
    split_manifest_path = config.split_manifest_path
    if split_manifest_path is None:
        raise ValueError('split_manifest_path is required for split-manifest training')
    manifest = load_skab_split_manifest(split_manifest_path)

    train_windows = _load_windows_multi(manifest.train, config)
    validation_windows = _load_windows_multi(manifest.validation, config)
    test_windows = _load_windows_multi(manifest.test, config) if manifest.test else None
    return _fit_and_write_artifacts_common(
        config=config,
        train_windows=train_windows,
        validation_windows=validation_windows,
        test_windows=test_windows,
        provenance_input_files=_split_input_files(config, manifest),
        manifest=manifest,
    )


def _fit_and_write_artifacts_common(
    *,
    config: SupervisedTrainingConfig,
    train_windows: WindowedSensorDataset,
    validation_windows: WindowedSensorDataset,
    test_windows: WindowedSensorDataset | None,
    provenance_input_files: Sequence[Path],
    manifest: SkabSplitManifest | None = None,
) -> tuple[SupervisedTrainingResult, Any]:
    if len(train_windows.features) == 0:
        raise ValueError('training input must produce at least one window')
    if len(validation_windows.features) == 0:
        raise ValueError('validation input must produce at least one window')

    train_labels = train_windows.labels.astype(int, copy=False)
    validation_labels = validation_windows.labels.astype(int, copy=False)
    _require_two_training_classes(train_labels)

    scaler = _make_scaler(config.scaler)
    train_x = _fit_transform_scaler(scaler, train_windows.features)
    validation_x = _transform_scaler(scaler, validation_windows.features)
    test_x = _transform_scaler(scaler, test_windows.features) if test_windows is not None else None

    model = _fit_model(config, train_x, train_labels, validation_x, validation_labels)
    validation_scores = _positive_probabilities(model, validation_x)
    threshold = _threshold_for_max_f1(validation_labels, validation_scores)
    thresholds = {'threshold': threshold, 't2_threshold': threshold, 'q_threshold': threshold}
    validation_predictions = _predict_from_threshold(validation_scores, threshold)
    validation_metrics = evaluate_split(
        validation_labels,
        validation_predictions,
        validation_scores,
        transient_mask=validation_windows.changepoints,
    ) | _accuracy_metric(validation_labels, validation_predictions)

    metrics = validation_metrics | {
        **_count_metrics(validation_labels, validation_windows.changepoints),
        'training_sample_count': int(len(train_windows.labels)),
        'training_anomaly_count': int(np.count_nonzero(train_labels == 1)),
        'training_normal_count': int(np.count_nonzero(train_labels == 0)),
        'train_count': int(len(train_windows.labels)),
        'validation_count': int(len(validation_windows.labels)),
        **thresholds,
    }

    has_test = test_windows is not None and test_x is not None and len(test_windows.features) > 0
    if has_test and test_windows is not None:
        metrics['test_count'] = int(len(test_windows.labels))

    params = _params(config, train_windows, threshold)
    artifact_paths = _artifact_paths(config.output_dir, config.model_type, scaler is not None, manifest is not None, has_test)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    _dump_joblib(model, artifact_paths['model'])
    if scaler is not None:
        _dump_joblib(scaler, artifact_paths['scaler'])
    _scores_frame(validation_windows, validation_labels, validation_predictions, validation_scores).to_csv(
        artifact_paths['scores'], index=False
    )

    if has_test and test_windows is not None and test_x is not None:
        test_scores = _positive_probabilities(model, test_x)
        test_predictions = _predict_from_threshold(test_scores, threshold)
        test_labels = test_windows.labels.astype(int, copy=False)
        test_metrics = evaluate_split(
            test_labels,
            test_predictions,
            test_scores,
            transient_mask=test_windows.changepoints,
        ) | _accuracy_metric(test_labels, test_predictions)
        metrics.update({f'test_{name}': value for name, value in test_metrics.items()})
        metrics.update(_count_metrics(test_labels, test_windows.changepoints, prefix='test_'))
        _scores_frame(test_windows, test_labels, test_predictions, test_scores).to_csv(
            artifact_paths['test_scores'], index=False
        )

    if manifest is not None:
        _write_json(artifact_paths['split_manifest'], manifest.to_payload())

    result = SupervisedTrainingResult(
        output_dir=config.output_dir,
        artifact_paths=artifact_paths,
        params=params,
        metrics=metrics,
        thresholds=thresholds,
        sensor_columns=train_windows.sensor_columns,
        input_example=_input_example(train_x),
        output_example=_output_example(model, _input_example(train_x)),
    )
    _write_json(artifact_paths['metrics'], metrics)
    _write_json(
        artifact_paths['metadata'],
        _metadata_payload(config, result, provenance_input_files, manifest, train_windows, validation_windows, test_windows),
    )
    _log_skab_inputs_to_active_run(config, manifest, train_windows, validation_windows, test_windows)
    if not config.log_mlflow and _mlflow_module_for_live_boosting() is not None:
        _log_supervised_training_run_safely(result, model, config)
    return result, model


def _parse_args(argv: Sequence[str] | None) -> SupervisedTrainingConfig:
    parser = argparse.ArgumentParser(description='Train a supervised SKAB gradient-boosting anomaly classifier.')
    parser.add_argument('paths', type=Path, nargs='+', metavar='PATH')
    parser.add_argument('--validation-input-path', type=Path, default=None)
    parser.add_argument('--split-manifest', type=Path, default=None)
    parser.add_argument('--window-size', type=int, default=60)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--threshold-quantile', type=float, default=None)
    parser.add_argument('--scaler', default='standard')
    parser.add_argument('--feature-mode', choices=sorted(_FEATURE_MODES), default='enriched')
    parser.add_argument('--model-type', choices=sorted(_MODEL_TYPES), default='lightgbm')
    parser.add_argument('--n-estimators', type=int, default=300)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--early-stopping-rounds', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-mlflow', action='store_true')
    parser.add_argument('--register-model', action='store_true')
    parser.add_argument('--registered-model-name', default='PumpAD')
    parser.add_argument('--alias', default=None)
    args = parser.parse_args(argv)
    args_dict = vars(args)
    paths = args_dict.pop('paths')
    if 'split_manifest' in args_dict:
        args_dict['split_manifest_path'] = args_dict.pop('split_manifest')
    if args_dict['split_manifest_path'] is not None and len(paths) == 1:
        args_dict['input_path'] = paths[0]
        args_dict['output_dir'] = paths[0]
    elif len(paths) == 2:
        args_dict['input_path'] = paths[0]
        args_dict['output_dir'] = paths[1]
    else:
        parser.error('expected input_path output_dir, or output_dir with --split-manifest')
    return SupervisedTrainingConfig(**args_dict)


def _normalize_config(config: SupervisedTrainingConfig) -> SupervisedTrainingConfig:
    validation_input_path = Path(config.validation_input_path) if config.validation_input_path is not None else None
    split_manifest_path = Path(config.split_manifest_path) if config.split_manifest_path is not None else None
    feature_mode = config.feature_mode.lower()
    model_type = config.model_type.lower()
    if feature_mode not in _FEATURE_MODES:
        raise ValueError(f'feature_mode must be one of: {", ".join(sorted(_FEATURE_MODES))}')
    if model_type not in _MODEL_TYPES:
        raise ValueError(f'model_type must be one of: {", ".join(sorted(_MODEL_TYPES))}')
    _validated_positive_integer(config.window_size, 'window_size')
    _validated_positive_integer(config.stride, 'stride')
    _validated_positive_integer(config.n_estimators, 'n_estimators')
    if not np.isfinite(float(config.learning_rate)) or float(config.learning_rate) <= 0.0:
        raise ValueError('learning_rate must be positive')
    if isinstance(config.early_stopping_rounds, bool) or not isinstance(config.early_stopping_rounds, int):
        raise ValueError('early_stopping_rounds must be a non-negative integer')
    if config.early_stopping_rounds < 0:
        raise ValueError('early_stopping_rounds must be a non-negative integer')
    return replace(
        config,
        input_path=Path(config.input_path),
        output_dir=Path(config.output_dir),
        validation_input_path=validation_input_path,
        split_manifest_path=split_manifest_path,
        feature_mode=feature_mode,
        model_type=model_type,
    )


def _load_windows(path: Path, config: SupervisedTrainingConfig) -> WindowedSensorDataset:
    frame = load_skab_csv(path)
    features, labels, changepoints, timestamps = build_enriched_window_features(
        frame,
        window_size=config.window_size,
        stride=config.stride,
        sensor_columns=SENSOR_COLUMNS,
        n_bands=_SPECTRAL_N_BANDS,
    )
    return WindowedSensorDataset(
        features=features,
        labels=labels,
        changepoints=changepoints,
        timestamps=timestamps,
        sensor_columns=tuple(SENSOR_COLUMNS),
        window_size=config.window_size,
        stride=config.stride,
    )


def _load_windows_multi(paths: list[Path], config: SupervisedTrainingConfig) -> WindowedSensorDataset:
    datasets = [_load_windows(path, config) for path in paths]
    if not datasets:
        return WindowedSensorDataset(
            features=np.empty((0, _feature_count()), dtype=float),
            labels=np.empty((0,), dtype=int),
            changepoints=np.empty((0,), dtype=int),
            timestamps=np.empty((0,), dtype=object),
            sensor_columns=tuple(SENSOR_COLUMNS),
            window_size=config.window_size,
            stride=config.stride,
        )
    return _concat_datasets(datasets)


def _concat_datasets(datasets: list[WindowedSensorDataset]) -> WindowedSensorDataset:
    if len(datasets) == 1:
        return datasets[0]
    return WindowedSensorDataset(
        features=np.vstack([dataset.features for dataset in datasets]),
        labels=np.concatenate([dataset.labels for dataset in datasets]),
        changepoints=np.concatenate([dataset.changepoints for dataset in datasets]),
        timestamps=np.concatenate([dataset.timestamps for dataset in datasets]),
        sensor_columns=datasets[0].sensor_columns,
        window_size=datasets[0].window_size,
        stride=datasets[0].stride,
    )


def _feature_count() -> int:
    return len(SENSOR_COLUMNS) * (_SPECTRAL_N_BANDS + 7) + _ENRICHED_DOMAIN_FEATURE_COUNT


def _make_scaler(name: str | None):
    if name is None:
        return None
    scaler_name = name.lower()
    if scaler_name == 'standard':
        return StandardScaler()
    if scaler_name == 'robust':
        return RobustScaler()
    if scaler_name == 'none':
        return None
    raise ValueError("scaler must be one of 'standard', 'robust', 'none', or None")


def _fit_transform_scaler(scaler: Any, features: np.ndarray) -> np.ndarray:
    if scaler is None:
        return np.asarray(features, dtype=np.float64)
    return np.asarray(scaler.fit_transform(features), dtype=np.float64)


def _transform_scaler(scaler: Any, features: np.ndarray) -> np.ndarray:
    if scaler is None:
        return np.asarray(features, dtype=np.float64)
    return np.asarray(scaler.transform(features), dtype=np.float64)


def _fit_model(
    config: SupervisedTrainingConfig,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> Any:
    if config.model_type == 'xgboost':
        return _fit_xgboost(config, train_x, train_y, validation_x, validation_y)
    if config.model_type == 'lightgbm':
        return _fit_lightgbm(config, train_x, train_y, validation_x, validation_y)
    raise ValueError(f'unsupported model_type: {config.model_type}')


def _fit_xgboost(
    config: SupervisedTrainingConfig,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> Any:
    xgboost = import_module('xgboost')
    neg_count = int(np.count_nonzero(train_y == 0))
    pos_count = int(np.count_nonzero(train_y == 1))
    model_params: dict[str, Any] = {
        'n_estimators': config.n_estimators,
        'learning_rate': config.learning_rate,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': config.seed,
        'n_jobs': 1,
        'tree_method': 'hist',
        'scale_pos_weight': float(neg_count / pos_count) if pos_count else 1.0,
        'verbosity': 0,
    }
    eval_set = None
    callbacks = _xgboost_mlflow_callbacks(xgboost, config)
    if callbacks:
        model_params['callbacks'] = callbacks
    if config.early_stopping_rounds > 0 and len(validation_x) > 0:
        model_params['early_stopping_rounds'] = config.early_stopping_rounds
        eval_set = [(train_x, train_y), (validation_x, validation_y)]
    model = xgboost.XGBClassifier(**model_params)
    model.fit(train_x, train_y, eval_set=eval_set, verbose=False)
    _mark_boosting_curves_live_streamed(model, callbacks)
    return model


def _fit_lightgbm(
    config: SupervisedTrainingConfig,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> Any:
    lightgbm = import_module('lightgbm')
    model = lightgbm.LGBMClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        objective='binary',
        class_weight='balanced',
        random_state=config.seed,
        n_jobs=1,
        verbosity=-1,
    )
    callbacks = []
    eval_set = None
    eval_names = None
    if config.early_stopping_rounds > 0 and len(validation_x) > 0:
        callbacks.append(lightgbm.early_stopping(config.early_stopping_rounds, verbose=False))
        eval_set = [(train_x, train_y), (validation_x, validation_y)]
        eval_names = ['train', 'validation']
    mlflow_callback = _lightgbm_mlflow_callback(config)
    if mlflow_callback is not None:
        callbacks.append(mlflow_callback)
    model.fit(
        train_x,
        train_y,
        eval_set=eval_set,
        eval_names=eval_names,
        eval_metric='binary_logloss',
        callbacks=callbacks or None,
    )
    _mark_boosting_curves_live_streamed(model, [mlflow_callback] if mlflow_callback is not None else [])
    return model


import xgboost.callback as _xgb_callback_module


class MlflowXGBCallback(_xgb_callback_module.TrainingCallback):
    def __init__(self, family_key: str = 'xgb') -> None:
        super().__init__()
        self._family_key = family_key

    def after_iteration(self, model: Any, epoch: int, evals_log: Mapping[str, Any]) -> bool:
        mlflow = _mlflow_module_for_live_boosting()
        if mlflow is None:
            return False
        dataset_items = list((evals_log or {}).items())
        for dataset_index, (dataset_name, metrics_by_name) in enumerate(dataset_items):
            if not isinstance(metrics_by_name, Mapping):
                continue
            dataset_key = _eval_dataset_key(str(dataset_name), dataset_index, len(dataset_items))
            for metric_name, values in metrics_by_name.items():
                if not values:
                    continue
                _log_live_boosting_metric(
                    mlflow,
                    f'{dataset_key}_{self._family_key}_{_metric_key(str(metric_name))}_round',
                    values[-1],
                    epoch,
                )
        return False


def _xgboost_mlflow_callbacks(xgboost: Any, config: SupervisedTrainingConfig) -> list[Any]:
    if config.model_type != 'xgboost':
        return []
    if _mlflow_module_for_live_boosting() is None:
        return []
    return [MlflowXGBCallback()]


class MlflowLGBMCallback:
    order = 25
    before_iteration = False

    def __init__(self, mlflow: Any):
        self._mlflow = mlflow

    def __call__(self, env: Any) -> None:
        results = list(env.evaluation_result_list or [])
        for dataset_index, item in enumerate(results):
            dataset_name, metric_name, metric_value, *_ = item
            dataset_key = _eval_dataset_key(str(dataset_name), dataset_index, len(results))
            _log_live_boosting_metric(
                self._mlflow,
                f'{dataset_key}_lgbm_{_metric_key(str(metric_name))}_round',
                metric_value,
                env.iteration,
            )


def _lightgbm_mlflow_callback(config: SupervisedTrainingConfig) -> Any | None:
    if config.model_type != 'lightgbm':
        return None
    mlflow = _mlflow_module_for_live_boosting()
    return MlflowLGBMCallback(mlflow) if mlflow is not None else None


def _mlflow_module_for_live_boosting() -> Any | None:
    try:
        mlflow = import_module('mlflow')
    except ImportError:
        return None
    active_run = getattr(mlflow, 'active_run', None)
    if callable(active_run):
        try:
            if active_run() is None:
                return None
        except Exception:
            return None
    return mlflow


def _log_live_boosting_metric(mlflow: Any, key: str, value: Any, step: int) -> None:
    try:
        metric_value = float(value)
    except (TypeError, ValueError):
        return
    if not np.isfinite(metric_value):
        return
    try:
        mlflow.log_metric(key, metric_value, step=int(step))
    except Exception:
        pass


def _mark_boosting_curves_live_streamed(model: Any, callbacks: Sequence[Any]) -> None:
    if callbacks:
        try:
            setattr(model, '_pump_sentinel_mlflow_live_streamed', True)
        except Exception:
            pass


def _positive_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.empty((0,), dtype=float)
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = [int(value) for value in np.asarray(getattr(model, 'classes_', [0, 1])).reshape(-1)]
    if probabilities.ndim == 1:
        return probabilities.reshape(-1)
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    return np.zeros((probabilities.shape[0],), dtype=float)


def _threshold_for_max_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(scores) == 0:
        raise ValueError('validation scores must not be empty')
    unique_labels = set(np.asarray(labels, dtype=int).reshape(-1).tolist())
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if unique_labels == {0}:
        return float(np.nextafter(max_score, np.inf))
    if unique_labels == {1}:
        return float(np.nextafter(min_score, -np.inf))

    candidates = np.unique(np.asarray(scores, dtype=np.float64))
    candidates = np.asarray(
        [np.nextafter(float(candidates[0]), -np.inf), *candidates.tolist(), np.nextafter(float(candidates[-1]), np.inf), 0.5],
        dtype=np.float64,
    )
    best_threshold = float(candidates[0])
    best_key = (-1.0, -1.0, -1.0, -1.0, -np.inf)
    for threshold in candidates:
        predictions = _predict_from_threshold(scores, float(threshold))
        point_metrics = point_classification_metrics(labels, predictions)
        key = (
            float(point_metrics['f1']),
            float(point_metrics['precision']),
            float(point_metrics['recall']),
            -float(point_metrics['false_alarm_rate']),
            float(threshold),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _predict_from_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores, dtype=np.float64) >= float(threshold)).astype(int)


def _accuracy_metric(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=int).reshape(-1)
    predictions_array = np.asarray(predictions, dtype=int).reshape(-1)
    if labels_array.size == 0:
        return {'accuracy': 0.0}
    return {'accuracy': float(np.mean(labels_array == predictions_array))}


def _artifact_paths(output_dir: Path, model_type: str, has_scaler: bool, has_manifest: bool, has_test: bool) -> dict[str, Path]:
    paths = {
        'model': output_dir / f'{model_type}.joblib',
        'metadata': output_dir / 'metadata.json',
        'metrics': output_dir / 'metrics.json',
        'scores': output_dir / 'scores.csv',
    }
    if has_scaler:
        paths['scaler'] = output_dir / 'scaler.joblib'
    if has_manifest:
        paths['split_manifest'] = output_dir / 'split_manifest.json'
    if has_test:
        paths['test_scores'] = output_dir / 'test_scores.csv'
    return paths


def _scores_frame(dataset: WindowedSensorDataset, labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> Any:
    pandas = import_module('pandas')
    return pandas.DataFrame(
        {
            'timestamp': dataset.timestamps.astype(str),
            'label': labels.astype(int),
            'changepoint': dataset.changepoints.astype(int),
            'prediction': predictions.astype(int),
            'score': scores.astype(float),
        }
    )


def _count_metrics(labels: np.ndarray, changepoints: np.ndarray, prefix: str = '') -> dict[str, int]:
    return {
        f'{prefix}sample_count': int(len(labels)),
        f'{prefix}anomaly_count': int(np.count_nonzero(labels == 1)),
        f'{prefix}normal_count': int(np.count_nonzero(labels == 0)),
        f'{prefix}changepoint_count': int(np.count_nonzero(changepoints == 1)),
    }


def _params(config: SupervisedTrainingConfig, training_windows: WindowedSensorDataset, threshold: float) -> dict[str, Any]:
    return {
        'input_path': str(config.input_path),
        'output_dir': str(config.output_dir),
        'validation_input_path': str(config.validation_input_path) if config.validation_input_path is not None else None,
        'split_manifest_path': str(config.split_manifest_path) if config.split_manifest_path is not None else None,
        'window_size': config.window_size,
        'stride': config.stride,
        'threshold': threshold,
        'threshold_quantile': config.threshold_quantile,
        'scaler': config.scaler,
        'feature_mode': config.feature_mode,
        'model_type': config.model_type,
        'model_family': config.model_type,
        'n_estimators': config.n_estimators,
        'learning_rate': config.learning_rate,
        'early_stopping_rounds': config.early_stopping_rounds,
        'seed': config.seed,
        'log_mlflow': config.log_mlflow,
        'register_model': config.register_model,
        'registered_model_name': config.registered_model_name,
        'alias': config.alias,
        'sensor_columns': list(training_windows.sensor_columns),
        'sensor_count': len(training_windows.sensor_columns),
        'feature_count': int(training_windows.features.shape[1]),
        'spectral_n_bands': _SPECTRAL_N_BANDS,
    }


def _metadata_payload(
    config: SupervisedTrainingConfig,
    result: SupervisedTrainingResult,
    provenance_input_files: Sequence[Path],
    manifest: SkabSplitManifest | None,
    train_windows: WindowedSensorDataset,
    validation_windows: WindowedSensorDataset,
    test_windows: WindowedSensorDataset | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'model_family': config.model_type,
        'params': result.params,
        'metrics': result.metrics,
        'thresholds': result.thresholds,
        'sensor_columns': list(result.sensor_columns),
        'artifact_paths': {name: str(path) for name, path in result.artifact_paths.items()},
        'config': _jsonable_config(config),
        'metric_protocol': _SUPERVISED_METRIC_PROTOCOL,
        'provenance': collect_provenance(config=_jsonable_config(config), input_files=provenance_input_files),
    }
    if manifest is not None:
        manifest_payload = manifest.to_payload()
        has_held_out_test = test_windows is not None and len(test_windows.features) > 0
        payload['test_split_held_out'] = has_held_out_test
        payload['split'] = {
            'train_files': manifest_payload['train'],
            'validation_files': manifest_payload['validation'],
            'test_files': manifest_payload['test'],
            'train_count': int(len(train_windows.labels)),
            'validation_count': int(len(validation_windows.labels)),
            'test_count': int(len(test_windows.labels)) if test_windows is not None else 0,
            'test_split_held_out': has_held_out_test,
        }
    return payload


def _non_split_input_files(config: SupervisedTrainingConfig) -> list[Path]:
    paths = [config.input_path]
    if config.validation_input_path is not None:
        paths.append(config.validation_input_path)
    return _unique_paths(paths)


def _split_input_files(config: SupervisedTrainingConfig, manifest: SkabSplitManifest) -> list[Path]:
    paths: list[Path] = []
    if config.split_manifest_path is not None:
        paths.append(config.split_manifest_path)
    paths.extend([*manifest.train, *manifest.validation, *manifest.test])
    return _unique_paths(paths)


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _log_skab_inputs_to_active_run(
    config: SupervisedTrainingConfig,
    manifest: SkabSplitManifest | None,
    train_windows: WindowedSensorDataset,
    validation_windows: WindowedSensorDataset,
    test_windows: WindowedSensorDataset | None,
) -> None:
    try:
        from ml.registry.mlflow_client import log_skab_inputs_to_active_run, skab_window_dataframe
    except Exception:
        return
    log_skab_inputs_to_active_run(
        train_df=skab_window_dataframe(train_windows),
        val_df=skab_window_dataframe(validation_windows),
        test_df=skab_window_dataframe(test_windows) if test_windows is not None else None,
        manifest_path=config.split_manifest_path or config.input_path,
        manifest_dict=manifest.to_payload() if manifest is not None else None,
        feature_mode=config.feature_mode,
        split_strategy='supervised_cross_group' if config.split_manifest_path is not None else 'single_csv',
    )


def _input_example(features: np.ndarray) -> np.ndarray | None:
    if len(features) == 0:
        return None
    return np.asarray(features[:5], dtype=np.float64)


def _output_example(model: Any, input_example: np.ndarray | None) -> np.ndarray | None:
    if input_example is None:
        return None
    return np.asarray(model.predict_proba(input_example), dtype=np.float64)


def _dump_joblib(value: Any, path: Path) -> None:
    joblib = import_module('joblib')
    joblib.dump(value, path)


def _start_mlflow_run_if_requested(config: SupervisedTrainingConfig) -> Any | None:
    if not config.log_mlflow:
        return None
    try:
        mlflow = import_module('mlflow')
    except ImportError:
        return None
    try:
        try:
            from ml.registry.mlflow_client import _enable_system_metrics_logging_if_available

            _enable_system_metrics_logging_if_available(mlflow)
        except Exception:
            pass
        active_run = mlflow.active_run() if hasattr(mlflow, 'active_run') else None
        if active_run is not None:
            return nullcontext(active_run)
        return mlflow.start_run()
    except Exception:
        return None


def _log_supervised_training_run_safely(result: SupervisedTrainingResult, model: Any, config: SupervisedTrainingConfig) -> None:
    try:
        mlflow = import_module('mlflow')
        mlflow_sklearn = import_module('mlflow.sklearn')
    except ImportError:
        return
    try:
        active_run = mlflow.active_run() if hasattr(mlflow, 'active_run') else None
        if active_run is not None:
            _log_supervised_training_run_to_active_run(mlflow, mlflow_sklearn, result, model, config)
            return
        with mlflow.start_run():
            _log_supervised_training_run_to_active_run(mlflow, mlflow_sklearn, result, model, config)
    except Exception:
        return


def _log_supervised_training_run_to_active_run(
    mlflow: Any,
    mlflow_sklearn: Any,
    result: SupervisedTrainingResult,
    model: Any,
    config: SupervisedTrainingConfig,
) -> None:
    try:
        from ml.registry.mlflow_client import set_run_traceability_tags

        set_run_traceability_tags(mlflow)
    except Exception:
        pass
    scalar_params = {
        name: value
        for name, value in result.params.items()
        if value is None or isinstance(value, bool | int | float | str)
    }
    if scalar_params:
        mlflow.log_params({name: value for name, value in scalar_params.items() if value is not None})
    numeric_metrics = {
        name: float(value)
        for name, value in result.metrics.items()
        if not isinstance(value, bool) and isinstance(value, int | float) and np.isfinite(float(value))
    }
    if numeric_metrics:
        mlflow.log_metrics(numeric_metrics)
    _log_boosting_eval_curves(mlflow, model, config)
    mlflow.log_artifacts(str(result.output_dir))
    mlflow_sklearn.log_model(
        model,
        name=f'{config.model_type}_model',
        registered_model_name=config.registered_model_name if config.register_model else None,
        **_model_signature_kwargs(result),
    )


def _log_boosting_eval_curves(mlflow: Any, model: Any, config: SupervisedTrainingConfig) -> None:
    if getattr(model, '_pump_sentinel_mlflow_live_streamed', False):
        return
    evals_result = _evals_result(model)
    if not isinstance(evals_result, Mapping) or not evals_result:
        return

    family_key = 'xgb' if config.model_type == 'xgboost' else 'lgbm'
    dataset_items = list(evals_result.items())
    for dataset_index, (dataset_name, metrics_by_name) in enumerate(dataset_items):
        if not isinstance(metrics_by_name, Mapping):
            continue
        dataset_key = _eval_dataset_key(str(dataset_name), dataset_index, len(dataset_items))
        for metric_name, values in metrics_by_name.items():
            metric_key = f'{dataset_key}_{family_key}_{_metric_key(str(metric_name))}_round'
            for step, value in enumerate(values or []):
                try:
                    metric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(metric_value):
                    continue
                mlflow.log_metric(metric_key, metric_value, step=int(step))


def _model_signature_kwargs(result: SupervisedTrainingResult) -> dict[str, Any]:
    if result.input_example is None:
        return {}
    try:
        mlflow_models = import_module('mlflow.models')
        signature = mlflow_models.infer_signature(result.input_example, result.output_example)
    except Exception:
        return {'input_example': result.input_example}
    return {'signature': signature, 'input_example': result.input_example}


def _evals_result(model: Any) -> Any:
    attr_result = getattr(model, 'evals_result_', None)
    if attr_result:
        return attr_result
    method = getattr(model, 'evals_result', None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:
        return None


def _eval_dataset_key(dataset_name: str, index: int, total: int) -> str:
    normalized = dataset_name.lower()
    if 'train' in normalized:
        return 'train'
    if total > 1 and index == 0:
        return 'train'
    if 'valid' in normalized or 'eval' in normalized or 'validation' in normalized:
        return 'val'
    if total > 1 and index == total - 1:
        return 'val'
    return _metric_key(normalized)


def _metric_key(metric_name: str) -> str:
    return ''.join(char if char.isalnum() else '_' for char in metric_name.lower()).strip('_') or 'metric'


def _require_two_training_classes(labels: np.ndarray) -> None:
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError(f'training split must contain both binary classes, got {classes.tolist()}')


def _validated_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def _jsonable_config(config: SupervisedTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ('input_path', 'output_dir', 'validation_input_path', 'split_manifest_path'):
        value = payload[key]
        payload[key] = str(value) if value is not None else None
    return payload


def _result_payload(result: SupervisedTrainingResult) -> dict[str, Any]:
    return {
        'output_dir': str(result.output_dir),
        'artifact_paths': {name: str(path) for name, path in result.artifact_paths.items()},
        'metrics': result.metrics,
        'thresholds': result.thresholds,
        'sensor_columns': list(result.sensor_columns),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
