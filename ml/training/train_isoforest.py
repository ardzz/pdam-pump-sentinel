from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler

from ml.datasets.skab_loader import SENSOR_COLUMNS, load_skab_csv
from ml.datasets.skab_manifest import SkabSplitManifest, load_skab_split_manifest
from ml.evaluation.metrics import evaluate_split
from ml.features.spectral import build_spectral_window_features
from ml.features.windowing import WindowedSensorDataset, build_sensor_windows
from ml.training.train_pca import _METRIC_PROTOCOL
from ml.utils.provenance import collect_provenance

_FEATURE_MODES = {'raw', 'spectral'}
_SPECTRAL_N_BANDS = 4


@dataclass(frozen=True)
class IsoForestTrainingConfig:
    input_path: Path
    output_dir: Path
    validation_input_path: Path | None = None
    split_manifest_path: Path | None = None
    window_size: int = 60
    stride: int = 1
    threshold_quantile: float = 0.95
    scaler: str | None = 'standard'
    feature_mode: str = 'spectral'
    log_mlflow: bool = False
    register_model: bool = False
    registered_model_name: str = 'PumpAD'
    alias: str | None = None
    n_estimators: int = 200
    contamination: str | float = 'auto'
    seed: int = 42


@dataclass(frozen=True)
class IsoForestTrainingResult:
    output_dir: Path
    artifact_paths: dict[str, Path]
    params: dict[str, Any]
    metrics: dict[str, Any]
    thresholds: dict[str, float]
    sensor_columns: tuple[str, ...]


def train_isoforest_from_skab(config: IsoForestTrainingConfig) -> IsoForestTrainingResult:
    normalized_config = _normalize_config(config)
    result, _model = _fit_and_write_artifacts(normalized_config)
    return result


def main(argv: Sequence[str] | None = None) -> IsoForestTrainingResult:
    config = _parse_args(argv)
    result = train_isoforest_from_skab(config)
    print(json.dumps(_result_payload(result), sort_keys=True))
    return result


def _fit_and_write_artifacts(config: IsoForestTrainingConfig) -> tuple[IsoForestTrainingResult, IsolationForest]:
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


def _fit_and_write_artifacts_split(config: IsoForestTrainingConfig) -> tuple[IsoForestTrainingResult, IsolationForest]:
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
    config: IsoForestTrainingConfig,
    train_windows: WindowedSensorDataset,
    validation_windows: WindowedSensorDataset,
    test_windows: WindowedSensorDataset | None,
    provenance_input_files: Sequence[Path],
    manifest: SkabSplitManifest | None = None,
) -> tuple[IsoForestTrainingResult, IsolationForest]:
    train_normal_mask = train_windows.labels == 0
    train_normal_features = train_windows.features[train_normal_mask]
    if len(train_normal_features) == 0:
        raise ValueError('training input must contain at least one normal window')
    if len(validation_windows.features) == 0:
        raise ValueError('validation input must produce at least one window')

    validation_normal_mask = validation_windows.labels == 0
    validation_normal_features = validation_windows.features[validation_normal_mask]
    if len(validation_normal_features) == 0:
        raise ValueError('validation input must contain at least one normal window')

    scaler = _make_scaler(config.scaler)
    train_normal_x = _fit_transform_scaler(scaler, train_normal_features)
    validation_x = _transform_scaler(scaler, validation_windows.features)
    validation_normal_x = validation_x[validation_normal_mask]
    test_x = _transform_scaler(scaler, test_windows.features) if test_windows is not None else None

    model_params: dict[str, Any] = {
        'n_estimators': config.n_estimators,
        'contamination': config.contamination,
        'random_state': config.seed,
    }
    model = IsolationForest(**model_params)
    model.fit(train_normal_x)

    validation_normal_scores = _anomaly_scores(model, validation_normal_x)
    threshold = float(np.percentile(validation_normal_scores, config.threshold_quantile * 100.0))
    thresholds = {'threshold': threshold, 't2_threshold': threshold, 'q_threshold': threshold}

    validation_scores = _anomaly_scores(model, validation_x)
    validation_predictions = (validation_scores > threshold).astype(int)
    validation_labels = validation_windows.labels.astype(int, copy=False)
    metrics = evaluate_split(
        validation_labels,
        validation_predictions,
        validation_scores,
        transient_mask=validation_windows.changepoints,
    ) | {
        **_count_metrics(validation_labels, validation_windows.changepoints),
        'training_sample_count': int(len(train_windows.labels)),
        'training_normal_count': int(len(train_normal_features)),
        'train_count': int(len(train_windows.labels)),
        'validation_count': int(len(validation_windows.labels)),
        **thresholds,
    }

    has_test = test_windows is not None and test_x is not None and len(test_windows.features) > 0
    if has_test and test_windows is not None:
        metrics['test_count'] = int(len(test_windows.labels))

    params = _params(config, train_windows, threshold)
    artifact_paths = _artifact_paths(config.output_dir, scaler is not None, manifest is not None, has_test)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    _dump_joblib(model, artifact_paths['model'])
    if scaler is not None:
        _dump_joblib(scaler, artifact_paths['scaler'])
    _scores_frame(validation_windows, validation_labels, validation_predictions, validation_scores).to_csv(
        artifact_paths['scores'], index=False
    )

    if has_test and test_windows is not None and test_x is not None:
        test_scores = _anomaly_scores(model, test_x)
        test_predictions = (test_scores > threshold).astype(int)
        test_labels = test_windows.labels.astype(int, copy=False)
        test_metrics = evaluate_split(
            test_labels,
            test_predictions,
            test_scores,
            transient_mask=test_windows.changepoints,
        )
        metrics.update({f'test_{name}': value for name, value in test_metrics.items()})
        metrics.update(_count_metrics(test_labels, test_windows.changepoints, prefix='test_'))
        _scores_frame(test_windows, test_labels, test_predictions, test_scores).to_csv(
            artifact_paths['test_scores'], index=False
        )

    if manifest is not None:
        _write_json(artifact_paths['split_manifest'], manifest.to_payload())

    result = IsoForestTrainingResult(
        output_dir=config.output_dir,
        artifact_paths=artifact_paths,
        params=params,
        metrics=metrics,
        thresholds=thresholds,
        sensor_columns=train_windows.sensor_columns,
    )
    _write_json(artifact_paths['metrics'], metrics)
    _write_json(
        artifact_paths['metadata'],
        _metadata_payload(config, result, provenance_input_files, manifest, train_windows, validation_windows, test_windows),
    )
    _log_skab_inputs_to_active_run(config, manifest, train_windows, validation_windows, test_windows)
    return result, model


def _parse_args(argv: Sequence[str] | None) -> IsoForestTrainingConfig:
    parser = argparse.ArgumentParser(description='Train an Isolation Forest detector from SKAB CSV telemetry.')
    parser.add_argument('paths', type=Path, nargs='+', metavar='PATH')
    parser.add_argument('--validation-input-path', type=Path, default=None)
    parser.add_argument('--split-manifest', type=Path, default=None)
    parser.add_argument('--window-size', type=int, default=60)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--threshold-quantile', type=float, default=0.95)
    parser.add_argument('--scaler', default='standard')
    parser.add_argument('--feature-mode', choices=sorted(_FEATURE_MODES), default='spectral')
    parser.add_argument('--log-mlflow', action='store_true')
    parser.add_argument('--register-model', action='store_true')
    parser.add_argument('--registered-model-name', default='PumpAD')
    parser.add_argument('--alias', default=None)
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--contamination', type=_parse_contamination, default='auto')
    parser.add_argument('--seed', type=int, default=42)
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
    return IsoForestTrainingConfig(**args_dict)


def _parse_contamination(value: str) -> str | float:
    if value.lower() == 'auto':
        return 'auto'
    return float(value)


def _normalize_config(config: IsoForestTrainingConfig) -> IsoForestTrainingConfig:
    validation_input_path = Path(config.validation_input_path) if config.validation_input_path is not None else None
    split_manifest_path = Path(config.split_manifest_path) if config.split_manifest_path is not None else None
    feature_mode = config.feature_mode.lower()
    if feature_mode not in _FEATURE_MODES:
        raise ValueError(f'feature_mode must be one of: {", ".join(sorted(_FEATURE_MODES))}')
    _validated_threshold_quantile(config.threshold_quantile)
    _validated_n_estimators(config.n_estimators)
    contamination = _validated_contamination(config.contamination)
    return replace(
        config,
        input_path=Path(config.input_path),
        output_dir=Path(config.output_dir),
        validation_input_path=validation_input_path,
        split_manifest_path=split_manifest_path,
        feature_mode=feature_mode,
        contamination=contamination,
    )


def _load_windows(path: Path, config: IsoForestTrainingConfig) -> WindowedSensorDataset:
    frame = load_skab_csv(path)
    if config.feature_mode == 'raw':
        return build_sensor_windows(
            frame,
            window_size=config.window_size,
            stride=config.stride,
            sensor_columns=SENSOR_COLUMNS,
        )

    features, labels, changepoints, timestamps = build_spectral_window_features(
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


def _load_windows_multi(paths: list[Path], config: IsoForestTrainingConfig) -> WindowedSensorDataset:
    datasets = [_load_windows(p, config) for p in paths]
    if not datasets:
        return WindowedSensorDataset(
            features=np.empty((0, _feature_count(config.feature_mode, config.window_size, len(SENSOR_COLUMNS))), dtype=float),
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
        features=np.vstack([d.features for d in datasets]),
        labels=np.concatenate([d.labels for d in datasets]),
        changepoints=np.concatenate([d.changepoints for d in datasets]),
        timestamps=np.concatenate([d.timestamps for d in datasets]),
        sensor_columns=datasets[0].sensor_columns,
        window_size=datasets[0].window_size,
        stride=datasets[0].stride,
    )


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


def _anomaly_scores(model: IsolationForest, features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.empty((0,), dtype=float)
    return -np.asarray(model.score_samples(features), dtype=np.float64)


def _artifact_paths(output_dir: Path, has_scaler: bool, has_manifest: bool, has_test: bool) -> dict[str, Path]:
    paths = {
        'model': output_dir / 'isoforest.joblib',
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


def _scores_frame(
    dataset: WindowedSensorDataset,
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> Any:
    pd = import_module('pandas')
    return pd.DataFrame(
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


def _dump_joblib(value: Any, path: Path) -> None:
    joblib = import_module('joblib')
    joblib.dump(value, path)


def _params(config: IsoForestTrainingConfig, training_windows: WindowedSensorDataset, threshold: float) -> dict[str, Any]:
    params = {
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
        'log_mlflow': config.log_mlflow,
        'register_model': config.register_model,
        'registered_model_name': config.registered_model_name,
        'alias': config.alias,
        'n_estimators': config.n_estimators,
        'contamination': config.contamination,
        'seed': config.seed,
        'sensor_columns': list(training_windows.sensor_columns),
        'sensor_count': len(training_windows.sensor_columns),
        'feature_count': int(training_windows.features.shape[1]),
    }
    if config.feature_mode == 'spectral':
        params['spectral_n_bands'] = _SPECTRAL_N_BANDS
    return params


def _metadata_payload(
    config: IsoForestTrainingConfig,
    result: IsoForestTrainingResult,
    provenance_input_files: Sequence[Path],
    manifest: SkabSplitManifest | None,
    train_windows: WindowedSensorDataset,
    validation_windows: WindowedSensorDataset,
    test_windows: WindowedSensorDataset | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'model_family': 'isolation_forest',
        'params': result.params,
        'metrics': result.metrics,
        'thresholds': result.thresholds,
        'sensor_columns': list(result.sensor_columns),
        'artifact_paths': {name: str(path) for name, path in result.artifact_paths.items()},
        'config': _jsonable_config(config),
        'metric_protocol': _METRIC_PROTOCOL,
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


def _non_split_input_files(config: IsoForestTrainingConfig) -> list[Path]:
    paths = [config.input_path]
    if config.validation_input_path is not None:
        paths.append(config.validation_input_path)
    return _unique_paths(paths)


def _split_input_files(config: IsoForestTrainingConfig, manifest: SkabSplitManifest) -> list[Path]:
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
    config: IsoForestTrainingConfig,
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
        train_df=skab_window_dataframe(train_windows, normal_only=True),
        val_df=skab_window_dataframe(validation_windows),
        test_df=skab_window_dataframe(test_windows) if test_windows is not None else None,
        manifest_path=config.split_manifest_path or config.input_path,
        manifest_dict=manifest.to_payload() if manifest is not None else None,
        feature_mode=config.feature_mode,
        split_strategy='unsupervised_novel_fault' if config.split_manifest_path is not None else 'single_csv',
    )


def _feature_count(feature_mode: str, window_size: int, sensor_count: int) -> int:
    if feature_mode == 'spectral':
        return sensor_count * (_SPECTRAL_N_BANDS + 4)
    return window_size * sensor_count


def _validated_threshold_quantile(value: float) -> float:
    quantile = float(value)
    if not 0.0 < quantile < 1.0:
        raise ValueError('threshold_quantile must be between 0 and 1, exclusive')
    return quantile


def _validated_n_estimators(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError('n_estimators must be a positive integer')
    return value


def _validated_contamination(value: str | float) -> str | float:
    if isinstance(value, str):
        if value.lower() == 'auto':
            return 'auto'
        value = float(value)
    contamination = float(value)
    if not 0.0 < contamination <= 0.5:
        raise ValueError("contamination must be 'auto' or a float in (0, 0.5]")
    return contamination


def _jsonable_config(config: IsoForestTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ('input_path', 'output_dir', 'validation_input_path', 'split_manifest_path'):
        value = payload[key]
        payload[key] = str(value) if value is not None else None
    return payload


def _result_payload(result: IsoForestTrainingResult) -> dict[str, Any]:
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
