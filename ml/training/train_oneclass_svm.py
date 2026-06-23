from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.svm import OneClassSVM

from ml.datasets.skab_loader import SENSOR_COLUMNS, load_skab_csv
from ml.datasets.skab_manifest import SkabSplitManifest, load_skab_split_manifest
from ml.evaluation.metrics import evaluate_split
from ml.features.spectral import build_spectral_window_features
from ml.features.windowing import WindowedSensorDataset, build_sensor_windows
from ml.training.train_isoforest import (
    _FEATURE_MODES,
    _SPECTRAL_N_BANDS,
    _count_metrics,
    _dump_joblib,
    _fit_transform_scaler,
    _input_example,
    _make_scaler,
    _scores_frame,
    _transform_scaler,
    _unique_paths,
    _validated_threshold_quantile,
    _write_json,
)
from ml.training.train_pca import _METRIC_PROTOCOL
from ml.utils.provenance import collect_provenance

_KERNELS = {'linear', 'poly', 'rbf', 'sigmoid'}


@dataclass(frozen=True)
class OneClassSvmTrainingConfig:
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
    kernel: str = 'rbf'
    nu: float = 0.05
    gamma: str | float = 'scale'
    degree: int = 3
    coef0: float = 0.0
    shrinking: bool = True
    tol: float = 1e-3
    cache_size: float = 128.0
    max_iter: int = 1000


@dataclass(frozen=True)
class OneClassSvmTrainingResult:
    output_dir: Path
    artifact_paths: dict[str, Path]
    params: dict[str, Any]
    metrics: dict[str, Any]
    thresholds: dict[str, float]
    sensor_columns: tuple[str, ...]
    input_example: Any | None = None
    output_example: Any | None = None


def train_oneclass_svm_from_skab(config: OneClassSvmTrainingConfig) -> OneClassSvmTrainingResult:
    normalized_config = _normalize_config(config)
    result, model = _fit_and_write_artifacts(normalized_config)
    return result


def main(argv: Sequence[str] | None = None) -> OneClassSvmTrainingResult:
    config = _parse_args(argv)
    result = train_oneclass_svm_from_skab(config)
    print(json.dumps(_result_payload(result), sort_keys=True))
    return result


def _fit_and_write_artifacts(config: OneClassSvmTrainingConfig) -> tuple[OneClassSvmTrainingResult, OneClassSVM]:
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


def _fit_and_write_artifacts_split(config: OneClassSvmTrainingConfig) -> tuple[OneClassSvmTrainingResult, OneClassSVM]:
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
    config: OneClassSvmTrainingConfig,
    train_windows: WindowedSensorDataset,
    validation_windows: WindowedSensorDataset,
    test_windows: WindowedSensorDataset | None,
    provenance_input_files: Sequence[Path],
    manifest: SkabSplitManifest | None = None,
) -> tuple[OneClassSvmTrainingResult, OneClassSVM]:
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

    model_params = _model_params(config)
    model = OneClassSVM(**model_params)
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

    result = OneClassSvmTrainingResult(
        output_dir=config.output_dir,
        artifact_paths=artifact_paths,
        params=params,
        metrics=metrics,
        thresholds=thresholds,
        sensor_columns=train_windows.sensor_columns,
        input_example=_input_example(train_normal_x),
        output_example=_output_example(model, _input_example(train_normal_x)),
    )
    _write_json(artifact_paths['metrics'], metrics)
    _write_json(
        artifact_paths['metadata'],
        _metadata_payload(config, result, provenance_input_files, manifest, train_windows, validation_windows, test_windows),
    )
    return result, model


def _parse_args(argv: Sequence[str] | None) -> OneClassSvmTrainingConfig:
    parser = argparse.ArgumentParser(description='Train a One-Class SVM novelty detector from SKAB CSV telemetry.')
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
    parser.add_argument('--kernel', choices=sorted(_KERNELS), default='rbf')
    parser.add_argument('--nu', type=float, default=0.05)
    parser.add_argument('--gamma', type=_parse_gamma, default='scale')
    parser.add_argument('--degree', type=int, default=3)
    parser.add_argument('--coef0', type=float, default=0.0)
    parser.add_argument('--shrinking', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument('--cache-size', type=float, default=128.0)
    parser.add_argument('--max-iter', type=int, default=1000)
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
    return OneClassSvmTrainingConfig(**args_dict)


def _parse_gamma(value: str) -> str | float:
    lower_value = value.lower()
    if lower_value in {'scale', 'auto'}:
        return lower_value
    return float(value)


def _normalize_config(config: OneClassSvmTrainingConfig) -> OneClassSvmTrainingConfig:
    validation_input_path = Path(config.validation_input_path) if config.validation_input_path is not None else None
    split_manifest_path = Path(config.split_manifest_path) if config.split_manifest_path is not None else None
    feature_mode = config.feature_mode.lower()
    if feature_mode not in _FEATURE_MODES:
        raise ValueError(f'feature_mode must be one of: {", ".join(sorted(_FEATURE_MODES))}')
    kernel = config.kernel.lower()
    if kernel not in _KERNELS:
        raise ValueError(f'kernel must be one of: {", ".join(sorted(_KERNELS))}')
    _validated_threshold_quantile(config.threshold_quantile)
    nu = _validated_nu(config.nu)
    gamma = _validated_gamma(config.gamma)
    _validated_degree(config.degree)
    _validated_positive_float(config.tol, 'tol')
    _validated_positive_float(config.cache_size, 'cache_size')
    _validated_max_iter(config.max_iter)
    return replace(
        config,
        input_path=Path(config.input_path),
        output_dir=Path(config.output_dir),
        validation_input_path=validation_input_path,
        split_manifest_path=split_manifest_path,
        feature_mode=feature_mode,
        kernel=kernel,
        nu=nu,
        gamma=gamma,
    )


def _model_params(config: OneClassSvmTrainingConfig) -> dict[str, Any]:
    return {
        'kernel': config.kernel,
        'nu': config.nu,
        'gamma': config.gamma,
        'degree': config.degree,
        'coef0': config.coef0,
        'shrinking': config.shrinking,
        'tol': config.tol,
        'cache_size': config.cache_size,
        'max_iter': config.max_iter,
    }


def _load_windows(path: Path, config: OneClassSvmTrainingConfig) -> WindowedSensorDataset:
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


def _load_windows_multi(paths: list[Path], config: OneClassSvmTrainingConfig) -> WindowedSensorDataset:
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


def _anomaly_scores(model: OneClassSVM, features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.empty((0,), dtype=float)
    return -np.asarray(model.decision_function(features), dtype=np.float64).reshape(-1)


def _artifact_paths(output_dir: Path, has_scaler: bool, has_manifest: bool, has_test: bool) -> dict[str, Path]:
    paths = {
        'model': output_dir / 'oneclass_svm.joblib',
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


def _params(config: OneClassSvmTrainingConfig, training_windows: WindowedSensorDataset, threshold: float) -> dict[str, Any]:
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
        **_model_params(config),
        'score_orientation': 'higher_is_more_anomalous:-decision_function',
        'threshold_calibration': 'validation_normal_quantile',
        'sensor_columns': list(training_windows.sensor_columns),
        'sensor_count': len(training_windows.sensor_columns),
        'feature_count': int(training_windows.features.shape[1]),
    }
    if config.feature_mode == 'spectral':
        params['spectral_n_bands'] = _SPECTRAL_N_BANDS
    return params


def _metadata_payload(
    config: OneClassSvmTrainingConfig,
    result: OneClassSvmTrainingResult,
    provenance_input_files: Sequence[Path],
    manifest: SkabSplitManifest | None,
    train_windows: WindowedSensorDataset,
    validation_windows: WindowedSensorDataset,
    test_windows: WindowedSensorDataset | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'model_family': 'oneclass_svm',
        'params': result.params,
        'metrics': result.metrics,
        'thresholds': result.thresholds,
        'sensor_columns': list(result.sensor_columns),
        'artifact_paths': {name: str(path) for name, path in result.artifact_paths.items()},
        'config': _jsonable_config(config),
        'metric_protocol': _METRIC_PROTOCOL,
        'score_orientation': 'higher_is_more_anomalous:-decision_function',
        'threshold_calibration': 'threshold calibrated from validation-normal scores only',
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


def _non_split_input_files(config: OneClassSvmTrainingConfig) -> list[Path]:
    paths = [config.input_path]
    if config.validation_input_path is not None:
        paths.append(config.validation_input_path)
    return _unique_paths(paths)


def _split_input_files(config: OneClassSvmTrainingConfig, manifest: SkabSplitManifest) -> list[Path]:
    paths: list[Path] = []
    if config.split_manifest_path is not None:
        paths.append(config.split_manifest_path)
    paths.extend([*manifest.train, *manifest.validation, *manifest.test])
    return _unique_paths(paths)


def _validated_nu(value: float) -> float:
    nu = float(value)
    if not 0.0 < nu <= 1.0:
        raise ValueError('nu must be in the interval (0, 1]')
    return nu


def _validated_gamma(value: str | float) -> str | float:
    if isinstance(value, str):
        gamma = value.lower()
        if gamma in {'scale', 'auto'}:
            return gamma
        value = float(value)
    return _validated_positive_float(float(value), 'gamma')


def _validated_degree(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError('degree must be a positive integer')
    return value


def _validated_positive_float(value: float, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f'{name} must be a positive finite float')
    return parsed


def _validated_max_iter(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (value <= 0 and value != -1):
        raise ValueError('max_iter must be -1 or a positive integer')
    return value


def _feature_count(feature_mode: str, window_size: int, sensor_count: int) -> int:
    if feature_mode == 'spectral':
        return sensor_count * (_SPECTRAL_N_BANDS + 4)
    return window_size * sensor_count


def _output_example(model: OneClassSVM, input_example: np.ndarray | None) -> np.ndarray | None:
    if input_example is None:
        return None
    return _anomaly_scores(model, input_example)


def _jsonable_config(config: OneClassSvmTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ('input_path', 'output_dir', 'validation_input_path', 'split_manifest_path'):
        value = payload[key]
        payload[key] = str(value) if value is not None else None
    return payload


def _result_payload(result: OneClassSvmTrainingResult) -> dict[str, Any]:
    return {
        'output_dir': str(result.output_dir),
        'artifact_paths': {name: str(path) for name, path in result.artifact_paths.items()},
        'metrics': result.metrics,
        'thresholds': result.thresholds,
        'sensor_columns': list(result.sensor_columns),
    }


if __name__ == '__main__':
    main()
