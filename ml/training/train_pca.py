from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from ml.datasets.skab_loader import SENSOR_COLUMNS, load_skab_csv
from ml.datasets.skab_manifest import SkabSplitManifest, load_skab_split_manifest
from ml.evaluation.metrics import evaluate_split
from ml.features.enriched import build_enriched_window_features
from ml.features.spectral import build_spectral_window_features
from ml.features.windowing import WindowedSensorDataset, build_sensor_windows
from ml.training.pca_detector import PcaT2QDetector
from ml.utils.provenance import collect_provenance

_FEATURE_MODES = {'raw', 'spectral', 'enriched'}
_SPECTRAL_N_BANDS = 4
_ENRICHED_DOMAIN_FEATURE_COUNT = 6

_METRIC_PROTOCOL = {
    'schema_version': 1,
    'label_source': 'anomaly',
    'window_label_rule': 'label=1 when any nonzero anomaly occurs inside the window',
    'changepoint_source': 'changepoint',
    'window_changepoint_rule': 'changepoint=1 when any nonzero changepoint occurs inside the window',
    'changepoint_role': 'transient_mask_only',
    'split_evaluator': 'ml.evaluation.metrics.evaluate_split',
    'validation_metrics': 'top-level metrics are validation metrics for split-manifest training',
    'test_metrics_prefix': 'test_',
    'threshold_calibration': 'split-manifest thresholds use validation windows with anomaly label 0 only',
    'threshold_free_metrics': ['pr_auc', 'roc_auc'],
    'event_metrics': ['event_count', 'event_recall', 'missed_events', 'false_alarm_events', 'mean_detection_delay_windows'],
    'transient_exclusion_suffix': '_excluding_transient',
    'point_adjusted_metrics': False,
    'nab_metrics': False,
}


@dataclass(frozen=True)
class PcaTrainingConfig:
    input_path: Path
    output_dir: Path
    validation_input_path: Path | None = None
    split_manifest_path: Path | None = None
    window_size: int = 60
    stride: int = 1
    n_components: int | float | None = 0.9
    threshold_quantile: float = 0.95
    scaler: str | None = 'robust'
    log_mlflow: bool = False
    register_model: bool = False
    registered_model_name: str = 'PumpAD'
    alias: str | None = None
    feature_mode: str = 'raw'


@dataclass(frozen=True)
class PcaTrainingResult:
    output_dir: Path
    artifact_paths: dict[str, Path]
    params: dict[str, Any]
    metrics: dict[str, Any]
    thresholds: dict[str, float]
    sensor_columns: tuple[str, ...]
    input_example: Any | None = None
    output_example: Any | None = None


def train_pca_from_skab(config: PcaTrainingConfig) -> PcaTrainingResult:
    normalized_config = _normalize_config(config)
    result, detector = _fit_and_write_artifacts(normalized_config)

    if normalized_config.log_mlflow or _active_mlflow_run_exists():
        from ml.registry.mlflow_client import log_pca_training_run

        log_pca_training_run(result, detector, normalized_config)

    return result


def main(argv: Sequence[str] | None = None) -> PcaTrainingResult:
    config = _parse_args(argv)
    result = train_pca_from_skab(config)
    print(json.dumps(_result_payload(result), sort_keys=True))
    return result


def _fit_and_write_artifacts(config: PcaTrainingConfig) -> tuple[PcaTrainingResult, PcaT2QDetector]:
    if config.split_manifest_path is not None:
        return _fit_and_write_artifacts_split(config)

    training_windows = _load_windows(config.input_path, config)
    validation_path = config.validation_input_path or config.input_path
    scoring_windows = training_windows if validation_path == config.input_path else _load_windows(validation_path, config)

    normal_mask = training_windows.labels == 0
    normal_features = training_windows.features[normal_mask]
    if len(normal_features) == 0:
        raise ValueError('training input must contain at least one normal window')
    if len(scoring_windows.features) == 0:
        raise ValueError('scoring input must produce at least one window')

    detector = PcaT2QDetector(
        n_components=config.n_components,
        threshold_quantile=config.threshold_quantile,
        scaler=config.scaler,
    ).fit(normal_features)

    statistics = detector.transform(scoring_windows.features)
    scores = detector.score_samples(scoring_windows.features)
    predictions = detector.predict(scoring_windows.features)
    labels = scoring_windows.labels.astype(int, copy=False)
    thresholds = {
        't2_threshold': float(detector.t2_threshold_),
        'q_threshold': float(detector.q_threshold_),
    }
    metrics = evaluate_split(labels, predictions, scores, transient_mask=scoring_windows.changepoints) | {
        **_count_metrics(labels, scoring_windows.changepoints),
        'training_sample_count': int(len(training_windows.labels)),
        'training_normal_count': int(len(normal_features)),
        **thresholds,
    }
    params = _params(config, training_windows)

    artifact_paths = _artifact_paths(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    scores_frame = _scores_frame(scoring_windows, labels, predictions, statistics, scores)

    _dump_detector(detector, artifact_paths['detector'])
    scores_frame.to_csv(artifact_paths['scores'], index=False)

    result = PcaTrainingResult(
        output_dir=config.output_dir,
        artifact_paths=artifact_paths,
        params=params,
        metrics=metrics,
        thresholds=thresholds,
        sensor_columns=training_windows.sensor_columns,
        input_example=_input_example(normal_features),
        output_example=_output_example(detector, _input_example(normal_features)),
    )
    _write_json(artifact_paths['metrics'], metrics)
    _write_json(artifact_paths['metadata'], _metadata_payload(config, result, _non_split_input_files(config)))
    _log_skab_inputs_to_active_run(config, None, training_windows, scoring_windows, None)
    return result, detector


def _fit_and_write_artifacts_split(config: PcaTrainingConfig) -> tuple[PcaTrainingResult, PcaT2QDetector]:
    split_manifest_path = config.split_manifest_path
    if split_manifest_path is None:
        raise ValueError('split_manifest_path is required for split-manifest training')
    manifest = load_skab_split_manifest(split_manifest_path)

    train_windows = _load_windows_multi(manifest.train, config)
    validation_windows = _load_windows_multi(manifest.validation, config)
    test_windows = _load_windows_multi(manifest.test, config) if manifest.test else None

    train_normal_mask = train_windows.labels == 0
    train_normal_features = train_windows.features[train_normal_mask]
    if len(train_normal_features) == 0:
        raise ValueError('training input must contain at least one normal window')
    if len(validation_windows.features) == 0:
        raise ValueError('validation input must produce at least one window')

    detector = PcaT2QDetector(
        n_components=config.n_components,
        threshold_quantile=config.threshold_quantile,
        scaler=config.scaler,
    ).fit(train_normal_features)

    validation_normal_mask = validation_windows.labels == 0
    validation_normal_features = validation_windows.features[validation_normal_mask]
    if len(validation_normal_features) == 0:
        raise ValueError('validation input must contain at least one normal window')
    detector.calibrate_thresholds(validation_normal_features)

    thresholds = {
        't2_threshold': float(detector.t2_threshold_),
        'q_threshold': float(detector.q_threshold_),
    }

    val_statistics = detector.transform(validation_windows.features)
    val_scores = detector.score_samples(validation_windows.features)
    val_predictions = detector.predict(validation_windows.features)
    val_labels = validation_windows.labels.astype(int, copy=False)

    val_metrics = evaluate_split(
        val_labels,
        val_predictions,
        val_scores,
        transient_mask=validation_windows.changepoints,
    )

    metrics = val_metrics | {
        **_count_metrics(val_labels, validation_windows.changepoints),
        'training_sample_count': int(len(train_windows.labels)),
        'training_normal_count': int(len(train_normal_features)),
        'train_count': int(len(train_windows.labels)),
        'validation_count': int(len(validation_windows.labels)),
        **thresholds,
    }

    if test_windows is not None and len(test_windows.features) > 0:
        metrics['test_count'] = int(len(test_windows.labels))

    params = _params(config, train_windows)

    artifact_paths = _artifact_paths_split(config.output_dir, test_windows is not None and len(test_windows.features) > 0)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    val_scores_frame = _scores_frame(validation_windows, val_labels, val_predictions, val_statistics, val_scores)

    _dump_detector(detector, artifact_paths['detector'])
    val_scores_frame.to_csv(artifact_paths['scores'], index=False)

    if test_windows is not None and len(test_windows.features) > 0:
        test_statistics = detector.transform(test_windows.features)
        test_scores = detector.score_samples(test_windows.features)
        test_predictions = detector.predict(test_windows.features)
        test_labels = test_windows.labels.astype(int, copy=False)
        test_metrics = evaluate_split(
            test_labels,
            test_predictions,
            test_scores,
            transient_mask=test_windows.changepoints,
        )
        metrics.update({f'test_{name}': value for name, value in test_metrics.items()})
        metrics.update(_count_metrics(test_labels, test_windows.changepoints, prefix='test_'))
        test_scores_frame = _scores_frame(test_windows, test_labels, test_predictions, test_statistics, test_scores)
        test_scores_frame.to_csv(artifact_paths['test_scores'], index=False)

    _write_json(artifact_paths['split_manifest'], manifest.to_payload())

    result = PcaTrainingResult(
        output_dir=config.output_dir,
        artifact_paths=artifact_paths,
        params=params,
        metrics=metrics,
        thresholds=thresholds,
        sensor_columns=train_windows.sensor_columns,
        input_example=_input_example(train_normal_features),
        output_example=_output_example(detector, _input_example(train_normal_features)),
    )
    _write_json(artifact_paths['metrics'], metrics)
    _write_json(
        artifact_paths['metadata'],
        _metadata_payload_split(config, result, manifest, train_windows, validation_windows, test_windows),
    )
    _log_skab_inputs_to_active_run(config, manifest, train_windows, validation_windows, test_windows)
    return result, detector


def _parse_args(argv: Sequence[str] | None) -> PcaTrainingConfig:
    parser = argparse.ArgumentParser(description='Train a PCA T²/Q detector from SKAB CSV telemetry.')
    parser.add_argument('paths', type=Path, nargs='+', metavar='PATH')
    parser.add_argument('--validation-input-path', type=Path, default=None)
    parser.add_argument('--split-manifest', type=Path, default=None)
    parser.add_argument('--window-size', type=int, default=60)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--n-components', type=_parse_n_components, default=0.9)
    parser.add_argument('--threshold-quantile', type=float, default=0.95)
    parser.add_argument('--scaler', default='robust')
    parser.add_argument('--log-mlflow', action='store_true')
    parser.add_argument('--register-model', action='store_true')
    parser.add_argument('--registered-model-name', default='PumpAD')
    parser.add_argument('--alias', default=None)
    parser.add_argument('--feature-mode', choices=sorted(_FEATURE_MODES), default='raw')
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
    return PcaTrainingConfig(**args_dict)


def _parse_n_components(value: str) -> int | float | None:
    if value.lower() == 'none':
        return None
    parsed_float = float(value)
    if parsed_float.is_integer() and not any(marker in value.lower() for marker in ('.', 'e')):
        return int(parsed_float)
    return parsed_float


def _normalize_config(config: PcaTrainingConfig) -> PcaTrainingConfig:
    validation_input_path = Path(config.validation_input_path) if config.validation_input_path is not None else None
    split_manifest_path = Path(config.split_manifest_path) if config.split_manifest_path is not None else None
    feature_mode = config.feature_mode.lower()
    if feature_mode not in _FEATURE_MODES:
        raise ValueError(f'feature_mode must be one of: {", ".join(sorted(_FEATURE_MODES))}')
    return replace(
        config,
        input_path=Path(config.input_path),
        output_dir=Path(config.output_dir),
        validation_input_path=validation_input_path,
        split_manifest_path=split_manifest_path,
        feature_mode=feature_mode,
    )


def _load_windows(path: Path, config: PcaTrainingConfig) -> WindowedSensorDataset:
    frame = load_skab_csv(path)
    if config.feature_mode == 'raw':
        return build_sensor_windows(
            frame,
            window_size=config.window_size,
            stride=config.stride,
            sensor_columns=SENSOR_COLUMNS,
        )

    if config.feature_mode == 'spectral':
        feature_builder = build_spectral_window_features
    elif config.feature_mode == 'enriched':
        feature_builder = build_enriched_window_features
    else:
        raise ValueError(f'unsupported feature_mode: {config.feature_mode}')

    features, labels, changepoints, timestamps = feature_builder(
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


def _load_windows_multi(paths: list[Path], config: PcaTrainingConfig) -> WindowedSensorDataset:
    datasets = [_load_windows(p, config) for p in paths]
    if not datasets:
        return WindowedSensorDataset(
            features=np.empty((0, _feature_count(config)), dtype=float),
            labels=np.empty((0,), dtype=int),
            changepoints=np.empty((0,), dtype=int),
            timestamps=np.empty((0,), dtype=object),
            sensor_columns=tuple(SENSOR_COLUMNS),
            window_size=config.window_size,
            stride=config.stride,
        )
    return _concat_datasets(datasets)


def _feature_count(config: PcaTrainingConfig) -> int:
    if config.feature_mode == 'spectral':
        return len(SENSOR_COLUMNS) * (_SPECTRAL_N_BANDS + 4)
    if config.feature_mode == 'enriched':
        return len(SENSOR_COLUMNS) * (_SPECTRAL_N_BANDS + 7) + _ENRICHED_DOMAIN_FEATURE_COUNT
    return config.window_size * len(SENSOR_COLUMNS)


def _concat_datasets(datasets: list[WindowedSensorDataset]) -> WindowedSensorDataset:
    if len(datasets) == 1:
        return datasets[0]
    features = np.vstack([d.features for d in datasets])
    labels = np.concatenate([d.labels for d in datasets])
    changepoints = np.concatenate([d.changepoints for d in datasets])
    timestamps = np.concatenate([d.timestamps for d in datasets])
    return WindowedSensorDataset(
        features=features,
        labels=labels,
        changepoints=changepoints,
        timestamps=timestamps,
        sensor_columns=datasets[0].sensor_columns,
        window_size=datasets[0].window_size,
        stride=datasets[0].stride,
    )


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        'detector': output_dir / 'pca_detector.joblib',
        'metadata': output_dir / 'metadata.json',
        'metrics': output_dir / 'metrics.json',
        'scores': output_dir / 'scores.csv',
    }


def _artifact_paths_split(output_dir: Path, has_test: bool) -> dict[str, Path]:
    paths = _artifact_paths(output_dir)
    paths['split_manifest'] = output_dir / 'split_manifest.json'
    if has_test:
        paths['test_scores'] = output_dir / 'test_scores.csv'
    return paths


def _scores_frame(
    dataset: WindowedSensorDataset,
    labels: np.ndarray,
    predictions: np.ndarray,
    statistics: np.ndarray,
    scores: np.ndarray,
) -> Any:
    pd = import_module('pandas')
    return pd.DataFrame(
        {
            'timestamp': dataset.timestamps.astype(str),
            'label': labels.astype(int),
            'changepoint': dataset.changepoints.astype(int),
            'prediction': predictions.astype(int),
            't2': statistics[:, 0].astype(float),
            'q': statistics[:, 1].astype(float),
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


def _dump_detector(detector: PcaT2QDetector, path: Path) -> None:
    joblib = import_module('joblib')
    joblib.dump(detector, path)


def _params(config: PcaTrainingConfig, training_windows: WindowedSensorDataset) -> dict[str, Any]:
    params = {
        'input_path': str(config.input_path),
        'output_dir': str(config.output_dir),
        'validation_input_path': str(config.validation_input_path) if config.validation_input_path is not None else None,
        'split_manifest_path': str(config.split_manifest_path) if config.split_manifest_path is not None else None,
        'window_size': config.window_size,
        'stride': config.stride,
        'n_components': config.n_components,
        'threshold_quantile': config.threshold_quantile,
        'scaler': config.scaler,
        'log_mlflow': config.log_mlflow,
        'register_model': config.register_model,
        'registered_model_name': config.registered_model_name,
        'alias': config.alias,
        'feature_count': int(training_windows.features.shape[1]),
    }
    if config.feature_mode != 'raw':
        params['feature_mode'] = config.feature_mode
        params['spectral_n_bands'] = _SPECTRAL_N_BANDS
    return params


def _metadata_payload(
    config: PcaTrainingConfig,
    result: PcaTrainingResult,
    provenance_input_files: Sequence[Path],
) -> dict[str, Any]:
    return {
        'model_family': 'pca',
        'params': result.params,
        'metrics': result.metrics,
        'thresholds': result.thresholds,
        'sensor_columns': list(result.sensor_columns),
        'artifact_paths': {name: str(path) for name, path in result.artifact_paths.items()},
        'config': _jsonable_config(config),
        'metric_protocol': _METRIC_PROTOCOL,
        'provenance': collect_provenance(config=_jsonable_config(config), input_files=provenance_input_files),
    }


def _metadata_payload_split(
    config: PcaTrainingConfig,
    result: PcaTrainingResult,
    manifest: SkabSplitManifest,
    train_windows: WindowedSensorDataset,
    validation_windows: WindowedSensorDataset,
    test_windows: WindowedSensorDataset | None,
) -> dict[str, Any]:
    payload = _metadata_payload(config, result, _split_input_files(config, manifest))
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


def _non_split_input_files(config: PcaTrainingConfig) -> list[Path]:
    paths = [config.input_path]
    if config.validation_input_path is not None:
        paths.append(config.validation_input_path)
    return _unique_paths(paths)


def _split_input_files(config: PcaTrainingConfig, manifest: SkabSplitManifest) -> list[Path]:
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
    config: PcaTrainingConfig,
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


def _active_mlflow_run_exists() -> bool:
    try:
        mlflow = import_module('mlflow')
        active_run = getattr(mlflow, 'active_run', None)
        return bool(active_run()) if callable(active_run) else False
    except Exception:
        return False


def _input_example(features: np.ndarray) -> np.ndarray | None:
    if len(features) == 0:
        return None
    return np.asarray(features[:5], dtype=np.float64)


def _output_example(detector: PcaT2QDetector, input_example: np.ndarray | None) -> np.ndarray | None:
    if input_example is None:
        return None
    return np.asarray(detector.score_samples(input_example), dtype=np.float64)


def _jsonable_config(config: PcaTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ('input_path', 'output_dir', 'validation_input_path', 'split_manifest_path'):
        value = payload[key]
        payload[key] = str(value) if value is not None else None
    if payload['feature_mode'] == 'raw':
        payload.pop('feature_mode')
    return payload


def _result_payload(result: PcaTrainingResult) -> dict[str, Any]:
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
