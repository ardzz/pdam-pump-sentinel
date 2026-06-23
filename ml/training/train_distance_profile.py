from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ml.datasets.skab_loader import SENSOR_COLUMNS, load_skab_csv
from ml.datasets.skab_manifest import SkabSplitManifest, load_skab_split_manifest
from ml.evaluation.metrics import evaluate_split
from ml.features.windowing import WindowedSensorDataset, build_sensor_windows
from ml.training.train_isoforest import (
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

_REFERENCE_SELECTIONS = {'evenly_spaced'}


@dataclass(frozen=True)
class DistanceProfileTrainingConfig:
    input_path: Path
    output_dir: Path
    validation_input_path: Path | None = None
    split_manifest_path: Path | None = None
    window_size: int = 60
    stride: int = 1
    threshold_quantile: float = 0.95
    scaler: str | None = 'standard'
    max_reference_windows: int = 512
    max_distance_comparisons: int = 2_000_000
    reference_selection: str = 'evenly_spaced'


@dataclass(frozen=True)
class DistanceProfileTrainingResult:
    output_dir: Path
    artifact_paths: dict[str, Path]
    params: dict[str, Any]
    metrics: dict[str, Any]
    thresholds: dict[str, float]
    sensor_columns: tuple[str, ...]
    input_example: Any | None = None
    output_example: Any | None = None


@dataclass
class NearestNormalDistanceProfileDetector:
    max_distance_comparisons: int
    distance_normalization: str = 'root_mean_square'
    reference_windows_: np.ndarray | None = None

    def fit(self, reference_windows: np.ndarray) -> NearestNormalDistanceProfileDetector:
        references = _validated_2d_array(reference_windows, 'reference_windows')
        if len(references) == 0:
            raise ValueError('reference_windows must contain at least one normal window')
        self.reference_windows_ = references.astype(np.float64, copy=True)
        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        if self.reference_windows_ is None:
            raise ValueError('detector must be fit before scoring')
        return _nearest_reference_distances(
            features,
            self.reference_windows_,
            max_distance_comparisons=self.max_distance_comparisons,
        )

    def predict(self, features: np.ndarray, threshold: float) -> np.ndarray:
        return (self.score_samples(features) > float(threshold)).astype(int)


def train_distance_profile_from_skab(config: DistanceProfileTrainingConfig) -> DistanceProfileTrainingResult:
    normalized_config = _normalize_config(config)
    result, _detector = _fit_and_write_artifacts(normalized_config)
    return result


def main(argv: Sequence[str] | None = None) -> DistanceProfileTrainingResult:
    config = _parse_args(argv)
    result = train_distance_profile_from_skab(config)
    print(json.dumps(_result_payload(result), sort_keys=True))
    return result


def _fit_and_write_artifacts(
    config: DistanceProfileTrainingConfig,
) -> tuple[DistanceProfileTrainingResult, NearestNormalDistanceProfileDetector]:
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


def _fit_and_write_artifacts_split(
    config: DistanceProfileTrainingConfig,
) -> tuple[DistanceProfileTrainingResult, NearestNormalDistanceProfileDetector]:
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
    config: DistanceProfileTrainingConfig,
    train_windows: WindowedSensorDataset,
    validation_windows: WindowedSensorDataset,
    test_windows: WindowedSensorDataset | None,
    provenance_input_files: Sequence[Path],
    manifest: SkabSplitManifest | None = None,
) -> tuple[DistanceProfileTrainingResult, NearestNormalDistanceProfileDetector]:
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
    reference_x = _select_reference_windows(train_normal_x, config.max_reference_windows)

    detector = NearestNormalDistanceProfileDetector(max_distance_comparisons=config.max_distance_comparisons).fit(
        reference_x
    )
    validation_normal_scores = detector.score_samples(validation_normal_x)
    threshold = float(np.percentile(validation_normal_scores, config.threshold_quantile * 100.0))
    thresholds = {'threshold': threshold, 't2_threshold': threshold, 'q_threshold': threshold}

    validation_scores = detector.score_samples(validation_x)
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
        'reference_window_count': int(len(reference_x)),
        'train_count': int(len(train_windows.labels)),
        'validation_count': int(len(validation_windows.labels)),
        **thresholds,
    }

    has_test = test_windows is not None and test_x is not None and len(test_windows.features) > 0
    if has_test and test_windows is not None:
        metrics['test_count'] = int(len(test_windows.labels))

    params = _params(config, train_windows, threshold, reference_count=len(reference_x))
    artifact_paths = _artifact_paths(config.output_dir, scaler is not None, manifest is not None, has_test)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    _dump_joblib(detector, artifact_paths['model'])
    if scaler is not None:
        _dump_joblib(scaler, artifact_paths['scaler'])
    _scores_frame(validation_windows, validation_labels, validation_predictions, validation_scores).to_csv(
        artifact_paths['scores'], index=False
    )

    if has_test and test_windows is not None and test_x is not None:
        test_scores = detector.score_samples(test_x)
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

    input_example = _input_example(train_normal_x)
    result = DistanceProfileTrainingResult(
        output_dir=config.output_dir,
        artifact_paths=artifact_paths,
        params=params,
        metrics=metrics,
        thresholds=thresholds,
        sensor_columns=train_windows.sensor_columns,
        input_example=input_example,
        output_example=_output_example(detector, input_example),
    )
    _write_json(artifact_paths['metrics'], metrics)
    _write_json(
        artifact_paths['metadata'],
        _metadata_payload(config, result, provenance_input_files, manifest, train_windows, validation_windows, test_windows),
    )
    return result, detector


def _parse_args(argv: Sequence[str] | None) -> DistanceProfileTrainingConfig:
    parser = argparse.ArgumentParser(description='Train a bounded nearest-normal distance-profile detector from SKAB CSV telemetry.')
    parser.add_argument('paths', type=Path, nargs='+', metavar='PATH')
    parser.add_argument('--validation-input-path', type=Path, default=None)
    parser.add_argument('--split-manifest', type=Path, default=None)
    parser.add_argument('--window-size', type=int, default=60)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--threshold-quantile', type=float, default=0.95)
    parser.add_argument('--scaler', default='standard')
    parser.add_argument('--max-reference-windows', type=int, default=512)
    parser.add_argument('--max-distance-comparisons', type=int, default=2_000_000)
    parser.add_argument('--reference-selection', choices=sorted(_REFERENCE_SELECTIONS), default='evenly_spaced')
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
    return DistanceProfileTrainingConfig(**args_dict)


def _normalize_config(config: DistanceProfileTrainingConfig) -> DistanceProfileTrainingConfig:
    validation_input_path = Path(config.validation_input_path) if config.validation_input_path is not None else None
    split_manifest_path = Path(config.split_manifest_path) if config.split_manifest_path is not None else None
    _validated_threshold_quantile(config.threshold_quantile)
    _validated_positive_integer(config.window_size, 'window_size')
    _validated_positive_integer(config.stride, 'stride')
    max_reference_windows = _validated_positive_integer(config.max_reference_windows, 'max_reference_windows')
    max_distance_comparisons = _validated_positive_integer(config.max_distance_comparisons, 'max_distance_comparisons')
    reference_selection = config.reference_selection.lower()
    if reference_selection not in _REFERENCE_SELECTIONS:
        raise ValueError(f'reference_selection must be one of: {", ".join(sorted(_REFERENCE_SELECTIONS))}')
    return replace(
        config,
        input_path=Path(config.input_path),
        output_dir=Path(config.output_dir),
        validation_input_path=validation_input_path,
        split_manifest_path=split_manifest_path,
        max_reference_windows=max_reference_windows,
        max_distance_comparisons=max_distance_comparisons,
        reference_selection=reference_selection,
    )


def _load_windows(path: Path, config: DistanceProfileTrainingConfig) -> WindowedSensorDataset:
    frame = load_skab_csv(path)
    return build_sensor_windows(
        frame,
        window_size=config.window_size,
        stride=config.stride,
        sensor_columns=SENSOR_COLUMNS,
    )


def _load_windows_multi(paths: list[Path], config: DistanceProfileTrainingConfig) -> WindowedSensorDataset:
    datasets = [_load_windows(p, config) for p in paths]
    if not datasets:
        return WindowedSensorDataset(
            features=np.empty((0, _feature_count(config.window_size, len(SENSOR_COLUMNS))), dtype=float),
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


def _select_reference_windows(features: np.ndarray, max_reference_windows: int) -> np.ndarray:
    values = _validated_2d_array(features, 'features')
    if len(values) <= max_reference_windows:
        return values.astype(np.float64, copy=True)
    indices = np.linspace(0, len(values) - 1, num=max_reference_windows, dtype=int)
    return values[indices].astype(np.float64, copy=True)


def _nearest_reference_distances(
    features: np.ndarray,
    reference_windows: np.ndarray,
    *,
    max_distance_comparisons: int,
) -> np.ndarray:
    values = _validated_2d_array(features, 'features')
    references = _validated_2d_array(reference_windows, 'reference_windows')
    if values.shape[1] != references.shape[1]:
        raise ValueError('features and reference_windows must have the same number of columns')
    if len(values) == 0:
        return np.empty((0,), dtype=np.float64)
    if len(references) == 0:
        raise ValueError('reference_windows must contain at least one normal window')

    batch_size = max(1, max_distance_comparisons // len(references))
    reference_norms = np.sum(references * references, axis=1).reshape(1, -1)
    distances = np.empty((len(values),), dtype=np.float64)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        batch = values[start:stop]
        batch_norms = np.sum(batch * batch, axis=1).reshape(-1, 1)
        squared_distances = np.maximum(batch_norms + reference_norms - 2.0 * batch @ references.T, 0.0)
        distances[start:stop] = np.sqrt(np.min(squared_distances, axis=1)) / np.sqrt(values.shape[1])
    return distances


def _artifact_paths(output_dir: Path, has_scaler: bool, has_manifest: bool, has_test: bool) -> dict[str, Path]:
    paths = {
        'model': output_dir / 'distance_profile_nearest_normal.joblib',
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


def _params(
    config: DistanceProfileTrainingConfig,
    training_windows: WindowedSensorDataset,
    threshold: float,
    *,
    reference_count: int,
) -> dict[str, Any]:
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
        'feature_mode': 'raw_window_nearest_normal_distance',
        'model_type': 'nearest_normal_window_distance_profile',
        'score_orientation': 'higher_is_more_anomalous:nearest_train_normal_window_distance',
        'threshold_calibration': 'validation_normal_quantile',
        'reference_selection': config.reference_selection,
        'reference_window_count': int(reference_count),
        'max_reference_windows': config.max_reference_windows,
        'max_distance_comparisons_per_batch': config.max_distance_comparisons,
        'dependency_status': 'implemented_with_existing_numpy_sklearn_only_no_stumpy_dependency',
        'sensor_columns': list(training_windows.sensor_columns),
        'sensor_count': len(training_windows.sensor_columns),
        'feature_count': int(training_windows.features.shape[1]),
    }


def _metadata_payload(
    config: DistanceProfileTrainingConfig,
    result: DistanceProfileTrainingResult,
    provenance_input_files: Sequence[Path],
    manifest: SkabSplitManifest | None,
    train_windows: WindowedSensorDataset,
    validation_windows: WindowedSensorDataset,
    test_windows: WindowedSensorDataset | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'model_family': 'distance_profile_nearest_normal',
        'params': result.params,
        'metrics': result.metrics,
        'thresholds': result.thresholds,
        'sensor_columns': list(result.sensor_columns),
        'artifact_paths': {name: str(path) for name, path in result.artifact_paths.items()},
        'config': _jsonable_config(config),
        'metric_protocol': _METRIC_PROTOCOL,
        'score_orientation': result.params['score_orientation'],
        'threshold_calibration': 'threshold calibrated from validation-normal nearest-distance scores only',
        'train_label_policy': 'fit nearest-normal reference set on manifest train windows with anomaly label 0 only',
        'runtime_guard': {
            'reference_selection': config.reference_selection,
            'max_reference_windows': config.max_reference_windows,
            'reference_window_count': result.params['reference_window_count'],
            'max_distance_comparisons_per_batch': config.max_distance_comparisons,
        },
        'dependency_status': result.params['dependency_status'],
        'test_metrics_policy': 'held-out test split used only for final test_ metrics after calibration',
        'test_labels_used_for_training_or_threshold': False,
        'test_scores_used_for_training_or_threshold': False,
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


def _non_split_input_files(config: DistanceProfileTrainingConfig) -> list[Path]:
    paths = [config.input_path]
    if config.validation_input_path is not None:
        paths.append(config.validation_input_path)
    return _unique_paths(paths)


def _split_input_files(config: DistanceProfileTrainingConfig, manifest: SkabSplitManifest) -> list[Path]:
    paths: list[Path] = []
    if config.split_manifest_path is not None:
        paths.append(config.split_manifest_path)
    paths.extend([*manifest.train, *manifest.validation, *manifest.test])
    return _unique_paths(paths)


def _feature_count(window_size: int, sensor_count: int) -> int:
    return window_size * sensor_count


def _validated_2d_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f'{name} must be a 2D array')
    return array


def _validated_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _output_example(detector: NearestNormalDistanceProfileDetector, input_example: np.ndarray | None) -> np.ndarray | None:
    if input_example is None:
        return None
    return detector.score_samples(input_example)


def _jsonable_config(config: DistanceProfileTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ('input_path', 'output_dir', 'validation_input_path', 'split_manifest_path'):
        value = payload[key]
        payload[key] = str(value) if value is not None else None
    return payload


def _result_payload(result: DistanceProfileTrainingResult) -> dict[str, Any]:
    return {
        'output_dir': str(result.output_dir),
        'artifact_paths': {name: str(path) for name, path in result.artifact_paths.items()},
        'metrics': result.metrics,
        'thresholds': result.thresholds,
        'sensor_columns': list(result.sensor_columns),
    }


if __name__ == '__main__':
    main()
