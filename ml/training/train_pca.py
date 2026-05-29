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
from ml.features.windowing import WindowedSensorDataset, build_sensor_windows
from ml.training.pca_detector import PcaT2QDetector


@dataclass(frozen=True)
class PcaTrainingConfig:
    input_path: Path
    output_dir: Path
    validation_input_path: Path | None = None
    window_size: int = 60
    stride: int = 1
    n_components: int | float | None = 0.9
    threshold_quantile: float = 0.95
    scaler: str | None = 'standard'
    log_mlflow: bool = False
    register_model: bool = False
    registered_model_name: str = 'PumpAD'
    alias: str | None = None


@dataclass(frozen=True)
class PcaTrainingResult:
    output_dir: Path
    artifact_paths: dict[str, Path]
    params: dict[str, Any]
    metrics: dict[str, int | float]
    thresholds: dict[str, float]
    sensor_columns: tuple[str, ...]


def train_pca_from_skab(config: PcaTrainingConfig) -> PcaTrainingResult:
    normalized_config = _normalize_config(config)
    result, detector = _fit_and_write_artifacts(normalized_config)

    if normalized_config.log_mlflow:
        from ml.registry.mlflow_client import log_pca_training_run

        log_pca_training_run(result, detector, normalized_config)

    return result


def main(argv: Sequence[str] | None = None) -> PcaTrainingResult:
    config = _parse_args(argv)
    result = train_pca_from_skab(config)
    print(json.dumps(_result_payload(result), sort_keys=True))
    return result


def _fit_and_write_artifacts(config: PcaTrainingConfig) -> tuple[PcaTrainingResult, PcaT2QDetector]:
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
    metrics = _classification_metrics(labels, predictions) | {
        'sample_count': int(len(labels)),
        'anomaly_count': int(np.count_nonzero(labels == 1)),
        'normal_count': int(np.count_nonzero(labels == 0)),
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
    )
    _write_json(artifact_paths['metrics'], metrics)
    _write_json(artifact_paths['metadata'], _metadata_payload(config, result))
    return result, detector


def _parse_args(argv: Sequence[str] | None) -> PcaTrainingConfig:
    parser = argparse.ArgumentParser(description='Train a PCA T²/Q detector from SKAB CSV telemetry.')
    parser.add_argument('input_path', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--validation-input-path', type=Path, default=None)
    parser.add_argument('--window-size', type=int, default=60)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--n-components', type=_parse_n_components, default=0.9)
    parser.add_argument('--threshold-quantile', type=float, default=0.95)
    parser.add_argument('--scaler', default='standard')
    parser.add_argument('--log-mlflow', action='store_true')
    parser.add_argument('--register-model', action='store_true')
    parser.add_argument('--registered-model-name', default='PumpAD')
    parser.add_argument('--alias', default=None)
    args = parser.parse_args(argv)
    return PcaTrainingConfig(**vars(args))


def _parse_n_components(value: str) -> int | float | None:
    if value.lower() == 'none':
        return None
    parsed_float = float(value)
    if parsed_float.is_integer() and not any(marker in value.lower() for marker in ('.', 'e')):
        return int(parsed_float)
    return parsed_float


def _normalize_config(config: PcaTrainingConfig) -> PcaTrainingConfig:
    validation_input_path = Path(config.validation_input_path) if config.validation_input_path is not None else None
    return replace(
        config,
        input_path=Path(config.input_path),
        output_dir=Path(config.output_dir),
        validation_input_path=validation_input_path,
    )


def _load_windows(path: Path, config: PcaTrainingConfig) -> WindowedSensorDataset:
    return build_sensor_windows(
        load_skab_csv(path),
        window_size=config.window_size,
        stride=config.stride,
        sensor_columns=SENSOR_COLUMNS,
    )


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        'detector': output_dir / 'pca_detector.joblib',
        'metadata': output_dir / 'metadata.json',
        'metrics': output_dir / 'metrics.json',
        'scores': output_dir / 'scores.csv',
    }


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
            'prediction': predictions.astype(int),
            't2': statistics[:, 0].astype(float),
            'q': statistics[:, 1].astype(float),
            'score': scores.astype(float),
        }
    )


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    positive = labels == 1
    negative = labels == 0
    predicted_positive = predictions == 1
    predicted_negative = predictions == 0

    true_positive = int(np.count_nonzero(predicted_positive & positive))
    false_positive = int(np.count_nonzero(predicted_positive & negative))
    false_negative = int(np.count_nonzero(predicted_negative & positive))
    true_negative = int(np.count_nonzero(predicted_negative & negative))

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    return {
        'precision': precision,
        'recall': recall,
        'f1': _safe_divide(2.0 * precision * recall, precision + recall),
        'false_alarm_rate': _safe_divide(false_positive, false_positive + true_negative),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _dump_detector(detector: PcaT2QDetector, path: Path) -> None:
    joblib = import_module('joblib')
    joblib.dump(detector, path)


def _params(config: PcaTrainingConfig, training_windows: WindowedSensorDataset) -> dict[str, Any]:
    return {
        'input_path': str(config.input_path),
        'output_dir': str(config.output_dir),
        'validation_input_path': str(config.validation_input_path) if config.validation_input_path is not None else None,
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


def _metadata_payload(config: PcaTrainingConfig, result: PcaTrainingResult) -> dict[str, Any]:
    return {
        'params': result.params,
        'metrics': result.metrics,
        'thresholds': result.thresholds,
        'sensor_columns': list(result.sensor_columns),
        'artifact_paths': {name: str(path) for name, path in result.artifact_paths.items()},
        'config': _jsonable_config(config),
    }


def _jsonable_config(config: PcaTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ('input_path', 'output_dir', 'validation_input_path'):
        value = payload[key]
        payload[key] = str(value) if value is not None else None
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
