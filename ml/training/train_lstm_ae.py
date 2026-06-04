from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from ml.datasets.skab_loader import SENSOR_COLUMNS, load_skab_csv
from ml.datasets.skab_manifest import SkabSplitManifest, load_skab_split_manifest
from ml.evaluation.metrics import evaluate_split
from ml.features.windowing import WindowedSensorDataset, build_sensor_windows
from ml.training.train_pca import _METRIC_PROTOCOL
from ml.utils.provenance import collect_provenance


@dataclass(frozen=True)
class LstmAeTrainingConfig:
    input_path: Path
    output_dir: Path
    validation_input_path: Path | None = None
    split_manifest_path: Path | None = None
    window_size: int = 60
    stride: int = 1
    threshold_quantile: float = 0.99
    log_mlflow: bool = False
    register_model: bool = False
    registered_model_name: str = 'PumpAD'
    alias: str | None = None
    lstm_units: int = 64
    latent_dim: int = 16
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 10
    seed: int = 42


@dataclass(frozen=True)
class LstmAeTrainingResult:
    output_dir: Path
    artifact_paths: dict[str, Path]
    params: dict[str, Any]
    metrics: dict[str, Any]
    thresholds: dict[str, float]
    sensor_columns: tuple[str, ...]


def train_lstm_ae_from_skab(config: LstmAeTrainingConfig) -> LstmAeTrainingResult:
    normalized_config = _normalize_config(config)
    mlflow_run_context = _start_mlflow_run_if_requested(normalized_config)
    if mlflow_run_context is None:
        result, _ = _fit_and_write_artifacts(normalized_config)
        return result

    with mlflow_run_context:
        result, model = _fit_and_write_artifacts(normalized_config, log_mlflow_epoch_metrics=True)
        _log_lstm_ae_training_run_safely(result, model, normalized_config)

    return result


def main(argv: Sequence[str] | None = None) -> LstmAeTrainingResult:
    config = _parse_args(argv)
    result = train_lstm_ae_from_skab(config)
    print(json.dumps(_result_payload(result), sort_keys=True))
    return result


def _fit_and_write_artifacts(
    config: LstmAeTrainingConfig,
    *,
    log_mlflow_epoch_metrics: bool = False,
) -> tuple[LstmAeTrainingResult, Any]:
    if config.split_manifest_path is not None:
        return _fit_and_write_artifacts_split(config, log_mlflow_epoch_metrics=log_mlflow_epoch_metrics)

    training_windows = _load_windows(config.input_path, config)
    validation_path = config.validation_input_path or config.input_path
    validation_windows = training_windows if validation_path == config.input_path else _load_windows(validation_path, config)
    input_files = _non_split_input_files(config)
    return _fit_and_write_artifacts_common(
        config=config,
        train_windows=training_windows,
        validation_windows=validation_windows,
        test_windows=None,
        scaler_fit_paths=[config.input_path],
        provenance_input_files=input_files,
        log_mlflow_epoch_metrics=log_mlflow_epoch_metrics,
    )


def _fit_and_write_artifacts_split(
    config: LstmAeTrainingConfig,
    *,
    log_mlflow_epoch_metrics: bool = False,
) -> tuple[LstmAeTrainingResult, Any]:
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
        scaler_fit_paths=manifest.train,
        provenance_input_files=_split_input_files(config, manifest),
        manifest=manifest,
        log_mlflow_epoch_metrics=log_mlflow_epoch_metrics,
    )


def _fit_and_write_artifacts_common(
    *,
    config: LstmAeTrainingConfig,
    train_windows: WindowedSensorDataset,
    validation_windows: WindowedSensorDataset,
    test_windows: WindowedSensorDataset | None,
    scaler_fit_paths: Sequence[Path],
    provenance_input_files: Sequence[Path],
    manifest: SkabSplitManifest | None = None,
    log_mlflow_epoch_metrics: bool = False,
) -> tuple[LstmAeTrainingResult, Any]:
    train_normal_mask = train_windows.labels == 0
    if len(train_windows.features[train_normal_mask]) == 0:
        raise ValueError('training input must contain at least one normal window')
    if len(validation_windows.features) == 0:
        raise ValueError('validation input must produce at least one window')

    validation_normal_mask = validation_windows.labels == 0
    if len(validation_windows.features[validation_normal_mask]) == 0:
        raise ValueError('validation input must contain at least one normal window')

    scaler = StandardScaler().fit(_load_normal_sensor_rows(scaler_fit_paths))
    train_x = _scaled_windows(train_windows, scaler)
    validation_x = _scaled_windows(validation_windows, scaler)
    test_x = _scaled_windows(test_windows, scaler) if test_windows is not None else None

    keras = import_module('keras')
    keras.utils.set_random_seed(config.seed)
    model = _build_model(keras, config, len(train_windows.sensor_columns))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate), loss='mse')

    train_normal_x = train_x[train_normal_mask]
    validation_normal_x = validation_x[validation_normal_mask]
    fit_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.patience,
            restore_best_weights=True,
        )
    ]
    if log_mlflow_epoch_metrics:
        mlflow_callback = _mlflow_epoch_metrics_callback(keras)
        if mlflow_callback is not None:
            fit_callbacks.append(mlflow_callback)

    model.fit(
        train_normal_x,
        train_normal_x,
        validation_data=(validation_normal_x, validation_normal_x),
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=False,
        verbose=0,
        callbacks=fit_callbacks,
    )

    validation_normal_scores = _reconstruction_errors(model, validation_normal_x, config.batch_size)
    threshold = float(np.percentile(validation_normal_scores, config.threshold_quantile * 100.0))
    thresholds = {'threshold': threshold, 't2_threshold': threshold, 'q_threshold': threshold}

    validation_scores = _reconstruction_errors(model, validation_x, config.batch_size)
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
        'training_normal_count': int(np.count_nonzero(train_normal_mask)),
        'train_count': int(len(train_windows.labels)),
        'validation_count': int(len(validation_windows.labels)),
        **thresholds,
    }

    has_test = False
    if test_windows is not None and test_x is not None and len(test_windows.features) > 0:
        has_test = True
        metrics['test_count'] = int(len(test_windows.labels))

    params = _params(config, train_windows, threshold)
    artifact_paths = _artifact_paths_split(config.output_dir, has_test) if manifest is not None else _artifact_paths(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    _save_model(model, artifact_paths['model'])
    _dump_scaler(scaler, artifact_paths['scaler'])
    _scores_frame(validation_windows, validation_labels, validation_predictions, validation_scores).to_csv(
        artifact_paths['scores'], index=False
    )

    if test_windows is not None and test_x is not None and len(test_windows.features) > 0:
        test_scores = _reconstruction_errors(model, test_x, config.batch_size)
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

    result = LstmAeTrainingResult(
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


def _parse_args(argv: Sequence[str] | None) -> LstmAeTrainingConfig:
    parser = argparse.ArgumentParser(description='Train an LSTM autoencoder detector from SKAB CSV telemetry.')
    parser.add_argument('paths', type=Path, nargs='+', metavar='PATH')
    parser.add_argument('--validation-input-path', type=Path, default=None)
    parser.add_argument('--split-manifest', type=Path, default=None)
    parser.add_argument('--window-size', type=int, default=60)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--threshold-quantile', type=float, default=0.99)
    parser.add_argument('--log-mlflow', action='store_true')
    parser.add_argument('--register-model', action='store_true')
    parser.add_argument('--registered-model-name', default='PumpAD')
    parser.add_argument('--alias', default=None)
    parser.add_argument('--lstm-units', type=int, default=64)
    parser.add_argument('--latent-dim', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
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
    return LstmAeTrainingConfig(**args_dict)


def _normalize_config(config: LstmAeTrainingConfig) -> LstmAeTrainingConfig:
    validation_input_path = Path(config.validation_input_path) if config.validation_input_path is not None else None
    split_manifest_path = Path(config.split_manifest_path) if config.split_manifest_path is not None else None
    return replace(
        config,
        input_path=Path(config.input_path),
        output_dir=Path(config.output_dir),
        validation_input_path=validation_input_path,
        split_manifest_path=split_manifest_path,
    )


def _load_windows(path: Path, config: LstmAeTrainingConfig) -> WindowedSensorDataset:
    return build_sensor_windows(
        load_skab_csv(path),
        window_size=config.window_size,
        stride=config.stride,
        sensor_columns=SENSOR_COLUMNS,
    )


def _load_windows_multi(paths: list[Path], config: LstmAeTrainingConfig) -> WindowedSensorDataset:
    datasets = [_load_windows(p, config) for p in paths]
    if not datasets:
        return WindowedSensorDataset(
            features=np.empty((0, config.window_size * len(SENSOR_COLUMNS)), dtype=float),
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


def _load_normal_sensor_rows(paths: Sequence[Path]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for path in paths:
        frame = load_skab_csv(path)
        normal = frame['anomaly'].to_numpy() == 0
        if np.any(normal):
            rows.append(frame.loc[normal, SENSOR_COLUMNS].to_numpy(dtype=float, copy=False))
    if not rows:
        raise ValueError('training input must contain at least one normal sensor row')
    return np.vstack(rows)


def _scaled_windows(dataset: WindowedSensorDataset, scaler: StandardScaler) -> np.ndarray:
    sensor_count = len(dataset.sensor_columns)
    shaped = dataset.features.reshape(len(dataset.features), dataset.window_size, sensor_count)
    if len(shaped) == 0:
        return shaped.astype(np.float64, copy=False)
    scaled = np.asarray(scaler.transform(shaped.reshape(-1, sensor_count)), dtype=np.float64)
    return scaled.reshape(shaped.shape).astype(np.float64, copy=False)


def _build_model(keras: Any, config: LstmAeTrainingConfig, sensor_count: int) -> Any:
    inputs = keras.Input(shape=(config.window_size, sensor_count))
    encoded = keras.layers.LSTM(config.lstm_units)(inputs)
    latent = keras.layers.Dense(config.latent_dim)(encoded)
    repeated = keras.layers.RepeatVector(config.window_size)(latent)
    decoded = keras.layers.LSTM(config.lstm_units, return_sequences=True)(repeated)
    outputs = keras.layers.TimeDistributed(keras.layers.Dense(sensor_count))(decoded)
    return keras.Model(inputs, outputs)


def _mlflow_epoch_metrics_callback(keras: Any) -> Any | None:
    try:
        mlflow = import_module('mlflow')
    except ImportError:
        return None

    def on_epoch_end(epoch: int, logs: dict[str, Any] | None = None) -> None:
        for metric_name in ('loss', 'val_loss'):
            value = (logs or {}).get(metric_name)
            if value is None:
                continue
            try:
                metric_value = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(metric_value):
                continue
            try:
                mlflow.log_metric(metric_name, metric_value, step=int(epoch))
            except Exception:
                pass

    return keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)


def _start_mlflow_run_if_requested(config: LstmAeTrainingConfig) -> Any | None:
    if not config.log_mlflow:
        return None
    try:
        from ml.registry.mlflow_client import start_lstm_ae_training_run
    except Exception:
        return None
    return start_lstm_ae_training_run(config)


def _log_lstm_ae_training_run_safely(result: LstmAeTrainingResult, model: Any, config: LstmAeTrainingConfig) -> None:
    try:
        from ml.registry.mlflow_client import log_lstm_ae_training_run

        log_lstm_ae_training_run(result, model, config)
    except Exception:
        pass


def _reconstruction_errors(model: Any, windows: np.ndarray, batch_size: int) -> np.ndarray:
    if len(windows) == 0:
        return np.empty((0,), dtype=float)
    reconstructed = np.asarray(model.predict(windows, batch_size=batch_size, verbose=0), dtype=np.float64)
    return np.mean(np.abs(windows - reconstructed), axis=(1, 2))


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        'model': output_dir / 'lstm_ae.keras',
        'scaler': output_dir / 'scaler.joblib',
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


def _save_model(model: Any, path: Path) -> None:
    model.save(str(path))


def _dump_scaler(scaler: StandardScaler, path: Path) -> None:
    joblib = import_module('joblib')
    joblib.dump(scaler, path)


def _params(config: LstmAeTrainingConfig, training_windows: WindowedSensorDataset, threshold: float) -> dict[str, Any]:
    return {
        'input_path': str(config.input_path),
        'output_dir': str(config.output_dir),
        'validation_input_path': str(config.validation_input_path) if config.validation_input_path is not None else None,
        'split_manifest_path': str(config.split_manifest_path) if config.split_manifest_path is not None else None,
        'window_size': config.window_size,
        'stride': config.stride,
        'threshold': threshold,
        'threshold_quantile': config.threshold_quantile,
        'score_type': 'mae',
        'log_mlflow': config.log_mlflow,
        'register_model': config.register_model,
        'registered_model_name': config.registered_model_name,
        'alias': config.alias,
        'lstm_units': config.lstm_units,
        'latent_dim': config.latent_dim,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'patience': config.patience,
        'seed': config.seed,
        'sensor_columns': list(training_windows.sensor_columns),
        'sensor_count': len(training_windows.sensor_columns),
        'feature_count': int(training_windows.features.shape[1]),
    }


def _metadata_payload(
    config: LstmAeTrainingConfig,
    result: LstmAeTrainingResult,
    provenance_input_files: Sequence[Path],
    manifest: SkabSplitManifest | None,
    train_windows: WindowedSensorDataset,
    validation_windows: WindowedSensorDataset,
    test_windows: WindowedSensorDataset | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'model_family': 'lstm_ae',
        'params': result.params,
        'metrics': result.metrics,
        'thresholds': result.thresholds,
        'sensor_columns': list(result.sensor_columns),
        'artifact_paths': {name: str(path) for name, path in result.artifact_paths.items()},
        'config': _jsonable_config(config),
        'metric_protocol': _METRIC_PROTOCOL,
        'provenance': collect_provenance(config=config, input_files=provenance_input_files),
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


def _non_split_input_files(config: LstmAeTrainingConfig) -> list[Path]:
    paths = [config.input_path]
    if config.validation_input_path is not None:
        paths.append(config.validation_input_path)
    return _unique_paths(paths)


def _split_input_files(config: LstmAeTrainingConfig, manifest: SkabSplitManifest) -> list[Path]:
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
    config: LstmAeTrainingConfig,
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
        feature_mode='raw',
        split_strategy='unsupervised_novel_fault' if config.split_manifest_path is not None else 'single_csv',
    )


def _jsonable_config(config: LstmAeTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ('input_path', 'output_dir', 'validation_input_path', 'split_manifest_path'):
        value = payload[key]
        payload[key] = str(value) if value is not None else None
    return payload


def _result_payload(result: LstmAeTrainingResult) -> dict[str, Any]:
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
