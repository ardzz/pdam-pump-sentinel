from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from importlib import import_module
from numbers import Real
from pathlib import Path
from typing import Any, Protocol

from ml.inference.loader import load_inference_service_from_artifacts
from ml.inference.pca_inference import PcaAnomalyInferenceService as PcaAnomalyInferenceService

MODEL_ARTIFACT_NAME = 'pca_anomaly_model'
LSTM_AE_MODEL_ARTIFACT_NAME = 'lstm_ae_model'
DEFAULT_REGISTERED_MODEL_NAME = 'PumpAD'
MODEL_DIR_ENV = 'PUMPAD_MODEL_DIR'


class InferenceService(Protocol):
    def observe(self, station: str, timestamp: str | None, sensors: Mapping[str, Any]) -> Any:
        ...


def log_pca_training_run(result: Any, detector: Any, config: Any) -> str | None:
    """Log a PCA training run to MLflow when the optional dependency is available.

    The helper intentionally imports MLflow inside the function so importing the
    training lane does not require a live MLflow installation or server. It logs
    flat scalar params from the config/result objects, numeric metrics from the
    result, local artifacts from the result output directory, and the fitted
    sklearn-compatible detector.
    """

    try:
        mlflow = import_module('mlflow')
        mlflow_sklearn = import_module('mlflow.sklearn')
    except ImportError:
        return None

    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    run_id: str | None = None
    registered_model_name = _registered_model_name(config)

    with mlflow.start_run() as run:
        run_id = _run_id(run)

        params = _collect_params(config, result)
        if params:
            mlflow.log_params(params)

        metrics = _collect_metrics(result)
        if metrics:
            mlflow.log_metrics(metrics)

        output_dir = _result_output_dir(result)
        if output_dir is not None and output_dir.exists():
            mlflow.log_artifacts(str(output_dir))

        model_info = mlflow_sklearn.log_model(
            detector,
            name=MODEL_ARTIFACT_NAME,
            registered_model_name=registered_model_name,
        )
        _set_model_alias_if_requested(mlflow, config, registered_model_name, model_info, run_id)

    return run_id


def skab_window_dataframe(dataset: Any, *, normal_only: bool = False) -> Any | None:
    try:
        pandas = import_module('pandas')
    except ImportError:
        return None
    features = getattr(dataset, 'features', None)
    if features is None:
        return None
    labels = getattr(dataset, 'labels', None)
    changepoints = getattr(dataset, 'changepoints', None)
    if labels is None or changepoints is None:
        return None
    mask = None
    if normal_only:
        try:
            mask = labels == 0
        except Exception:
            mask = None
    selected_features = features[mask] if mask is not None else features
    selected_labels = labels[mask] if mask is not None else labels
    selected_changepoints = changepoints[mask] if mask is not None else changepoints
    if len(selected_features) == 0:
        return None
    frame = pandas.DataFrame(
        selected_features,
        columns=[f'feature_{index}' for index in range(int(selected_features.shape[1]))],
    )
    frame['label'] = selected_labels.astype(int)
    frame['changepoint'] = selected_changepoints.astype(int)
    return frame


def log_skab_inputs_to_active_run(
    *,
    train_df: Any,
    val_df: Any | None,
    test_df: Any | None,
    manifest_path: Path | str | None,
    manifest_dict: Mapping[str, Any] | None,
    feature_mode: str,
    split_strategy: str,
) -> None:
    try:
        mlflow = import_module('mlflow')
    except ImportError:
        return
    active_run = getattr(mlflow, 'active_run', None)
    if callable(active_run):
        try:
            if active_run() is None:
                return
        except Exception:
            return
    try:
        mlflow_data = getattr(mlflow, 'data', None) or import_module('mlflow.data')
    except ImportError:
        return

    source_path = Path(manifest_path).resolve() if manifest_path is not None else None
    tags = {
        'dataset.manifest_sha256': _manifest_sha256(source_path, manifest_dict),
        'dataset.feature_mode': str(feature_mode),
        'dataset.split_strategy': str(split_strategy),
    }
    try:
        mlflow.set_tags(tags)
    except Exception:
        pass

    targets = 'label' if str(split_strategy).startswith('supervised') else None
    split_frames = (('train', 'training', train_df), ('validation', 'validation', val_df), ('test', 'test', test_df))
    for split_name, context, frame in split_frames:
        if frame is None or len(frame) == 0:
            continue
        try:
            dataset = mlflow_data.from_pandas(
                frame,
                source=_dataset_source_uri(source_path, split_name),
                name=f'skab.{split_strategy}.{split_name}',
                targets=targets if targets in frame.columns else None,
            )
            mlflow.log_input(dataset, context=context)
        except Exception:
            continue


def load_champion_service(
    model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    alias: str = 'champion',
    local_model_dir: str | None = None,
) -> InferenceService | None:
    service = _load_service_from_mlflow_alias(model_name, alias)
    if service is not None:
        return service
    return _load_service_from_local_dir(local_model_dir or os.getenv(MODEL_DIR_ENV))


def _load_service_from_mlflow_alias(model_name: str, alias: str) -> InferenceService | None:
    try:
        mlflow = import_module('mlflow')
        mlflow_artifacts = import_module('mlflow.artifacts')
        mlflow_tracking = import_module('mlflow.tracking')
    except ImportError:
        return None

    try:
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        client = mlflow_tracking.MlflowClient()
        version = client.get_model_version_by_alias(model_name, alias)
        run_id = getattr(version, 'run_id', None)
        if not run_id:
            return None

        downloaded_dir = mlflow_artifacts.download_artifacts(run_id=str(run_id), artifact_path='')
        model_dir = _find_logged_artifact_dir(downloaded_dir)
        if model_dir is None:
            return None
        return load_inference_service_from_artifacts(model_dir, _model_version(version))
    except Exception:
        return None


def _load_service_from_local_dir(model_dir: str | None) -> InferenceService | None:
    if not model_dir:
        return None
    directory = Path(model_dir)
    if not _looks_like_model_dir(directory):
        return None
    try:
        return load_inference_service_from_artifacts(directory)
    except Exception:
        return None


def log_lstm_ae_training_run(result: Any, model: Any, config: Any) -> str | None:
    try:
        mlflow = import_module('mlflow')
        mlflow_tensorflow = import_module('mlflow.tensorflow')
    except ImportError:
        return None

    try:
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        registered_model_name = _registered_model_name(config)
        active_run = mlflow.active_run() if hasattr(mlflow, 'active_run') else None
        if active_run is not None:
            run_id = _run_id(active_run)
            _log_lstm_ae_training_run_to_active_run(
                mlflow,
                mlflow_tensorflow,
                result,
                model,
                config,
                registered_model_name,
                run_id,
            )
            return run_id

        with mlflow.start_run() as run:
            run_id = _run_id(run)
            _log_lstm_ae_training_run_to_active_run(
                mlflow,
                mlflow_tensorflow,
                result,
                model,
                config,
                registered_model_name,
                run_id,
            )

        return run_id
    except Exception:
        return None


def start_lstm_ae_training_run(config: Any) -> Any | None:
    try:
        mlflow = import_module('mlflow')
    except ImportError:
        return None

    try:
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        active_run = mlflow.active_run() if hasattr(mlflow, 'active_run') else None
        if active_run is not None:
            return nullcontext(active_run)
        return mlflow.start_run()
    except Exception:
        return None


def _log_lstm_ae_training_run_to_active_run(
    mlflow: Any,
    mlflow_tensorflow: Any,
    result: Any,
    model: Any,
    config: Any,
    registered_model_name: str | None,
    run_id: str | None,
) -> None:
    params = _collect_params(config, result)
    if params:
        mlflow.log_params(params)

    metrics = _collect_metrics(result)
    if metrics:
        mlflow.log_metrics(metrics)

    output_dir = _result_output_dir(result)
    if output_dir is not None and output_dir.exists():
        mlflow.log_artifacts(str(output_dir))

    model_info = mlflow_tensorflow.log_model(
        model,
        name=LSTM_AE_MODEL_ARTIFACT_NAME,
        registered_model_name=registered_model_name,
    )
    _set_model_alias_if_requested(mlflow, config, registered_model_name, model_info, run_id)


def _looks_like_model_dir(model_dir: Path) -> bool:
    return any((model_dir / name).exists() for name in ('metadata.json', 'pca_detector.joblib', 'lstm_ae.keras'))


def _model_version(version: Any) -> str | None:
    value = getattr(version, 'version', None)
    return None if value is None else str(value)


def _registered_model_name(config: Any) -> str | None:
    if not bool(_lookup(config, 'register_model', False)):
        return None
    return str(_lookup(config, 'registered_model_name', DEFAULT_REGISTERED_MODEL_NAME) or DEFAULT_REGISTERED_MODEL_NAME)


def _set_model_alias_if_requested(
    mlflow: Any,
    config: Any,
    registered_model_name: str | None,
    model_info: Any,
    run_id: str | None,
) -> None:
    alias = _lookup(config, 'alias', None)
    if not registered_model_name or not alias:
        return

    version = _resolve_registered_model_version(registered_model_name, run_id, model_info)
    if version is None:
        return

    try:
        mlflow_tracking = import_module('mlflow.tracking')
        client = mlflow_tracking.MlflowClient()
        client.set_registered_model_alias(registered_model_name, str(alias), str(version))
    except Exception:
        if hasattr(mlflow, 'set_registered_model_alias'):
            mlflow.set_registered_model_alias(registered_model_name, str(alias), str(version))


def _resolve_registered_model_version(registered_model_name: str, run_id: str | None, model_info: Any) -> str | None:
    try:
        mlflow_tracking = import_module('mlflow.tracking')
        client = mlflow_tracking.MlflowClient()
        if run_id:
            versions = client.search_model_versions(filter_string=f"run_id='{run_id}'")
            version = _select_model_version(versions, registered_model_name)
            if version is not None:
                return version
        version = _select_model_version(client.get_latest_versions(registered_model_name), registered_model_name)
        if version is not None:
            return version
    except Exception:
        pass
    return _model_version(model_info)


def _select_model_version(versions: Any, registered_model_name: str) -> str | None:
    candidates = list(versions or [])
    matching = [version for version in candidates if getattr(version, 'name', registered_model_name) == registered_model_name]
    selected = matching or candidates
    if not selected:
        return None
    return max((_model_version(version) for version in selected), key=_version_sort_key, default=None)


def _version_sort_key(version: str | None) -> tuple[int, str]:
    if version is None:
        return (-1, '')
    return (int(version), version) if version.isdigit() else (0, version)


def _find_logged_artifact_dir(artifact_dir: str | Path) -> Path | None:
    directory = Path(artifact_dir)
    if (directory / 'metadata.json').exists():
        return directory
    for metadata_path in sorted(directory.rglob('metadata.json')):
        return metadata_path.parent
    return None


def _collect_params(config: Any, result: Any) -> dict[str, str | int | float | bool]:
    params: dict[str, str | int | float | bool] = {}
    params.update(_scalar_params(_iter_public_values(config)))
    params.update(_scalar_params(_iter_public_values(_lookup(result, 'params', {}))))

    excluded_result_fields = {'metrics', 'params', 'output_dir', 'artifact_dir', 'artifacts_dir', 'run_dir'}
    result_params = ((name, value) for name, value in _iter_public_values(result) if name not in excluded_result_fields)
    params.update(_scalar_params(result_params))
    return params


def _collect_metrics(result: Any) -> dict[str, float]:
    return _numeric_metrics(_iter_public_values(_lookup(result, 'metrics', {})))


def _result_output_dir(result: Any) -> Path | None:
    for field_name in ('output_dir', 'artifact_dir', 'artifacts_dir', 'run_dir'):
        value = _lookup(result, field_name, None)
        if value:
            return Path(value)
    return None


def _run_id(run: Any) -> str | None:
    info = getattr(run, 'info', None)
    run_id = getattr(info, 'run_id', None) or getattr(run, 'run_id', None)
    if run_id is None:
        return None
    return str(run_id)


def _lookup(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _iter_public_values(obj: Any) -> list[tuple[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, Mapping):
        return [(str(key), value) for key, value in obj.items() if not str(key).startswith('_')]
    if is_dataclass(obj) and not isinstance(obj, type):
        return [(field.name, getattr(obj, field.name)) for field in fields(obj)]
    as_dict = getattr(obj, '_asdict', None)
    if callable(as_dict):
        return _iter_public_values(as_dict())
    values = getattr(obj, '__dict__', None)
    if isinstance(values, Mapping):
        return [(str(key), value) for key, value in values.items() if not str(key).startswith('_')]
    return []


def _scalar_params(values: Any) -> dict[str, str | int | float | bool]:
    params: dict[str, str | int | float | bool] = {}
    for name, value in values:
        normalized = _param_value(value)
        if normalized is not None:
            params[name] = normalized
    return params


def _param_value(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    return None


def _numeric_metrics(values: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        metric = float(value)
        if math.isfinite(metric):
            metrics[name] = metric
    return metrics


def _manifest_sha256(path: Path | None, manifest_dict: Mapping[str, Any] | None) -> str:
    if path is not None and path.exists():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest_dict is None:
        return hashlib.sha256(b'').hexdigest()
    import json

    payload = json.dumps(manifest_dict, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _dataset_source_uri(path: Path | None, split_name: str) -> str:
    if path is None:
        return f'skab://in-memory#{split_name}'
    return f'{path.as_uri()}#{split_name}'


__all__ = [
    'PcaAnomalyInferenceService',
    'load_champion_service',
    'log_skab_inputs_to_active_run',
    'log_lstm_ae_training_run',
    'log_pca_training_run',
    'skab_window_dataframe',
    'start_lstm_ae_training_run',
]
