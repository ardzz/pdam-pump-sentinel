from __future__ import annotations

import math
import os
from collections.abc import Mapping
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

            model_info = mlflow_tensorflow.log_model(
                model,
                name=LSTM_AE_MODEL_ARTIFACT_NAME,
                registered_model_name=registered_model_name,
            )
            _set_model_alias_if_requested(mlflow, config, registered_model_name, model_info, run_id)

        return run_id
    except Exception:
        return None


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


__all__ = ['PcaAnomalyInferenceService', 'load_champion_service', 'log_lstm_ae_training_run', 'log_pca_training_run']
