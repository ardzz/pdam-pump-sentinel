from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from app.observability.metrics import set_model_info
from ml.inference.loader import load_inference_service_from_artifacts

MODEL_DIR_ENV = 'PUMPAD_MODEL_DIR'
logger = logging.getLogger(__name__)


class InferenceService(Protocol):
    def observe(self, station: str, timestamp: str | None, sensors: Mapping[str, Any]) -> Any:
        ...


_service: InferenceService | None = None
_loaded = False
_service_alias_metadata: dict[str, Any] | None = None
_service_alias_signature: tuple[str, str, str] | None = None
_service_model_info: dict[str, Any] | None = None
_service_lock = threading.Lock()


def get_inference_service() -> InferenceService | None:
    global _loaded
    with _service_lock:
        if _loaded:
            service = _service
            model_info = _service_model_info
        else:
            service = None
            model_info = None

    if service is not None:
        if model_info is not None:
            set_model_info(model_info)
        else:
            _set_service_model_info(service, os.getenv(MODEL_DIR_ENV, ''))
        return service
    if _loaded:
        return None

    loaded_service: InferenceService | None = None
    model_dir = os.getenv(MODEL_DIR_ENV)
    if model_dir and _looks_like_model_dir(Path(model_dir)):
        try:
            loaded_service = load_inference_service_from_artifacts(model_dir)
        except Exception as exc:
            logger.warning('PUMPAD_MODEL_DIR inference service load failed: %s', exc, exc_info=False)
    if loaded_service is None:
        try:
            from ml.registry.mlflow_client import load_champion_service

            loaded_service = load_champion_service(model_name='PumpAD', alias='champion')
        except Exception as exc:
            logger.warning('MLflow champion inference service load failed: %s', exc, exc_info=False)
            with _service_lock:
                if not _loaded:
                    _publish_service(None, None, None, None)
            return None

    metadata = _metadata_from_service(loaded_service) if loaded_service is not None else None
    signature = _alias_signature(metadata)
    model_info = (
        _service_model_info_payload(loaded_service, model_dir or _service_model_dir(loaded_service), metadata)
        if loaded_service is not None
        else None
    )
    with _service_lock:
        if not _loaded:
            _publish_service(loaded_service, metadata, signature, model_info)
        service = _service
        model_info = _service_model_info
    if service is not None and model_info is not None:
        set_model_info(model_info)
    return service


def refresh_inference_service_from_alias(model_name: str = 'PumpAD', alias: str = 'champion') -> InferenceService | None:
    try:
        from ml.registry import mlflow_client

        metadata = mlflow_client.get_model_alias_metadata(model_name, alias)
    except Exception as exc:
        logger.warning('MLflow alias metadata refresh failed: %s', exc, exc_info=False)
        return get_inference_service()
    if metadata is None:
        logger.warning('MLflow alias metadata refresh returned no metadata for %s@%s', model_name, alias)
        return get_inference_service()

    new_signature = _alias_signature(metadata)
    with _service_lock:
        incumbent = _service
        incumbent_signature = _service_alias_signature or _alias_signature(_metadata_from_service(_service))
    if incumbent is not None and new_signature is not None and new_signature == incumbent_signature:
        return incumbent

    try:
        from ml.registry import mlflow_client

        replacement = mlflow_client.load_champion_service(model_name=model_name, alias=alias)
    except Exception as exc:
        logger.warning('MLflow alias service refresh failed: %s', exc, exc_info=False)
        return get_inference_service()
    if replacement is None:
        logger.warning('MLflow alias service refresh returned no service for %s@%s', model_name, alias)
        return get_inference_service()

    _attach_alias_metadata(replacement, metadata)
    model_info = _service_model_info_payload(replacement, _service_model_dir(replacement), metadata)
    with _service_lock:
        _publish_service(replacement, metadata, new_signature, model_info)
    set_model_info(model_info)
    return replacement


def set_inference_service(service: InferenceService | None) -> None:
    metadata = _metadata_from_service(service) if service is not None else None
    signature = _alias_signature(metadata)
    model_info = _service_model_info_payload(service, _service_model_dir(service), metadata) if service is not None else None
    with _service_lock:
        _publish_service(service, metadata, signature, model_info)


def reset_inference_service() -> None:
    with _service_lock:
        _publish_service(None, None, None, None, loaded=False)


def _looks_like_model_dir(model_dir: Path) -> bool:
    return any((model_dir / name).exists() for name in ('metadata.json', 'pca_detector.joblib', 'lstm_ae.keras'))


def _set_service_model_info(service: InferenceService, model_dir: str) -> None:
    set_model_info(_service_model_info_payload(service, model_dir, _metadata_from_service(service)))


def _publish_service(
    service: InferenceService | None,
    metadata: Mapping[str, Any] | None,
    signature: tuple[str, str, str] | None,
    model_info: Mapping[str, Any] | None,
    *,
    loaded: bool = True,
) -> None:
    global _service, _loaded, _service_alias_metadata, _service_alias_signature, _service_model_info
    _service = service
    _loaded = loaded
    _service_alias_metadata = dict(metadata) if metadata is not None else None
    _service_alias_signature = signature
    _service_model_info = dict(model_info) if model_info is not None else None


def _service_model_info_payload(
    service: InferenceService,
    model_dir: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    values: Mapping[str, Any] = metadata or {}
    return {
        'name': _metadata_value(values, 'name') or _metadata_value(values, 'registered_model_name') or 'PumpAD',
        'version': getattr(service, 'model_version', '') or _metadata_value(values, 'version'),
        'alias': _metadata_value(values, 'alias') or 'champion',
        'model_dir': model_dir or _service_model_dir(service),
        'run_id': getattr(service, 'run_id', '') or _metadata_value(values, 'run_id'),
    }


def _metadata_from_service(service: InferenceService | None) -> dict[str, Any] | None:
    if service is None:
        return None
    metadata = getattr(service, 'metadata', {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    values = dict(metadata)
    version = getattr(service, 'model_version', None)
    if version and not values.get('version'):
        values['version'] = str(version)
    run_id = getattr(service, 'run_id', None)
    if run_id and not values.get('run_id'):
        values['run_id'] = str(run_id)
    return values


def _attach_alias_metadata(service: InferenceService, metadata: Mapping[str, Any]) -> None:
    current = _metadata_from_service(service) or {}
    current.update(metadata)
    try:
        setattr(service, 'metadata', current)
    except Exception as exc:
        logger.debug('could not attach alias metadata to inference service: %s', exc, exc_info=False)
    for attr_name, key in (('model_version', 'mlflow_version'), ('run_id', 'run_id')):
        value = _metadata_value(metadata, key)
        if value is None:
            continue
        try:
            setattr(service, attr_name, str(value))
        except Exception as exc:
            logger.debug('could not attach alias %s to inference service: %s', attr_name, exc, exc_info=False)


def _alias_signature(metadata: Mapping[str, Any] | None) -> tuple[str, str, str] | None:
    if not metadata:
        return None
    name = _metadata_value(metadata, 'registered_model_name') or _metadata_value(metadata, 'name')
    version = _metadata_value(metadata, 'mlflow_version') or _metadata_value(metadata, 'version')
    run_id = _metadata_value(metadata, 'run_id')
    if name is None or version is None or run_id is None:
        return None
    return (str(name), str(version), str(run_id))


def _service_model_dir(service: InferenceService) -> str:
    value = getattr(service, 'model_dir', '') or getattr(service, 'artifact_dir', '')
    return '' if value is None else str(value)


def _metadata_value(metadata: Mapping[str, Any], key: str) -> Any:
    return metadata.get(key)
