from __future__ import annotations

import logging
import os
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


def get_inference_service() -> InferenceService | None:
    global _service, _loaded
    if _loaded:
        if _service is not None:
            _set_service_model_info(_service, os.getenv(MODEL_DIR_ENV, ''))
        return _service

    _loaded = True
    model_dir = os.getenv(MODEL_DIR_ENV)
    if model_dir and _looks_like_model_dir(Path(model_dir)):
        try:
            _service = load_inference_service_from_artifacts(model_dir)
        except Exception as exc:
            logger.warning('PUMPAD_MODEL_DIR inference service load failed: %s', exc, exc_info=False)
    if _service is None:
        try:
            from ml.registry.mlflow_client import load_champion_service

            _service = load_champion_service(model_name='PumpAD', alias='champion')
        except Exception as exc:
            logger.warning('MLflow champion inference service load failed: %s', exc, exc_info=False)
            return None
    if _service is not None:
        _set_service_model_info(_service, model_dir or _service_model_dir(_service))
    return _service


def set_inference_service(service: InferenceService | None) -> None:
    global _service, _loaded
    _service = service
    _loaded = True


def reset_inference_service() -> None:
    global _service, _loaded
    _service = None
    _loaded = False


def _looks_like_model_dir(model_dir: Path) -> bool:
    return any((model_dir / name).exists() for name in ('metadata.json', 'pca_detector.joblib', 'lstm_ae.keras'))


def _set_service_model_info(service: InferenceService, model_dir: str) -> None:
    metadata = getattr(service, 'metadata', {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    set_model_info(
        {
            'name': _metadata_value(metadata, 'name') or _metadata_value(metadata, 'registered_model_name') or 'PumpAD',
            'version': getattr(service, 'model_version', '') or _metadata_value(metadata, 'version'),
            'alias': _metadata_value(metadata, 'alias') or 'champion',
            'model_dir': model_dir or _service_model_dir(service),
            'run_id': getattr(service, 'run_id', '') or _metadata_value(metadata, 'run_id'),
        }
    )


def _service_model_dir(service: InferenceService) -> str:
    value = getattr(service, 'model_dir', '') or getattr(service, 'artifact_dir', '')
    return '' if value is None else str(value)


def _metadata_value(metadata: Mapping[str, Any], key: str) -> Any:
    return metadata.get(key)
