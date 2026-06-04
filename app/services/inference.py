from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

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
