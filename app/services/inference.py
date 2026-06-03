from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from ml.inference.loader import load_inference_service_from_artifacts

MODEL_DIR_ENV = 'PUMPAD_MODEL_DIR'


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
        _service = load_inference_service_from_artifacts(model_dir)
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
