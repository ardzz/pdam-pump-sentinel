from __future__ import annotations

import os
from pathlib import Path

from ml.inference.pca_inference import PcaAnomalyInferenceService

MODEL_DIR_ENV = 'PUMPAD_MODEL_DIR'

_service: PcaAnomalyInferenceService | None = None
_loaded = False


def get_inference_service() -> PcaAnomalyInferenceService | None:
    global _service, _loaded
    if _loaded:
        return _service

    _loaded = True
    model_dir = os.getenv(MODEL_DIR_ENV)
    if model_dir and (Path(model_dir) / 'pca_detector.joblib').exists():
        _service = PcaAnomalyInferenceService.from_artifacts(model_dir)
    return _service


def set_inference_service(service: PcaAnomalyInferenceService | None) -> None:
    global _service, _loaded
    _service = service
    _loaded = True


def reset_inference_service() -> None:
    global _service, _loaded
    _service = None
    _loaded = False
