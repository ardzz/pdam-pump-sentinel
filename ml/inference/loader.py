from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

from ml.inference.lstm_ae_inference import LstmAeAnomalyInferenceService
from ml.inference.pca_inference import PcaAnomalyInferenceService
from ml.inference.supervised_inference import SupervisedAnomalyInferenceService

IsoForestAnomalyInferenceService: Any = import_module(
    'ml.inference.isoforest_inference'
).IsoForestAnomalyInferenceService


def load_inference_service_from_artifacts(model_dir: str | Path, model_version: str | None = None):
    directory = Path(model_dir)
    model_family = _read_model_family(directory / 'metadata.json')
    if model_family in ('', 'pca'):
        return PcaAnomalyInferenceService.from_artifacts(model_dir, model_version)
    if model_family == 'isolation_forest':
        return IsoForestAnomalyInferenceService.from_artifacts(model_dir, model_version)
    if model_family == 'lstm_ae':
        return LstmAeAnomalyInferenceService.from_artifacts(model_dir, model_version)
    if model_family in ('xgboost', 'lightgbm'):
        return SupervisedAnomalyInferenceService.from_artifacts(model_dir, model_version)
    raise ValueError(f'unsupported model_family: {model_family}')


def _read_model_family(metadata_path: Path) -> str:
    if not metadata_path.exists():
        return 'pca'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    return str(metadata.get('model_family') or 'pca')


__all__ = ['load_inference_service_from_artifacts']
