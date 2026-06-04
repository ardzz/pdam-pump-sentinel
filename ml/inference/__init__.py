from importlib import import_module
from typing import Any

from ml.inference.loader import load_inference_service_from_artifacts
from ml.inference.lstm_ae_inference import LstmAeAnomalyInferenceService, LstmAeAnomalyVerdict
from ml.inference.pca_inference import AnomalyVerdict, PcaAnomalyInferenceService
from ml.inference.supervised_inference import SupervisedAnomalyInferenceService, SupervisedAnomalyVerdict

_isoforest_inference: Any = import_module('ml.inference.isoforest_inference')
IsoForestAnomalyInferenceService = _isoforest_inference.IsoForestAnomalyInferenceService
IsoForestAnomalyVerdict = _isoforest_inference.IsoForestAnomalyVerdict

__all__ = [
    'AnomalyVerdict',
    'IsoForestAnomalyInferenceService',
    'IsoForestAnomalyVerdict',
    'LstmAeAnomalyInferenceService',
    'LstmAeAnomalyVerdict',
    'PcaAnomalyInferenceService',
    'SupervisedAnomalyInferenceService',
    'SupervisedAnomalyVerdict',
    'load_inference_service_from_artifacts',
]
