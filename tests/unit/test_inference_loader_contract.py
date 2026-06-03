import json

import pytest

from ml.inference import loader


def test_loader_defaults_missing_model_family_to_pca(tmp_path, monkeypatch):
    calls = []
    (tmp_path / 'metadata.json').write_text(json.dumps({'params': {'window_size': 2}}) + '\n')
    sentinel = object()

    def from_artifacts(model_dir, model_version=None):
        calls.append((model_dir, model_version))
        return sentinel

    monkeypatch.setattr(loader.PcaAnomalyInferenceService, 'from_artifacts', staticmethod(from_artifacts))

    service = loader.load_inference_service_from_artifacts(tmp_path, model_version='pca-v1')

    assert service is sentinel
    assert calls == [(tmp_path, 'pca-v1')]


def test_loader_dispatches_lstm_ae_family(tmp_path, monkeypatch):
    calls = []
    (tmp_path / 'metadata.json').write_text(json.dumps({'model_family': 'lstm_ae'}) + '\n')
    sentinel = object()

    def from_artifacts(model_dir, model_version=None):
        calls.append((model_dir, model_version))
        return sentinel

    monkeypatch.setattr(loader.LstmAeAnomalyInferenceService, 'from_artifacts', staticmethod(from_artifacts))

    service = loader.load_inference_service_from_artifacts(tmp_path, model_version='lstm-v1')

    assert service is sentinel
    assert calls == [(tmp_path, 'lstm-v1')]


def test_loader_rejects_unknown_model_family(tmp_path):
    (tmp_path / 'metadata.json').write_text(json.dumps({'model_family': 'unknown'}) + '\n')

    with pytest.raises(ValueError, match='unsupported model_family'):
        loader.load_inference_service_from_artifacts(tmp_path)
