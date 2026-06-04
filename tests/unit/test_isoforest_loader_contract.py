import json

from ml.inference import loader


def test_loader_dispatches_isolation_forest_family(tmp_path, monkeypatch):
    calls = []
    (tmp_path / 'metadata.json').write_text(json.dumps({'model_family': 'isolation_forest'}) + '\n')
    sentinel = object()

    def from_artifacts(model_dir, model_version=None):
        calls.append((model_dir, model_version))
        return sentinel

    monkeypatch.setattr(loader.IsoForestAnomalyInferenceService, 'from_artifacts', staticmethod(from_artifacts))

    service = loader.load_inference_service_from_artifacts(tmp_path, model_version='if-v1')

    assert service is sentinel
    assert calls == [(tmp_path, 'if-v1')]
