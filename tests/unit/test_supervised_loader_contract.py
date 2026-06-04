import json

import pytest

from ml.inference import loader


@pytest.mark.parametrize('model_family', ['lightgbm', 'xgboost'])
def test_loader_dispatches_supervised_gradient_boosting_families(tmp_path, monkeypatch, model_family):
    calls = []
    (tmp_path / 'metadata.json').write_text(json.dumps({'model_family': model_family}) + '\n')
    sentinel = object()

    def from_artifacts(model_dir, model_version=None):
        calls.append((model_dir, model_version))
        return sentinel

    monkeypatch.setattr(loader.SupervisedAnomalyInferenceService, 'from_artifacts', staticmethod(from_artifacts))

    service = loader.load_inference_service_from_artifacts(tmp_path, model_version='gbm-v1')

    assert service is sentinel
    assert calls == [(tmp_path, 'gbm-v1')]
