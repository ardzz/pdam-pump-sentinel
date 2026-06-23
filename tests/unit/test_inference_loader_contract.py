import json
import threading
from types import SimpleNamespace

import pytest

from app.services import inference
from ml.inference import loader
from ml.registry import mlflow_client


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


class _RefreshService:
    def __init__(self, name: str, *, version: str, run_id: str, alias: str = 'champion'):
        self.name = name
        self.model_version = version
        self.run_id = run_id
        self.metadata = {
            'registered_model_name': 'PumpAD',
            'alias': alias,
            'version': version,
            'mlflow_version': version,
            'run_id': run_id,
        }

    def observe(self, station, timestamp, sensors):
        return SimpleNamespace(station=station, timestamp=timestamp, model_version=self.model_version)


def _alias_metadata(version: str, run_id: str, *, alias: str = 'champion') -> dict[str, object]:
    return {
        'name': 'PumpAD',
        'registered_model_name': 'PumpAD',
        'alias': alias,
        'version': version,
        'mlflow_version': version,
        'run_id': run_id,
        'source': f'runs:/{run_id}/pca_anomaly_model',
        'artifact_path': 'pca_anomaly_model',
    }


def test_refresh_inference_service_from_alias_swaps_changed_alias_after_loading(monkeypatch):
    inference.reset_inference_service()
    incumbent = _RefreshService('old', version='1', run_id='run-old')
    replacement = _RefreshService('new', version='2', run_id='run-new')
    model_info_calls = []
    load_calls = []
    try:
        inference.set_inference_service(incumbent)
        monkeypatch.setattr(inference, 'set_model_info', lambda payload: model_info_calls.append(dict(payload)))
        monkeypatch.setattr(
            mlflow_client,
            'get_model_alias_metadata',
            lambda model_name, alias: _alias_metadata('2', 'run-new', alias=alias),
        )

        def load_champion_service(*, model_name, alias):
            load_calls.append((model_name, alias, inference.get_inference_service()))
            return replacement

        monkeypatch.setattr(mlflow_client, 'load_champion_service', load_champion_service)

        refreshed = inference.refresh_inference_service_from_alias()

        assert refreshed is replacement
        assert inference.get_inference_service() is replacement
        assert load_calls == [('PumpAD', 'champion', incumbent)]
        assert replacement.metadata['run_id'] == 'run-new'
        assert replacement.metadata['artifact_path'] == 'pca_anomaly_model'
        assert model_info_calls[-1]['version'] == '2'
        assert model_info_calls[-1]['run_id'] == 'run-new'
    finally:
        inference.reset_inference_service()


def test_refresh_inference_service_from_alias_noops_when_alias_signature_is_unchanged(monkeypatch):
    inference.reset_inference_service()
    incumbent = _RefreshService('old', version='1', run_id='run-old')
    try:
        inference.set_inference_service(incumbent)
        monkeypatch.setattr(
            mlflow_client,
            'get_model_alias_metadata',
            lambda model_name, alias: _alias_metadata('1', 'run-old', alias=alias),
        )
        monkeypatch.setattr(
            mlflow_client,
            'load_champion_service',
            lambda *, model_name, alias: (_ for _ in ()).throw(AssertionError('unchanged alias should not load')),
        )

        refreshed = inference.refresh_inference_service_from_alias()

        assert refreshed is incumbent
        assert inference.get_inference_service() is incumbent
    finally:
        inference.reset_inference_service()


def test_refresh_inference_service_from_alias_keeps_incumbent_on_metadata_failure(monkeypatch):
    inference.reset_inference_service()
    incumbent = _RefreshService('old', version='1', run_id='run-old')
    try:
        inference.set_inference_service(incumbent)
        monkeypatch.setattr(mlflow_client, 'get_model_alias_metadata', lambda model_name, alias: None)
        monkeypatch.setattr(
            mlflow_client,
            'load_champion_service',
            lambda *, model_name, alias: (_ for _ in ()).throw(AssertionError('metadata failure should not load')),
        )

        refreshed = inference.refresh_inference_service_from_alias()

        assert refreshed is incumbent
        assert inference.get_inference_service() is incumbent
    finally:
        inference.reset_inference_service()


def test_refresh_inference_service_from_alias_keeps_incumbent_on_load_failure(monkeypatch):
    inference.reset_inference_service()
    incumbent = _RefreshService('old', version='1', run_id='run-old')
    try:
        inference.set_inference_service(incumbent)
        monkeypatch.setattr(
            mlflow_client,
            'get_model_alias_metadata',
            lambda model_name, alias: _alias_metadata('2', 'run-new', alias=alias),
        )
        monkeypatch.setattr(mlflow_client, 'load_champion_service', lambda *, model_name, alias: None)

        refreshed = inference.refresh_inference_service_from_alias()

        assert refreshed is incumbent
        assert inference.get_inference_service() is incumbent
    finally:
        inference.reset_inference_service()


def test_refresh_inference_service_from_alias_does_not_block_readers_or_publish_partial_service(monkeypatch):
    inference.reset_inference_service()
    incumbent = _RefreshService('old', version='1', run_id='run-old')
    replacement = _RefreshService('new', version='2', run_id='run-new')
    load_started = threading.Event()
    release_load = threading.Event()
    result_holder = {}
    try:
        inference.set_inference_service(incumbent)
        monkeypatch.setattr(
            mlflow_client,
            'get_model_alias_metadata',
            lambda model_name, alias: _alias_metadata('2', 'run-new', alias=alias),
        )

        def load_champion_service(*, model_name, alias):
            load_started.set()
            release_load.wait(timeout=5)
            return replacement

        monkeypatch.setattr(mlflow_client, 'load_champion_service', load_champion_service)

        thread = threading.Thread(
            target=lambda: result_holder.setdefault('service', inference.refresh_inference_service_from_alias())
        )
        thread.start()
        assert load_started.wait(timeout=5)

        assert inference.get_inference_service() is incumbent

        release_load.set()
        thread.join(timeout=5)

        assert result_holder['service'] is replacement
        assert inference.get_inference_service() is replacement
    finally:
        release_load.set()
        inference.reset_inference_service()
