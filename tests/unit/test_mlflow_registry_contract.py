from __future__ import annotations

import sys
import types
from importlib import import_module
from types import SimpleNamespace
from typing import Any, cast


def _mlflow_client():
    return import_module('ml.registry.mlflow_client')


class _FakeRun:
    def __init__(self, run_id: str | None = None):
        self.info = SimpleNamespace(run_id=run_id) if run_id is not None else SimpleNamespace()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _install_fake_mlflow(monkeypatch, *, run_id: str | None = 'run-123', model_version: str | None = '7'):
    calls: dict[str, list[Any]] = {
        'tracking_uris': [],
        'params': [],
        'metrics': [],
        'artifacts': [],
        'models': [],
        'aliases': [],
        'searches': [],
        'start_runs': [],
    }

    mlflow = types.ModuleType('mlflow')
    mlflow.__path__ = []
    mlflow_any = cast(Any, mlflow)

    def set_tracking_uri(uri):
        calls['tracking_uris'].append(uri)

    def start_run():
        calls['start_runs'].append(True)
        return _FakeRun(run_id=run_id)

    def log_params(params):
        calls['params'].append(dict(params))

    def log_metrics(metrics):
        calls['metrics'].append(dict(metrics))

    def log_artifacts(local_dir):
        calls['artifacts'].append(local_dir)

    def set_registered_model_alias(name, alias, version):
        calls['aliases'].append((name, alias, version))

    mlflow_any.set_tracking_uri = set_tracking_uri
    mlflow_any.start_run = start_run
    mlflow_any.log_params = log_params
    mlflow_any.log_metrics = log_metrics
    mlflow_any.log_artifacts = log_artifacts
    mlflow_any.set_registered_model_alias = set_registered_model_alias

    sklearn = types.ModuleType('mlflow.sklearn')
    sklearn_any = cast(Any, sklearn)

    def log_model(model, *, name, registered_model_name=None):
        calls['models'].append(
            {
                'model': model,
                'name': name,
                'registered_model_name': registered_model_name,
            }
        )
        return SimpleNamespace(registered_model_version=model_version)

    sklearn_any.log_model = log_model
    mlflow_any.sklearn = sklearn

    tracking = types.ModuleType('mlflow.tracking')
    tracking_any = cast(Any, tracking)

    class FakeMlflowClient:
        def search_model_versions(self, filter_string):
            calls['searches'].append(filter_string)
            if model_version is None or not calls['models']:
                return []
            name = calls['models'][-1]['registered_model_name']
            return [SimpleNamespace(name=name, version=model_version)] if name else []

        def get_latest_versions(self, name):
            if model_version is None:
                return []
            return [SimpleNamespace(name=name, version=model_version)]

        def set_registered_model_alias(self, name, alias, version):
            calls['aliases'].append((name, alias, version))

    tracking_any.MlflowClient = FakeMlflowClient
    mlflow_any.tracking = tracking

    monkeypatch.setitem(sys.modules, 'mlflow', mlflow)
    monkeypatch.setitem(sys.modules, 'mlflow.sklearn', sklearn)
    monkeypatch.setitem(sys.modules, 'mlflow.tracking', tracking)
    return calls


def test_importing_helper_does_not_import_mlflow(monkeypatch):
    monkeypatch.delitem(sys.modules, 'mlflow', raising=False)
    monkeypatch.delitem(sys.modules, 'mlflow.sklearn', raising=False)

    _mlflow_client()

    assert 'mlflow' not in sys.modules
    assert 'mlflow.sklearn' not in sys.modules


def test_log_pca_training_run_logs_payload_artifacts_model_registration_and_alias(monkeypatch, tmp_path):
    calls = _install_fake_mlflow(monkeypatch)
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'file:///tmp/mlruns-contract')
    output_dir = tmp_path / 'pca-output'
    output_dir.mkdir()
    (output_dir / 'metrics.json').write_text('{"f1": 0.91}\n')
    detector = object()
    config = SimpleNamespace(
        n_components=0.9,
        window_size=12,
        stride=3,
        register_model=True,
        registered_model_name='PumpADCandidate',
        alias='champion',
    )
    result = SimpleNamespace(
        output_dir=output_dir,
        params={'train_windows': 120, 'source': 'skab'},
        metrics={'f1': 0.91, 'false_alarm_rate': 0.02, 'ignored_note': 'not-a-metric'},
    )

    run_id = _mlflow_client().log_pca_training_run(result, detector, config)

    assert run_id == 'run-123'
    assert calls['tracking_uris'] == ['file:///tmp/mlruns-contract']
    assert calls['params'] == [
        {
            'n_components': 0.9,
            'window_size': 12,
            'stride': 3,
            'register_model': True,
            'registered_model_name': 'PumpADCandidate',
            'alias': 'champion',
            'train_windows': 120,
            'source': 'skab',
        }
    ]
    assert calls['metrics'] == [{'f1': 0.91, 'false_alarm_rate': 0.02}]
    assert calls['artifacts'] == [str(output_dir)]
    assert calls['models'] == [
        {
            'model': detector,
            'name': 'pca_anomaly_model',
            'registered_model_name': 'PumpADCandidate',
        }
    ]
    assert calls['aliases'] == [('PumpADCandidate', 'champion', '7')]
    assert calls['searches'] == ["run_id='run-123'"]


def test_log_pca_training_run_skips_optional_registration_alias_and_missing_run_id(monkeypatch):
    calls = _install_fake_mlflow(monkeypatch, run_id=None, model_version=None)
    monkeypatch.delenv('MLFLOW_TRACKING_URI', raising=False)
    detector = object()
    config = SimpleNamespace(register_model=False, registered_model_name='ShouldNotRegister', alias='candidate')
    result = SimpleNamespace(params={'train_windows': 8}, metrics={'q_threshold': 1.25})

    run_id = _mlflow_client().log_pca_training_run(result, detector, config)

    assert run_id is None
    assert calls['tracking_uris'] == []
    assert calls['params'] == [
        {
            'register_model': False,
            'registered_model_name': 'ShouldNotRegister',
            'alias': 'candidate',
            'train_windows': 8,
        }
    ]
    assert calls['metrics'] == [{'q_threshold': 1.25}]
    assert calls['artifacts'] == []
    assert calls['models'] == [
        {
            'model': detector,
            'name': 'pca_anomaly_model',
            'registered_model_name': None,
        }
    ]
    assert calls['aliases'] == []
