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
        'tags': [],
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

    def log_metric(key, value, step=None):
        calls.setdefault('metric', []).append((key, value, step))

    def log_artifacts(local_dir):
        calls['artifacts'].append(local_dir)

    def set_registered_model_alias(name, alias, version):
        calls['aliases'].append((name, alias, version))

    def set_tags(tags):
        calls['tags'].append(dict(tags))

    mlflow_any.set_tracking_uri = set_tracking_uri
    mlflow_any.start_run = start_run
    mlflow_any.log_params = log_params
    mlflow_any.log_metrics = log_metrics
    mlflow_any.log_metric = log_metric
    mlflow_any.log_artifacts = log_artifacts
    mlflow_any.set_registered_model_alias = set_registered_model_alias
    mlflow_any.set_tags = set_tags

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


def _install_fake_mlflow_alias_metadata(monkeypatch, *, version=None, error: Exception | None = None):
    calls: dict[str, list[Any]] = {
        'tracking_uris': [],
        'aliases': [],
        'downloads': [],
    }
    mlflow = types.ModuleType('mlflow')
    mlflow.__path__ = []
    mlflow_any = cast(Any, mlflow)

    def set_tracking_uri(uri):
        calls['tracking_uris'].append(uri)

    mlflow_any.set_tracking_uri = set_tracking_uri

    tracking = types.ModuleType('mlflow.tracking')
    tracking_any = cast(Any, tracking)
    model_version = version or SimpleNamespace(
        name='PumpAD',
        version='12',
        run_id='run-123',
        source='runs:/run-123/pca_anomaly_model',
        aliases=['champion'],
        status='READY',
        status_message='ready',
        creation_timestamp=1000,
        last_updated_timestamp=2000,
        tags={'model_family': 'pca'},
    )

    class FakeMlflowClient:
        def get_model_version_by_alias(self, model_name, alias):
            calls['aliases'].append((model_name, alias))
            if error is not None:
                raise error
            return model_version

    tracking_any.MlflowClient = FakeMlflowClient
    mlflow_any.tracking = tracking

    monkeypatch.setitem(sys.modules, 'mlflow', mlflow)
    monkeypatch.setitem(sys.modules, 'mlflow.tracking', tracking)
    monkeypatch.delitem(sys.modules, 'mlflow.artifacts', raising=False)
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
    assert calls['tags'][0]['observability.schema_version'].startswith('2026-06-08')
    assert 'git.commit.sha' in calls['tags'][0]


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


def test_get_model_alias_metadata_reads_alias_without_artifact_download(monkeypatch):
    calls = _install_fake_mlflow_alias_metadata(monkeypatch)
    client = _mlflow_client()
    original_import_module = client.import_module
    imports: list[str] = []

    def recording_import_module(name):
        imports.append(name)
        if name == 'mlflow.artifacts':
            raise AssertionError('metadata lookup must not import mlflow.artifacts')
        return original_import_module(name)

    monkeypatch.setattr(client, 'import_module', recording_import_module)
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'file:///tmp/mlruns-metadata')

    metadata = client.get_model_alias_metadata('PumpAD', 'champion')

    assert metadata == {
        'name': 'PumpAD',
        'registered_model_name': 'PumpAD',
        'alias': 'champion',
        'version': '12',
        'mlflow_version': '12',
        'run_id': 'run-123',
        'source': 'runs:/run-123/pca_anomaly_model',
        'artifact_path': 'pca_anomaly_model',
        'aliases': ['champion'],
        'status': 'READY',
        'status_message': 'ready',
        'creation_timestamp': 1000,
        'last_updated_timestamp': 2000,
        'tags': {'model_family': 'pca'},
    }
    assert calls['tracking_uris'] == ['file:///tmp/mlruns-metadata']
    assert calls['aliases'] == [('PumpAD', 'champion')]
    assert calls['downloads'] == []
    assert 'mlflow.artifacts' not in imports
    assert 'mlflow.artifacts' not in sys.modules


def test_get_model_alias_metadata_keeps_non_run_sources_opaque(monkeypatch):
    version = SimpleNamespace(
        name='PumpAD',
        version='13',
        run_id='run-456',
        source='s3://bucket/models/pca',
        tags={},
    )
    _install_fake_mlflow_alias_metadata(monkeypatch, version=version)

    metadata = _mlflow_client().get_model_alias_metadata('PumpAD', 'candidate')

    assert metadata is not None
    assert metadata['source'] == 's3://bucket/models/pca'
    assert metadata['run_id'] == 'run-456'
    assert 'artifact_path' not in metadata


def test_get_model_alias_metadata_returns_none_when_alias_lookup_fails(monkeypatch):
    _install_fake_mlflow_alias_metadata(monkeypatch, error=RuntimeError('missing alias'))

    metadata = _mlflow_client().get_model_alias_metadata('PumpAD', 'champion')

    assert metadata is None


def test_log_supervised_label_evidence_to_active_run_logs_bounded_tags_and_metrics(monkeypatch):
    calls = _install_fake_mlflow(monkeypatch)
    mlflow = cast(Any, sys.modules['mlflow'])
    mlflow.active_run = lambda: object()
    evidence = {
        'source': 'operator_labels',
        'include_labels': True,
        'train': {'sample_count': 4, 'normal_count': 2, 'anomaly_count': 2, 'two_class': True},
        'validation': {'sample_count': 2, 'normal_count': 1, 'anomaly_count': 1, 'two_class': True},
    }

    _mlflow_client().log_supervised_label_evidence_to_active_run(
        evidence,
        model_family='xgboost',
        data_source='live_clickhouse',
        skip_reason=None,
    )

    assert calls['tags'] == [
        {
            'supervised.label_source': 'operator_labels',
            'supervised.model_family': 'xgboost',
            'supervised.data_source': 'live_clickhouse',
            'supervised.include_labels': 'true',
            'supervised.evidence_status': 'accepted',
        }
    ]
    assert calls['metrics'] == [
        {
            'supervised_label_train_sample_count': 4.0,
            'supervised_label_train_normal_count': 2.0,
            'supervised_label_train_anomaly_count': 2.0,
            'supervised_label_validation_sample_count': 2.0,
            'supervised_label_validation_normal_count': 1.0,
            'supervised_label_validation_anomaly_count': 1.0,
            'supervised_label_train_two_class': 1.0,
            'supervised_label_validation_two_class': 1.0,
        }
    ]


def test_log_supervised_label_evidence_to_active_run_skips_without_active_run(monkeypatch):
    calls = _install_fake_mlflow(monkeypatch)
    mlflow = cast(Any, sys.modules['mlflow'])
    mlflow.active_run = lambda: None

    _mlflow_client().log_supervised_label_evidence_to_active_run(
        {'include_labels': False},
        model_family='lightgbm',
        data_source='live_clickhouse',
        skip_reason='supervised_operator_labels_unavailable',
    )

    assert calls['tags'] == []
    assert calls['metrics'] == []
