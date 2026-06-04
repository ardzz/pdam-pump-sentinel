import sys
import types
from importlib import import_module
from types import SimpleNamespace
from typing import Any, cast

import numpy as np


def _mlflow_client():
    return import_module('ml.registry.mlflow_client')


class _FakeRun:
    info = SimpleNamespace(run_id='run-observable')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_system_metrics_helper_enables_mlflow_once(monkeypatch):
    client = _mlflow_client()
    monkeypatch.setattr(client, '_SYSTEM_METRICS_LOGGING_ATTEMPTED', False)
    calls = []
    mlflow = SimpleNamespace(
        enable_system_metrics_logging=lambda: calls.append('enable'),
        set_system_metrics_sampling_interval=lambda value: calls.append(('interval', value)),
        set_system_metrics_samples_before_logging=lambda value: calls.append(('samples', value)),
    )

    assert client._enable_system_metrics_logging_if_available(mlflow) is True
    assert client._enable_system_metrics_logging_if_available(mlflow) is True

    assert calls == [('interval', 1), ('samples', 1), 'enable']


def test_set_run_traceability_tags_logs_expected_keys(monkeypatch):
    client = _mlflow_client()
    logged = []
    mlflow = SimpleNamespace(set_tags=lambda tags: logged.append(dict(tags)))

    def git_output(*args):
        values = {
            ('rev-parse', 'HEAD'): 'abc123',
            ('branch', '--show-current'): 'main',
            ('status', '--short'): '',
        }
        return values[args]

    monkeypatch.setattr(client, '_git_output', git_output)
    monkeypatch.setattr(client, '_package_version', lambda package_name: f'{package_name}-1.0')

    tags = client.set_run_traceability_tags(mlflow)

    expected_keys = {
        'git.commit.sha',
        'git.branch',
        'git.is_dirty',
        'python.version',
        'host.name',
        'package.mlflow',
        'package.scikit-learn',
        'package.tensorflow',
        'package.xgboost',
        'package.lightgbm',
    }
    assert expected_keys <= set(tags)
    assert logged == [tags]


def test_log_pca_training_run_logs_model_signature_and_input_example(monkeypatch, tmp_path):
    client = _mlflow_client()
    calls: dict[str, list[Any]] = {'models': [], 'tags': []}
    mlflow = types.ModuleType('mlflow')
    sklearn = types.ModuleType('mlflow.sklearn')
    mlflow_models = types.ModuleType('mlflow.models')
    mlflow_any = cast(Any, mlflow)
    sklearn_any = cast(Any, sklearn)
    models_any = cast(Any, mlflow_models)

    mlflow_any.active_run = lambda: None
    mlflow_any.start_run = lambda: _FakeRun()
    mlflow_any.set_tags = lambda tags: calls['tags'].append(dict(tags))
    mlflow_any.log_params = lambda params: None
    mlflow_any.log_metrics = lambda metrics: None
    mlflow_any.log_artifacts = lambda path: None
    mlflow_any.enable_system_metrics_logging = lambda: None
    mlflow_any.set_system_metrics_sampling_interval = lambda value: None
    mlflow_any.set_system_metrics_samples_before_logging = lambda value: None
    models_any.infer_signature = lambda input_example, output_example: SimpleNamespace(inputs='in', outputs='out')

    def log_model(model, *, name, registered_model_name=None, signature=None, input_example=None):
        calls['models'].append(
            {
                'name': name,
                'registered_model_name': registered_model_name,
                'signature': signature,
                'input_example': input_example,
            }
        )
        return SimpleNamespace(registered_model_version=None)

    sklearn_any.log_model = log_model
    mlflow_any.sklearn = sklearn
    mlflow_any.models = mlflow_models
    monkeypatch.setitem(sys.modules, 'mlflow', mlflow)
    monkeypatch.setitem(sys.modules, 'mlflow.sklearn', sklearn)
    monkeypatch.setitem(sys.modules, 'mlflow.models', mlflow_models)
    monkeypatch.setattr(client, '_SYSTEM_METRICS_LOGGING_ATTEMPTED', False)
    monkeypatch.setattr(client, '_git_output', lambda *args: 'tracked')
    monkeypatch.setattr(client, '_package_version', lambda package_name: '1.0')

    output_dir = tmp_path / 'model-output'
    output_dir.mkdir()
    input_example = np.asarray([[1.0, 2.0]])
    result = SimpleNamespace(
        output_dir=output_dir,
        params={},
        metrics={},
        input_example=input_example,
        output_example=np.asarray([0.1]),
    )
    config = SimpleNamespace(register_model=False, registered_model_name='PumpAD')

    run_id = client.log_pca_training_run(result, object(), config)

    assert run_id == 'run-observable'
    assert calls['models'][0]['signature'].inputs == 'in'
    assert calls['models'][0]['input_example'] is input_example
    assert {'git.commit.sha', 'python.version', 'host.name', 'package.mlflow'} <= set(calls['tags'][0])
