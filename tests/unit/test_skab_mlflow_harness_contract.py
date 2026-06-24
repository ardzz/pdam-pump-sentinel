from __future__ import annotations

import math
import sys
import types
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast


def _harness():
    return import_module('scripts.run_skab_model_experiments')


class _FakeRun:
    def __init__(self, run_id: str = 'run-1'):
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _harness_config(harness, tmp_path, *, log_mlflow=False, tracking_uri=None, experiment_name=None):
    return harness.HarnessConfig(
        manifest_path=Path(tmp_path / 'manifest.json'),
        output_dir=Path(tmp_path / 'artifacts'),
        window_size=4,
        stride=1,
        seed=7,
        lstm_epochs=1,
        lstm_batch_size=2,
        lstm_patience=1,
        supervised_estimators=4,
        isoforest_estimators=5,
        oneclass_svm_nu=0.05,
        oneclass_svm_max_iter=200,
        distance_profile_max_reference_windows=8,
        distance_profile_max_distance_comparisons=1000,
        pca_ensemble_size=2,
        pca_ensemble_subset_ratio=0.5,
        pca_variant_feature_mode='spectral',
        skip_lstm=True,
        smoke=True,
        log_mlflow=log_mlflow,
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment_name=experiment_name or harness.DEFAULT_MLFLOW_EXPERIMENT_NAME,
    )


def _record(harness, tmp_path, *, key='01_pca_raw', status='completed'):
    output_dir = Path(tmp_path / key)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'metrics.json').write_text('{}\n', encoding='utf-8')
    return harness.ExperimentRecord(
        key=key,
        display_name=key,
        model_family='pca',
        feature_mode='raw',
        split_protocol=harness.SPLIT_PROTOCOL,
        status=status,
        output_dir=output_dir,
        metrics={
            'f1': 0.91,
            'test_f1': 0.81,
            'nan_metric': math.nan,
            'inf_metric': math.inf,
            'text_metric': 'ignored',
            'bool_metric': True,
        },
        artifact_paths={'metrics': output_dir / 'metrics.json'},
        metadata={'threshold_method': 'validation_normal_quantile', 'params': {'metadata_scalar': 3, 'nested': {'skip': 1}}},
        params={'record_scalar': 0.2, 'bad_param': math.inf, 'artifact_dir': output_dir},
    )


def _install_fake_mlflow(monkeypatch, *, active_run=None):
    calls: dict[str, list[Any]] = {
        'tracking_uris': [],
        'experiments': [],
        'start_runs': [],
        'tags': [],
        'params': [],
        'metrics': [],
        'artifacts': [],
    }
    mlflow = types.ModuleType('mlflow')
    mlflow_any = cast(Any, mlflow)

    def start_run(*, run_name=None, nested=False):
        calls['start_runs'].append({'run_name': run_name, 'nested': nested})
        return _FakeRun(run_id=f'run-{len(calls["start_runs"])}')

    mlflow_any.active_run = lambda: active_run
    mlflow_any.set_tracking_uri = lambda uri: calls['tracking_uris'].append(uri)
    mlflow_any.set_experiment = lambda name: calls['experiments'].append(name)
    mlflow_any.start_run = start_run
    mlflow_any.set_tags = lambda tags: calls['tags'].append(dict(tags))
    mlflow_any.log_params = lambda params: calls['params'].append(dict(params))
    mlflow_any.log_metrics = lambda metrics: calls['metrics'].append(dict(metrics))
    mlflow_any.log_artifacts = lambda path, artifact_path=None: calls['artifacts'].append((path, artifact_path))
    monkeypatch.setitem(sys.modules, 'mlflow', mlflow)
    return calls


def test_disabled_harness_mlflow_logging_does_not_import_or_call_mlflow(monkeypatch, tmp_path):
    harness = _harness()
    monkeypatch.delitem(sys.modules, 'mlflow', raising=False)

    logged = harness._log_records_to_mlflow(_harness_config(harness, tmp_path), [_record(harness, tmp_path)])

    assert logged == []
    assert 'mlflow' not in sys.modules


def test_enabled_harness_mlflow_logging_uses_local_file_uri_and_logs_safe_payload(monkeypatch, tmp_path):
    harness = _harness()
    calls = _install_fake_mlflow(monkeypatch)
    config = _harness_config(harness, tmp_path, log_mlflow=True, experiment_name='skab-contract')
    completed = _record(harness, tmp_path, key='01_pca_raw')
    skipped = _record(harness, tmp_path, key='04_lstm_ae', status='skipped')

    logged = harness._log_records_to_mlflow(config, [completed, skipped])

    assert logged == ['01_pca_raw']
    assert calls['tracking_uris'] == [(config.output_dir / 'mlruns').resolve().as_uri()]
    assert calls['experiments'] == ['skab-contract']
    assert calls['start_runs'] == [{'run_name': '01_pca_raw', 'nested': False}]
    assert calls['tags'] == [
        {
            'dataset': 'skab',
            'protocol': harness.SPLIT_PROTOCOL,
            'experiment_key': '01_pca_raw',
            'model_family': 'pca',
            'feature_mode': 'raw',
            'split_protocol': harness.SPLIT_PROTOCOL,
            'split_strategy': harness.SPLIT_PROTOCOL,
            'status': 'completed',
            'threshold_method': 'validation_normal_quantile',
        }
    ]
    assert calls['params'][0]['record_scalar'] == 0.2
    assert calls['params'][0]['metadata_scalar'] == 3
    assert calls['params'][0]['window_size'] == 4
    assert calls['params'][0]['manifest_path'] == str(config.manifest_path)
    assert calls['params'][0]['artifact_dir'] == str(completed.output_dir)
    assert 'nested' not in calls['params'][0]
    assert 'bad_param' not in calls['params'][0]
    assert calls['metrics'] == [{'f1': 0.91, 'test_f1': 0.81}]
    assert calls['artifacts'] == [(str(completed.output_dir), '01_pca_raw')]


def test_enabled_harness_mlflow_logging_starts_nested_child_runs_when_parent_is_active(monkeypatch, tmp_path):
    harness = _harness()
    calls = _install_fake_mlflow(monkeypatch, active_run=_FakeRun('parent'))
    config = _harness_config(harness, tmp_path, log_mlflow=True, tracking_uri='file:///tmp/skab-contract')

    harness._log_records_to_mlflow(config, [_record(harness, tmp_path)])

    assert calls['tracking_uris'] == ['file:///tmp/skab-contract']
    assert calls['start_runs'] == [{'run_name': '01_pca_raw', 'nested': True}]


def test_cli_mlflow_flags_are_opt_in_and_preserve_offline_defaults(tmp_path):
    harness = _harness()

    default_config = harness._parse_args(['--output-dir', str(tmp_path / 'default')])
    assert default_config.log_mlflow is False
    assert default_config.mlflow_tracking_uri is None
    assert default_config.mlflow_experiment_name == harness.DEFAULT_MLFLOW_EXPERIMENT_NAME

    enabled_config = harness._parse_args(
        [
            '--output-dir',
            str(tmp_path / 'enabled'),
            '--log-mlflow',
            '--mlflow-tracking-uri',
            'file:///tmp/custom-mlruns',
            '--experiment-name',
            'custom-skab',
        ]
    )
    assert enabled_config.log_mlflow is True
    assert enabled_config.mlflow_tracking_uri == 'file:///tmp/custom-mlruns'
    assert enabled_config.mlflow_experiment_name == 'custom-skab'
