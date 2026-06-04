import json
import sys
import types
from importlib import import_module
from types import SimpleNamespace
from typing import Any, cast

import pytest


def _metrics():
    return import_module('ml.evaluation.metrics')


def _pandas():
    return import_module('pandas')


def _skab_loader():
    return import_module('ml.datasets.skab_loader')


def _joblib():
    return import_module('joblib')


def _train_supervised():
    return import_module('ml.training.train_supervised')


class _FakeRun:
    info = SimpleNamespace(run_id='run-curve')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _write_skab_csv(path, row_count=24, anomaly_start=None, offset=0.0, changepoint_indices=None):
    sensor_columns = _skab_loader().SENSOR_COLUMNS
    changepoint_indices = set(changepoint_indices or [])
    rows = [['datetime', *sensor_columns, 'anomaly', 'changepoint']]
    for row_index in range(row_count):
        anomaly = int(anomaly_start is not None and row_index >= anomaly_start)
        sensor_values = []
        for column_index, _column in enumerate(sensor_columns):
            baseline = 2.0 + offset + row_index * 0.05 + column_index * 0.15
            sensor_values.append(f'{baseline + anomaly * (8.0 + column_index):.6f}')
        rows.append([
            f'2024-01-01T00:00:{row_index:02d}Z',
            *sensor_values,
            str(anomaly),
            str(int(row_index in changepoint_indices)),
        ])
    path.write_text('\n'.join(';'.join(row) for row in rows) + '\n')
    return path


def _write_split_manifest(path, train, validation, test):
    payload = {
        'train': [str(item.relative_to(path.parent)) for item in train],
        'validation': [str(item.relative_to(path.parent)) for item in validation],
        'test': [str(item.relative_to(path.parent)) for item in test],
    }
    path.write_text(json.dumps(payload, indent=2) + '\n')
    return path


@pytest.mark.parametrize('model_type', ['lightgbm', 'xgboost'])
def test_train_supervised_writes_artifacts_and_evaluate_split_metrics(tmp_path, model_type):
    train_supervised = _train_supervised()
    train = _write_skab_csv(tmp_path / 'train.csv', anomaly_start=12, offset=0.0)
    validation = _write_skab_csv(
        tmp_path / 'validation.csv', anomaly_start=12, offset=100.0, changepoint_indices=[11]
    )
    test = _write_skab_csv(tmp_path / 'test.csv', anomaly_start=12, offset=200.0, changepoint_indices=[12])
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train], [validation], [test])
    output_dir = tmp_path / 'artifacts'

    result = train_supervised.train_supervised_from_skab(
        train_supervised.SupervisedTrainingConfig(
            input_path=tmp_path / 'unused.csv',
            output_dir=output_dir,
            split_manifest_path=manifest,
            window_size=4,
            stride=4,
            scaler='standard',
            feature_mode='enriched',
            model_type=model_type,
            n_estimators=12,
            learning_rate=0.2,
            early_stopping_rounds=3,
            seed=123,
        )
    )

    assert set(result.artifact_paths) == {
        'model',
        'scaler',
        'metadata',
        'metrics',
        'scores',
        'split_manifest',
        'test_scores',
    }
    for path in result.artifact_paths.values():
        assert path.exists()
        assert path.parent == output_dir
    assert result.artifact_paths['model'].name == f'{model_type}.joblib'
    assert result.sensor_columns == tuple(_skab_loader().SENSOR_COLUMNS)
    assert result.thresholds['threshold'] == result.thresholds['t2_threshold'] == result.thresholds['q_threshold']

    metadata = json.loads(result.artifact_paths['metadata'].read_text())
    assert metadata['model_family'] == model_type
    assert metadata['params']['window_size'] == 4
    assert metadata['params']['threshold'] == result.thresholds['threshold']
    assert metadata['params']['feature_mode'] == 'enriched'
    assert metadata['params']['model_type'] == model_type
    assert metadata['params']['sensor_columns'] == _skab_loader().SENSOR_COLUMNS
    assert metadata['params']['feature_count'] == 94
    assert metadata['split']['train_count'] == 6
    assert metadata['split']['validation_count'] == 6
    assert metadata['split']['test_count'] == 6

    expected_keys = set(_metrics().evaluate_split([0, 1], [0, 1], [0.1, 0.9], transient_mask=[0, 1]))
    metrics = json.loads(result.artifact_paths['metrics'].read_text())
    assert expected_keys <= set(metrics)
    assert {f'test_{name}' for name in expected_keys} <= set(metrics)
    assert metrics['training_sample_count'] == 6
    assert metrics['training_anomaly_count'] == 3
    assert metrics['training_normal_count'] == 3
    assert metrics['sample_count'] == 6
    assert metrics['test_sample_count'] == 6
    assert 'accuracy' in metrics
    assert 'test_accuracy' in metrics

    scores = _pandas().read_csv(result.artifact_paths['scores'])
    assert list(scores.columns) == ['timestamp', 'label', 'changepoint', 'prediction', 'score']
    assert scores['label'].tolist() == [0, 0, 0, 1, 1, 1]
    assert scores['changepoint'].tolist() == [0, 0, 1, 0, 0, 0]

    model = _joblib().load(result.artifact_paths['model'])
    scaler = _joblib().load(result.artifact_paths['scaler'])
    assert model.n_features_in_ == 94
    assert scaler.n_features_in_ == 94


@pytest.mark.parametrize(
    ('model_type', 'evals_result', 'expected_key'),
    [
        ('xgboost', {'validation_0': {'logloss': [0.9, 0.7, 0.5]}}, 'val_xgb_logloss_round'),
        ('lightgbm', {'valid_0': {'binary_logloss': [0.8, 0.6, 0.4]}}, 'val_lgbm_binary_logloss_round'),
    ],
)
def test_log_supervised_training_run_streams_boosting_round_metrics(monkeypatch, tmp_path, model_type, evals_result, expected_key):
    train_supervised = _train_supervised()
    calls = {'metric': [], 'metrics': [], 'params': [], 'artifacts': [], 'models': []}
    mlflow = types.ModuleType('mlflow')
    sklearn = types.ModuleType('mlflow.sklearn')
    mlflow_any = cast(Any, mlflow)
    sklearn_any = cast(Any, sklearn)

    mlflow_any.start_run = lambda: _FakeRun()
    mlflow_any.log_params = lambda params: calls['params'].append(dict(params))
    mlflow_any.log_metrics = lambda metrics: calls['metrics'].append(dict(metrics))
    mlflow_any.log_metric = lambda key, value, step=None: calls['metric'].append((key, float(value), step))
    mlflow_any.log_artifacts = lambda path: calls['artifacts'].append(path)
    sklearn_any.log_model = lambda model, *, name, registered_model_name=None: calls['models'].append(
        (name, registered_model_name)
    )
    monkeypatch.setitem(sys.modules, 'mlflow', mlflow)
    monkeypatch.setitem(sys.modules, 'mlflow.sklearn', sklearn)

    output_dir = tmp_path / 'artifacts'
    output_dir.mkdir()
    result = SimpleNamespace(output_dir=output_dir, params={'model_type': model_type}, metrics={'f1': 0.25})
    model = SimpleNamespace(evals_result_=evals_result)
    config = train_supervised.SupervisedTrainingConfig(
        input_path=tmp_path / 'unused.csv',
        output_dir=output_dir,
        model_type=model_type,
        registered_model_name='PumpAD',
    )

    train_supervised._log_supervised_training_run_safely(result, model, config)

    round_calls = [call for call in calls['metric'] if call[0] == expected_key]
    assert len(round_calls) >= 2
    assert [call[2] for call in round_calls] == sorted(call[2] for call in round_calls)
    assert [call[2] for call in round_calls] == list(range(len(round_calls)))
