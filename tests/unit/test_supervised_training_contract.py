import json
import sys
import types
from importlib import import_module
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
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


def _live_windows():
    return import_module('ml.datasets.live_windows')


class _FakeRun:
    info = SimpleNamespace(run_id='run-curve')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeBooster:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.classes_ = np.asarray([0, 1])
        self.n_features_in_ = 94
        self.feature_importances_ = np.ones(94)
        self.train_y = np.empty((0,), dtype=int)

    def fit(self, train_x, train_y, eval_set=None, verbose=False, eval_names=None, eval_metric=None, callbacks=None):
        self.n_features_in_ = train_x.shape[1]
        self.train_y = np.asarray(train_y, dtype=int)
        return self

    def predict_proba(self, features):
        base = np.linspace(0.2, 0.8, len(features)) if len(features) else np.empty((0,), dtype=float)
        return np.column_stack([1.0 - base, base])


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


def _live_bundle(*, train_labels=(0, 0, 1, 1), validation_labels=(0, 1), include_labels=True):
    live_windows = _live_windows()
    windowing = import_module('ml.features.windowing')
    sensor_columns = tuple(_skab_loader().SENSOR_COLUMNS)

    def dataset(labels, offset):
        rows = []
        timestamps = []
        for index, label in enumerate(labels):
            base = offset + index * 10.0
            rows.append(np.asarray([[base + sensor + step for sensor in range(len(sensor_columns))] for step in range(4)], dtype=float).reshape(-1))
            timestamps.append(f'2026-06-01T00:00:{index:02d}Z')
        return windowing.WindowedSensorDataset(
            features=np.vstack(rows) if rows else np.empty((0, 4 * len(sensor_columns)), dtype=float),
            labels=np.asarray(labels, dtype=int),
            changepoints=np.zeros(len(labels), dtype=int),
            timestamps=np.asarray(timestamps, dtype=object),
            sensor_columns=sensor_columns,
            window_size=4,
            stride=4,
        )

    return live_windows.LiveWindowDatasetBundle(
        train=dataset(train_labels, 10.0),
        validation=dataset(validation_labels, 100.0),
        test=None,
        provenance={'data_source': 'live_clickhouse', 'include_labels': include_labels, 'split_counts': {'train': len(train_labels), 'validation': len(validation_labels), 'test': 0}},
    )


@pytest.fixture
def fake_boosters(monkeypatch):
    original_import_module = _train_supervised().import_module

    def fake_import_module(name):
        if name == 'xgboost':
            return SimpleNamespace(XGBClassifier=_FakeBooster)
        if name == 'lightgbm':
            return SimpleNamespace(
                LGBMClassifier=_FakeBooster,
                early_stopping=lambda rounds, verbose=False: ('early_stopping', rounds, verbose),
            )
        return original_import_module(name)

    monkeypatch.setattr(_train_supervised(), 'import_module', fake_import_module)


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


def test_train_supervised_starts_mlflow_run_before_fit(monkeypatch, tmp_path):
    train_supervised = _train_supervised()
    state = {'active': False, 'fit_active': False, 'logged': False}
    mlflow = types.ModuleType('mlflow')
    sklearn = types.ModuleType('mlflow.sklearn')
    mlflow_any = cast(Any, mlflow)

    class Run:
        def __enter__(self):
            state['active'] = True
            return self

        def __exit__(self, exc_type, exc, traceback):
            state['active'] = False
            return False

    mlflow_any.start_run = lambda: Run()
    mlflow_any.active_run = lambda: object() if state['active'] else None
    monkeypatch.setitem(sys.modules, 'mlflow', mlflow)
    monkeypatch.setitem(sys.modules, 'mlflow.sklearn', sklearn)

    def fit_and_write(config):
        state['fit_active'] = state['active']
        output_dir = tmp_path / 'artifacts'
        output_dir.mkdir()
        result = SimpleNamespace(output_dir=output_dir, params={}, metrics={})
        return result, object()

    def log_run(result, model, config):
        state['logged'] = state['active']

    monkeypatch.setattr(train_supervised, '_fit_and_write_artifacts', fit_and_write)
    monkeypatch.setattr(train_supervised, '_log_supervised_training_run_safely', log_run)

    train_supervised.train_supervised_from_skab(
        train_supervised.SupervisedTrainingConfig(
            input_path=tmp_path / 'unused.csv',
            output_dir=tmp_path / 'artifacts',
            log_mlflow=True,
        )
    )

    assert state['fit_active'] is True
    assert state['logged'] is True


@pytest.mark.parametrize(
    ('model_type', 'callback_factory', 'payload', 'expected_keys'),
    [
        (
            'xgboost',
            'xgb',
            {'validation_0': {'logloss': [0.9]}, 'validation_1': {'logloss': [0.7]}},
            ['train_xgb_logloss_round', 'val_xgb_logloss_round'],
        ),
        (
            'lightgbm',
            'lgbm',
            [('train', 'binary_logloss', 0.8, False), ('validation', 'binary_logloss', 0.6, False)],
            ['train_lgbm_binary_logloss_round', 'val_lgbm_binary_logloss_round'],
        ),
    ],
)
def test_boosting_callbacks_stream_round_metrics(monkeypatch, tmp_path, model_type, callback_factory, payload, expected_keys):
    train_supervised = _train_supervised()
    calls = []
    mlflow = types.ModuleType('mlflow')
    mlflow_any = cast(Any, mlflow)
    mlflow_any.active_run = lambda: object()
    mlflow_any.log_metric = lambda key, value, step=None: calls.append((key, float(value), step))
    monkeypatch.setitem(sys.modules, 'mlflow', mlflow)
    config = train_supervised.SupervisedTrainingConfig(
        input_path=tmp_path / 'unused.csv',
        output_dir=tmp_path / 'artifacts',
        model_type=model_type,
    )

    if callback_factory == 'xgb':
        class TrainingCallback:
            pass

        xgboost = SimpleNamespace(callback=SimpleNamespace(TrainingCallback=TrainingCallback))
        callback = train_supervised._xgboost_mlflow_callbacks(xgboost, config)[0]
        callback.after_iteration(None, 3, payload)
    else:
        callback = train_supervised._lightgbm_mlflow_callback(config)
        callback(SimpleNamespace(evaluation_result_list=payload, iteration=4))

    assert [call[0] for call in calls] == expected_keys
    assert all(call[2] in {3, 4} for call in calls)


def test_xgb_mlflow_callback_pickle_roundtrip():
    import pickle
    train_supervised = _train_supervised()
    callback = train_supervised.MlflowXGBCallback()
    restored = pickle.loads(pickle.dumps(callback))
    assert isinstance(restored, train_supervised.MlflowXGBCallback)
    assert restored._family_key == 'xgb'


@pytest.mark.parametrize(
    ('include_labels', 'train_labels', 'validation_labels', 'expected_reason'),
    [
        (False, (0, 1), (0, 1), 'supervised_operator_labels_unavailable'),
        (True, (0, 0), (0, 1), 'supervised_train_labels_need_two_classes'),
        (True, (0, 1), (0, 0), 'supervised_validation_labels_need_two_classes'),
        (True, (0, 2), (0, 1), 'supervised_operator_labels_invalid'),
    ],
)
def test_train_supervised_from_live_windows_returns_honest_skip_for_invalid_label_evidence(
    tmp_path,
    include_labels,
    train_labels,
    validation_labels,
    expected_reason,
):
    train_supervised = _train_supervised()
    result = train_supervised.train_supervised_from_live_windows(
        train_supervised.SupervisedTrainingConfig(
            input_path=tmp_path / 'unused.csv',
            output_dir=tmp_path / 'artifacts',
            window_size=4,
            stride=4,
            model_type='xgboost',
        ),
        _live_bundle(train_labels=train_labels, validation_labels=validation_labels, include_labels=include_labels),
    )

    assert result.skipped is True
    assert result.skip_reason == expected_reason
    assert result.deployment_eligible is False
    assert set(result.artifact_paths) == {'metadata', 'metrics'}
    metadata = json.loads(result.artifact_paths['metadata'].read_text())
    assert metadata['skipped'] is True
    assert metadata['deployment_eligible'] is False
    assert metadata['skip_reason'] == expected_reason
    assert metadata['label_evidence']['include_labels'] is include_labels


@pytest.mark.parametrize('model_type', ['xgboost', 'lightgbm'])
def test_train_supervised_from_live_windows_trains_two_class_candidates_with_live_metadata(
    tmp_path,
    fake_boosters,
    model_type,
):
    train_supervised = _train_supervised()
    result = train_supervised.train_supervised_from_live_windows(
        train_supervised.SupervisedTrainingConfig(
            input_path=tmp_path / 'unused.csv',
            output_dir=tmp_path / 'artifacts',
            window_size=4,
            stride=4,
            model_type=model_type,
            n_estimators=3,
            early_stopping_rounds=0,
            log_mlflow=False,
            register_model=False,
            alias=None,
        ),
        _live_bundle(),
    )

    assert result.skipped is False
    assert result.deployment_eligible is False
    assert result.params['data_source'] == 'live_clickhouse'
    assert result.params['deployment_eligible'] is False
    assert result.metrics['data_source'] == 'live_clickhouse'
    assert result.metrics['deployment_eligible'] is False
    assert result.metrics['label_train_two_class'] is True
    assert result.metrics['label_validation_two_class'] is True
    metadata = json.loads(result.artifact_paths['metadata'].read_text())
    assert metadata['model_family'] == model_type
    assert metadata['data_source'] == 'live_clickhouse'
    assert metadata['deployment_eligible'] is False
    assert metadata['skipped'] is False
    assert metadata['label_evidence']['train_two_class'] is True
    assert metadata['label_evidence']['validation_two_class'] is True
    assert metadata['split']['train_count'] == 4
    assert metadata['split']['validation_count'] == 2
