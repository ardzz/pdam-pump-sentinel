import json
from importlib import import_module

import numpy as np


def _skab_loader():
    return import_module('ml.datasets.skab_loader')


def _joblib():
    return import_module('joblib')


def _pandas():
    return import_module('pandas')


def _train_lstm_ae():
    return import_module('ml.training.train_lstm_ae')


def _write_skab_csv(path, row_count=15, anomaly_start=None, offset=0.0, changepoint_indices=None):
    sensor_columns = _skab_loader().SENSOR_COLUMNS
    changepoint_indices = set(changepoint_indices or [])
    rows = [['datetime', *sensor_columns, 'anomaly', 'changepoint']]
    for row_index in range(row_count):
        anomaly = int(anomaly_start is not None and row_index >= anomaly_start)
        changepoint = int(row_index in changepoint_indices)
        sensor_values = []
        for column_index, _ in enumerate(sensor_columns):
            baseline = 1.0 + offset + row_index * 0.02 + column_index * 0.05
            sensor_values.append(f'{baseline + anomaly * 30.0:.6f}')
        rows.append([
            f'2024-01-01T00:00:{row_index:02d}Z',
            *sensor_values,
            str(anomaly),
            str(changepoint),
        ])
    path.write_text('\n'.join(';'.join(row) for row in rows) + '\n')
    return path


def _write_split_manifest(path, train, validation, test):
    payload = {
        'train': [str(p.relative_to(path.parent)) for p in train],
        'validation': [str(p.relative_to(path.parent)) for p in validation],
        'test': [str(p.relative_to(path.parent)) for p in test],
    }
    path.write_text(json.dumps(payload, indent=2) + '\n')
    return path


def test_train_lstm_ae_writes_artifact_contract_and_promotion_metrics(tmp_path):
    train_lstm_ae = _train_lstm_ae()
    train = _write_skab_csv(tmp_path / 'train.csv', anomaly_start=9)
    validation = _write_skab_csv(tmp_path / 'validation.csv', anomaly_start=9, offset=1.0, changepoint_indices=[9])
    test = _write_skab_csv(tmp_path / 'test.csv', anomaly_start=9, offset=2.0, changepoint_indices=[9])
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train], [validation], [test])
    output_dir = tmp_path / 'artifacts'

    result = train_lstm_ae.train_lstm_ae_from_skab(
        train_lstm_ae.LstmAeTrainingConfig(
            input_path=tmp_path / 'unused.csv',
            output_dir=output_dir,
            split_manifest_path=manifest,
            window_size=3,
            stride=3,
            threshold_quantile=0.9,
            lstm_units=2,
            latent_dim=1,
            epochs=1,
            batch_size=2,
            patience=1,
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
    assert result.sensor_columns == tuple(_skab_loader().SENSOR_COLUMNS)
    assert result.thresholds['threshold'] == result.thresholds['t2_threshold'] == result.thresholds['q_threshold']

    metadata = json.loads(result.artifact_paths['metadata'].read_text())
    assert metadata['model_family'] == 'lstm_ae'
    assert metadata['params']['window_size'] == 3
    assert metadata['params']['stride'] == 3
    assert metadata['params']['threshold_quantile'] == 0.9
    assert metadata['params']['score_type'] == 'mae'
    assert metadata['params']['seed'] == 123
    assert metadata['params']['sensor_columns'] == _skab_loader().SENSOR_COLUMNS
    assert metadata['thresholds']['threshold'] == result.thresholds['threshold']
    assert metadata['split']['train_count'] == 5
    assert metadata['split']['validation_count'] == 5
    assert metadata['split']['test_count'] == 5

    metrics = json.loads(result.artifact_paths['metrics'].read_text())
    for metric_name in ('precision', 'recall', 'f1', 'false_alarm_rate', 'pr_auc', 'roc_auc', 'test_f1'):
        assert metric_name in metrics
    assert metrics['training_sample_count'] == 5
    assert metrics['training_normal_count'] == 3
    assert metrics['sample_count'] == 5
    assert metrics['test_sample_count'] == 5

    scores = _pandas().read_csv(result.artifact_paths['scores'])
    assert list(scores.columns) == ['timestamp', 'label', 'changepoint', 'prediction', 'score']
    assert scores['label'].tolist() == [0, 0, 0, 1, 1]
    assert scores['changepoint'].tolist() == [0, 0, 0, 1, 0]

    scaler = _joblib().load(result.artifact_paths['scaler'])
    train_frame = _skab_loader().load_skab_csv(train)
    expected_mean = train_frame.loc[train_frame['anomaly'] == 0, _skab_loader().SENSOR_COLUMNS].to_numpy().mean(axis=0)
    np.testing.assert_allclose(scaler.mean_, expected_mean)

    should_promote = import_module('ml.monitoring.champion_challenger').should_promote
    champion = {
        'f1': max(0.0, float(result.metrics['f1']) - 0.03),
        'false_alarm_rate': float(result.metrics['false_alarm_rate']),
    }
    promoted, reason = should_promote(champion, result.metrics)
    assert isinstance(promoted, bool)
    assert 'missing f1 or false_alarm_rate' not in reason
