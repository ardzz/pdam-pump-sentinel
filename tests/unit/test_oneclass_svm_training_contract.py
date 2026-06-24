import json
from importlib import import_module

import numpy as np


def _metrics():
    return import_module('ml.evaluation.metrics')


def _pandas():
    return import_module('pandas')


def _skab_loader():
    return import_module('ml.datasets.skab_loader')


def _joblib():
    return import_module('joblib')


def _train_oneclass_svm():
    return import_module('ml.training.train_oneclass_svm')


def _write_skab_csv(path, row_count=16, anomaly_start=None, offset=0.0, changepoint_indices=None):
    sensor_columns = _skab_loader().SENSOR_COLUMNS
    changepoint_indices = set(changepoint_indices or [])
    rows = [['datetime', *sensor_columns, 'anomaly', 'changepoint']]
    for row_index in range(row_count):
        anomaly = int(anomaly_start is not None and row_index >= anomaly_start)
        changepoint = int(row_index in changepoint_indices)
        sensor_values = []
        for column_index, _column in enumerate(sensor_columns):
            baseline = 2.0 + offset + row_index * 0.05 + column_index * 0.15
            sensor_values.append(f'{baseline + anomaly * 12.0:.6f}')
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


def test_train_oneclass_svm_writes_artifacts_and_uses_validation_normal_threshold(tmp_path):
    train_oneclass_svm = _train_oneclass_svm()
    train = _write_skab_csv(tmp_path / 'train.csv', anomaly_start=12, offset=0.0)
    validation = _write_skab_csv(
        tmp_path / 'validation.csv', anomaly_start=12, offset=100.0, changepoint_indices=[11]
    )
    test = _write_skab_csv(tmp_path / 'test.csv', anomaly_start=12, offset=200.0, changepoint_indices=[12])
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train], [validation], [test])
    output_dir = tmp_path / 'artifacts'

    config = train_oneclass_svm.OneClassSvmTrainingConfig(
        input_path=tmp_path / 'unused.csv',
        output_dir=output_dir,
        split_manifest_path=manifest,
        window_size=4,
        stride=4,
        threshold_quantile=0.9,
        scaler='standard',
        feature_mode='spectral',
        kernel='rbf',
        nu=0.05,
        gamma='scale',
        max_iter=200,
    )

    result = train_oneclass_svm.train_oneclass_svm_from_skab(config)

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
    assert metadata['model_family'] == 'oneclass_svm'
    assert metadata['score_orientation'] == 'higher_is_more_anomalous:-decision_function'
    assert metadata['params']['window_size'] == 4
    assert metadata['params']['threshold'] == result.thresholds['threshold']
    assert metadata['params']['feature_mode'] == 'spectral'
    assert metadata['params']['sensor_columns'] == _skab_loader().SENSOR_COLUMNS
    assert metadata['params']['kernel'] == 'rbf'
    assert metadata['params']['nu'] == 0.05
    assert metadata['params']['gamma'] == 'scale'
    assert metadata['params']['max_iter'] == 200
    assert metadata['split']['train_count'] == 4
    assert metadata['split']['validation_count'] == 4
    assert metadata['split']['test_count'] == 4

    expected_keys = set(_metrics().evaluate_split([0, 1], [0, 1], [0.1, 1.1], transient_mask=[0, 1]))
    metrics = json.loads(result.artifact_paths['metrics'].read_text())
    assert expected_keys <= set(metrics)
    assert {f'test_{name}' for name in expected_keys} <= set(metrics)
    assert metrics['training_sample_count'] == 4
    assert metrics['training_normal_count'] == 3
    assert metrics['sample_count'] == 4
    assert metrics['test_sample_count'] == 4

    scores = _pandas().read_csv(result.artifact_paths['scores'])
    assert list(scores.columns) == ['timestamp', 'label', 'changepoint', 'prediction', 'score']
    assert scores['label'].tolist() == [0, 0, 0, 1]
    assert scores['changepoint'].tolist() == [0, 0, 1, 0]

    model = _joblib().load(result.artifact_paths['model'])
    scaler = _joblib().load(result.artifact_paths['scaler'])
    assert model.n_features_in_ == 64
    assert scaler.n_features_in_ == 64

    train_windows = train_oneclass_svm._load_windows(train, config)
    validation_windows = train_oneclass_svm._load_windows(validation, config)
    train_normal_features = train_windows.features[train_windows.labels == 0]
    np.testing.assert_allclose(scaler.mean_, train_normal_features.mean(axis=0))

    validation_x = scaler.transform(validation_windows.features)
    manual_scores = -model.decision_function(validation_x)
    np.testing.assert_allclose(scores['score'].to_numpy(dtype=float), manual_scores)
    validation_normal_scores = manual_scores[validation_windows.labels == 0]
    expected_threshold = float(np.percentile(validation_normal_scores, config.threshold_quantile * 100.0))
    assert result.thresholds['threshold'] == expected_threshold
