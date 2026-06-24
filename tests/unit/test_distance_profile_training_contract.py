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


def _train_distance_profile():
    return import_module('ml.training.train_distance_profile')


def _write_skab_csv(path, row_count=12, anomaly_start=None, offset=0.0, spike=30.0, changepoint_indices=None):
    sensor_columns = _skab_loader().SENSOR_COLUMNS
    changepoint_indices = set(changepoint_indices or [])
    rows = [['datetime', *sensor_columns, 'anomaly', 'changepoint']]
    for row_index in range(row_count):
        anomaly = int(anomaly_start is not None and row_index >= anomaly_start)
        changepoint = int(row_index in changepoint_indices)
        sensor_values = []
        for column_index, _column in enumerate(sensor_columns):
            baseline = offset + row_index * 0.25 + column_index * 0.1
            sensor_values.append(f'{baseline + anomaly * spike:.6f}')
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


def test_reference_selection_is_bounded_and_deterministic():
    train_distance_profile = _train_distance_profile()
    features = np.arange(30, dtype=float).reshape(10, 3)

    selected = train_distance_profile._select_reference_windows(features, max_reference_windows=4)

    np.testing.assert_allclose(selected, features[[0, 3, 6, 9]])


def test_nearest_reference_distances_use_chunked_runtime_guard_and_higher_scores_for_far_windows():
    train_distance_profile = _train_distance_profile()
    references = np.asarray([[0.0, 0.0], [3.0, 4.0]], dtype=float)
    features = np.asarray([[0.0, 0.0], [6.0, 8.0]], dtype=float)

    scores = train_distance_profile._nearest_reference_distances(
        features,
        references,
        max_distance_comparisons=2,
    )

    np.testing.assert_allclose(scores, np.asarray([0.0, 5.0 / np.sqrt(2.0)], dtype=float))
    assert scores[1] > scores[0]


def test_train_distance_profile_writes_artifacts_and_uses_validation_normal_threshold(tmp_path):
    train_distance_profile = _train_distance_profile()
    train = _write_skab_csv(tmp_path / 'train.csv', anomaly_start=None, offset=0.0)
    validation = _write_skab_csv(
        tmp_path / 'validation.csv', anomaly_start=9, offset=20.0, spike=60.0, changepoint_indices=[9]
    )
    test = _write_skab_csv(tmp_path / 'test.csv', anomaly_start=9, offset=30.0, spike=50.0, changepoint_indices=[9])
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train], [validation], [test])
    output_dir = tmp_path / 'artifacts'
    config = train_distance_profile.DistanceProfileTrainingConfig(
        input_path=tmp_path / 'unused.csv',
        output_dir=output_dir,
        split_manifest_path=manifest,
        window_size=3,
        stride=3,
        threshold_quantile=0.9,
        scaler='standard',
        max_reference_windows=2,
        max_distance_comparisons=16,
    )

    result = train_distance_profile.train_distance_profile_from_skab(config)

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
    assert metadata['model_family'] == 'distance_profile_nearest_normal'
    assert metadata['params']['window_size'] == 3
    assert metadata['params']['stride'] == 3
    assert metadata['params']['feature_mode'] == 'raw_window_nearest_normal_distance'
    assert metadata['params']['reference_window_count'] == 2
    assert metadata['params']['max_reference_windows'] == 2
    assert metadata['params']['max_distance_comparisons_per_batch'] == 16
    assert metadata['params']['sensor_columns'] == _skab_loader().SENSOR_COLUMNS
    assert metadata['score_orientation'] == 'higher_is_more_anomalous:nearest_train_normal_window_distance'
    assert metadata['dependency_status'] == 'implemented_with_existing_numpy_sklearn_only_no_stumpy_dependency'
    assert metadata['runtime_guard']['max_reference_windows'] == 2
    assert metadata['runtime_guard']['reference_window_count'] == 2
    assert metadata['runtime_guard']['max_distance_comparisons_per_batch'] == 16
    assert metadata['test_labels_used_for_training_or_threshold'] is False
    assert metadata['test_scores_used_for_training_or_threshold'] is False
    assert 'validation-normal nearest-distance scores only' in metadata['threshold_calibration']
    assert metadata['split']['train_count'] == 4
    assert metadata['split']['validation_count'] == 4
    assert metadata['split']['test_count'] == 4

    expected_keys = set(_metrics().evaluate_split([0, 1], [0, 1], [0.1, 1.1], transient_mask=[0, 1]))
    metrics = json.loads(result.artifact_paths['metrics'].read_text())
    assert expected_keys <= set(metrics)
    assert {f'test_{name}' for name in expected_keys} <= set(metrics)
    assert metrics['training_sample_count'] == 4
    assert metrics['training_normal_count'] == 4
    assert metrics['reference_window_count'] == 2
    assert metrics['sample_count'] == 4
    assert metrics['test_sample_count'] == 4

    scores = _pandas().read_csv(result.artifact_paths['scores'])
    assert list(scores.columns) == ['timestamp', 'label', 'changepoint', 'prediction', 'score']
    assert scores['label'].tolist() == [0, 0, 0, 1]
    assert scores['changepoint'].tolist() == [0, 0, 0, 1]

    detector = _joblib().load(result.artifact_paths['model'])
    scaler = _joblib().load(result.artifact_paths['scaler'])
    assert len(detector.reference_windows_) == 2
    train_windows = train_distance_profile._load_windows(train, config)
    validation_windows = train_distance_profile._load_windows(validation, config)
    train_normal_features = train_windows.features[train_windows.labels == 0]
    np.testing.assert_allclose(scaler.mean_, train_normal_features.mean(axis=0))

    validation_x = scaler.transform(validation_windows.features)
    manual_scores = detector.score_samples(validation_x)
    np.testing.assert_allclose(scores['score'].to_numpy(dtype=float), manual_scores)
    validation_normal_scores = manual_scores[validation_windows.labels == 0]
    expected_threshold = float(np.percentile(validation_normal_scores, config.threshold_quantile * 100.0))
    assert result.thresholds['threshold'] == expected_threshold
    assert result.thresholds['threshold'] < manual_scores[validation_windows.labels == 1][0]
