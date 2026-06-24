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


def _train_forecasting_residual():
    return import_module('ml.training.train_forecasting_residual')


def _write_skab_csv(path, row_count=9, anomaly_start=None, offset=0.0, spike=50.0, future_spike_index=None):
    sensor_columns = _skab_loader().SENSOR_COLUMNS
    rows = [['datetime', *sensor_columns, 'anomaly', 'changepoint']]
    for row_index in range(row_count):
        anomaly = int(anomaly_start is not None and row_index >= anomaly_start)
        sensor_values = []
        for column_index, _column in enumerate(sensor_columns):
            baseline = offset + row_index + column_index * 0.1
            future_spike = spike if future_spike_index is not None and row_index == future_spike_index else 0.0
            sensor_values.append(f'{baseline + anomaly * spike + future_spike:.6f}')
        rows.append([
            f'2024-01-01T00:00:{row_index:02d}Z',
            *sensor_values,
            str(anomaly),
            str(int(anomaly and row_index == anomaly_start)),
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


def test_lag_one_residuals_align_to_window_final_target_without_future_rows():
    train_forecasting_residual = _train_forecasting_residual()
    features = np.asarray(
        [
            [1.0, 2.0, 4.0, 8.0, 10.0, 9.0],
            [3.0, 1.0, 5.0, 6.0, 6.0, 16.0],
        ],
        dtype=float,
    )

    residuals = train_forecasting_residual._lag_one_abs_residuals(features, window_size=3, sensor_count=2)

    np.testing.assert_allclose(residuals, np.asarray([[6.0, 1.0], [1.0, 10.0]], dtype=float))


def test_validation_scoring_does_not_look_at_future_rows(tmp_path):
    train_forecasting_residual = _train_forecasting_residual()
    baseline = _write_skab_csv(tmp_path / 'baseline.csv', row_count=4, spike=100.0)
    future_spike = _write_skab_csv(tmp_path / 'future_spike.csv', row_count=4, spike=100.0, future_spike_index=3)
    config = train_forecasting_residual.ForecastingResidualTrainingConfig(
        input_path=baseline,
        output_dir=tmp_path / 'artifacts',
        window_size=3,
        stride=1,
    )
    baseline_windows = train_forecasting_residual._load_windows(baseline, config)
    spiked_windows = train_forecasting_residual._load_windows(future_spike, config)
    detector = train_forecasting_residual.LagOneResidualDetector(
        sensor_columns=tuple(_skab_loader().SENSOR_COLUMNS),
        window_size=3,
    )
    detector.residual_scale_ = np.ones(len(_skab_loader().SENSOR_COLUMNS), dtype=float)

    baseline_scores = detector.score_samples(baseline_windows.features)
    spiked_scores = detector.score_samples(spiked_windows.features)

    assert baseline_windows.timestamps.tolist() == ['2024-01-01T00:00:02Z', '2024-01-01T00:00:03Z']
    np.testing.assert_allclose(spiked_scores[0], baseline_scores[0])
    assert spiked_scores[1] > baseline_scores[1]


def test_train_forecasting_residual_writes_artifacts_and_uses_validation_normal_threshold(tmp_path):
    train_forecasting_residual = _train_forecasting_residual()
    train = _write_skab_csv(tmp_path / 'train.csv', anomaly_start=None, offset=0.0)
    validation = _write_skab_csv(tmp_path / 'validation.csv', anomaly_start=8, offset=100.0, spike=50.0)
    test = _write_skab_csv(tmp_path / 'test.csv', anomaly_start=8, offset=200.0, spike=30.0)
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train], [validation], [test])
    output_dir = tmp_path / 'artifacts'
    config = train_forecasting_residual.ForecastingResidualTrainingConfig(
        input_path=tmp_path / 'unused.csv',
        output_dir=output_dir,
        split_manifest_path=manifest,
        window_size=3,
        stride=3,
        threshold_quantile=0.9,
        aggregation='mean',
    )

    result = train_forecasting_residual.train_forecasting_residual_from_skab(config)

    assert set(result.artifact_paths) == {
        'model',
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
    assert metadata['model_family'] == 'forecasting_residual_naive'
    assert metadata['params']['window_size'] == 3
    assert metadata['params']['stride'] == 3
    assert metadata['params']['feature_mode'] == 'raw_lag_one_residual'
    assert metadata['params']['aggregation'] == 'mean'
    assert metadata['params']['sensor_columns'] == _skab_loader().SENSOR_COLUMNS
    assert metadata['score_orientation'] == 'higher_is_more_anomalous:mean_abs_scaled_lag_one_residual'
    assert metadata['test_labels_used_for_training_or_threshold'] is False
    assert metadata['test_scores_used_for_training_or_threshold'] is False
    assert 'validation-normal lag-one residual scores only' in metadata['threshold_calibration']
    assert 'no future rows are read' in metadata['residual_alignment']
    assert metadata['split']['train_count'] == 3
    assert metadata['split']['validation_count'] == 3
    assert metadata['split']['test_count'] == 3

    expected_keys = set(_metrics().evaluate_split([0, 1], [0, 1], [0.1, 1.1], transient_mask=[0, 1]))
    metrics = json.loads(result.artifact_paths['metrics'].read_text())
    assert expected_keys <= set(metrics)
    assert {f'test_{name}' for name in expected_keys} <= set(metrics)
    assert metrics['training_sample_count'] == 3
    assert metrics['training_normal_count'] == 3
    assert metrics['sample_count'] == 3
    assert metrics['test_sample_count'] == 3

    scores = _pandas().read_csv(result.artifact_paths['scores'])
    assert list(scores.columns) == ['timestamp', 'label', 'changepoint', 'prediction', 'score']
    assert scores['label'].tolist() == [0, 0, 1]
    assert scores['changepoint'].tolist() == [0, 0, 1]

    detector = _joblib().load(result.artifact_paths['model'])
    np.testing.assert_allclose(detector.residual_scale_, np.ones(len(_skab_loader().SENSOR_COLUMNS)))
    validation_windows = train_forecasting_residual._load_windows(validation, config)
    manual_scores = detector.score_samples(validation_windows.features)
    np.testing.assert_allclose(scores['score'].to_numpy(dtype=float), manual_scores)
    validation_normal_scores = manual_scores[validation_windows.labels == 0]
    expected_threshold = float(np.percentile(validation_normal_scores, config.threshold_quantile * 100.0))
    assert result.thresholds['threshold'] == expected_threshold
    assert result.thresholds['threshold'] < manual_scores[validation_windows.labels == 1][0]
