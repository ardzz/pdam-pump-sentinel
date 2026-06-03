import json
from importlib import import_module

import numpy as np


def _metrics():
    return import_module('ml.evaluation.metrics')


def _pandas():
    return import_module('pandas')


def _skab_loader():
    return import_module('ml.datasets.skab_loader')


def _spectral():
    return import_module('ml.features.spectral')


def _train_pca():
    return import_module('ml.training.train_pca')


def _write_skab_csv(path, row_count=16, anomaly_start=None, offset=0.0, changepoint_indices=None):
    sensor_columns = _skab_loader().SENSOR_COLUMNS
    changepoint_indices = set(changepoint_indices or [])
    rows = [['datetime', *sensor_columns, 'anomaly', 'changepoint']]
    for row_index in range(row_count):
        anomaly = int(anomaly_start is not None and row_index >= anomaly_start)
        changepoint = int(row_index in changepoint_indices)
        sensor_values = []
        for column_index, _ in enumerate(sensor_columns):
            baseline = 10.0 + offset + row_index * 0.1 + column_index * 0.5
            oscillation = ((row_index % 4) - 1.5) * (column_index + 1) * 0.02
            sensor_values.append(f'{baseline + oscillation + anomaly * 5.0:.6f}')
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


def test_build_spectral_window_features_shape_and_window_semantics():
    pd = _pandas()
    spectral = _spectral()
    frame = pd.DataFrame(
        {
            'datetime': pd.to_datetime(
                [
                    '2024-01-01T00:00:00Z',
                    '2024-01-01T00:00:01Z',
                    '2024-01-01T00:00:02Z',
                    '2024-01-01T00:00:03Z',
                    '2024-01-01T00:00:04Z',
                ],
                utc=True,
            ),
            's1': [0.0, 1.0, 2.0, 3.0, 4.0],
            's2': [10.0, 10.0, 10.0, 10.0, 10.0],
            'anomaly': [0, 2, 0, 0, 0],
            'changepoint': [0, 1, 0, 0, 0],
        }
    )

    features, labels, changepoints, timestamps = spectral.build_spectral_window_features(
        frame,
        window_size=3,
        stride=2,
        sensor_columns=('s1', 's2'),
        n_bands=2,
    )

    assert features.shape == (2, 12)
    assert labels.tolist() == [1, 0]
    assert changepoints.tolist() == [1, 0]
    assert timestamps.tolist() == ['2024-01-01T00:00:02Z', '2024-01-01T00:00:04Z']
    np.testing.assert_allclose(features[0, :6], [1.0, 10.0, np.sqrt(2.0 / 3.0), 0.0, 2.0, 0.0])
    assert np.all(features[:, 6:10] >= 0.0)
    assert features[0, -1] == 0.0


def test_spectral_training_fits_and_calibrates_without_split_leakage(tmp_path, monkeypatch):
    train_pca = _train_pca()
    train1 = _write_skab_csv(tmp_path / 'train1.csv', row_count=12, anomaly_start=8, offset=0.0)
    train2 = _write_skab_csv(tmp_path / 'train2.csv', row_count=12, anomaly_start=8, offset=50.0)
    validation = _write_skab_csv(tmp_path / 'val.csv', row_count=12, anomaly_start=8, offset=100.0)
    test = _write_skab_csv(tmp_path / 'test.csv', row_count=12, anomaly_start=8, offset=150.0)
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train1, train2], [validation], [test])
    fit_calls = []
    calibrate_calls = []
    original_fit = train_pca.PcaT2QDetector.fit
    original_calibrate = train_pca.PcaT2QDetector.calibrate_thresholds

    def patched_fit(self, X, y=None):
        fit_calls.append(np.asarray(X).copy())
        return original_fit(self, X, y)

    def patched_calibrate(self, X):
        calibrate_calls.append(np.asarray(X).copy())
        return original_calibrate(self, X)

    monkeypatch.setattr(train_pca.PcaT2QDetector, 'fit', patched_fit)
    monkeypatch.setattr(train_pca.PcaT2QDetector, 'calibrate_thresholds', patched_calibrate)

    config = train_pca.PcaTrainingConfig(
        input_path=tmp_path / 'dummy.csv',
        output_dir=tmp_path / 'out',
        split_manifest_path=manifest,
        window_size=4,
        stride=4,
        threshold_quantile=0.9,
        feature_mode='spectral',
    )

    result = train_pca.train_pca_from_skab(config)

    assert len(fit_calls) == 1
    assert fit_calls[0].shape == (4, 64)
    assert len(calibrate_calls) == 1
    assert calibrate_calls[0].shape == (2, 64)
    assert result.params['feature_mode'] == 'spectral'
    assert result.params['spectral_n_bands'] == 4


def test_spectral_training_path_returns_evaluate_split_metric_keys(tmp_path):
    train_pca = _train_pca()
    train = _write_skab_csv(tmp_path / 'train.csv', row_count=16, offset=0.0)
    validation = _write_skab_csv(tmp_path / 'val.csv', row_count=16, anomaly_start=12, offset=100.0, changepoint_indices=[11])
    test = _write_skab_csv(tmp_path / 'test.csv', row_count=16, anomaly_start=12, offset=200.0, changepoint_indices=[12])
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train], [validation], [test])

    result = train_pca.train_pca_from_skab(
        train_pca.PcaTrainingConfig(
            input_path=tmp_path / 'dummy.csv',
            output_dir=tmp_path / 'out',
            split_manifest_path=manifest,
            window_size=4,
            stride=4,
            threshold_quantile=0.9,
            feature_mode='spectral',
        )
    )

    expected_keys = set(
        _metrics().evaluate_split(
            [0, 1],
            [0, 1],
            [0.1, 1.1],
            transient_mask=[0, 1],
        )
    )
    assert expected_keys <= set(result.metrics)
    assert {f'test_{name}' for name in expected_keys} <= set(result.metrics)
    assert result.metrics['training_normal_count'] == 4
    assert result.metrics['test_count'] == 4
