from importlib import import_module

import numpy as np
import pytest


def _skab_loader():
    return import_module('ml.datasets.skab_loader')


def _windowing():
    return import_module('ml.features.windowing')


def _sensor_frame(tmp_path, row_count=5, anomalies=None, changepoints=None):
    loader = _skab_loader()
    rows = [['datetime', *loader.SENSOR_COLUMNS, 'anomaly', 'changepoint']]
    for row_index in range(row_count):
        sensor_values = [
            str(float(row_index * 100 + column_index))
            for column_index in range(len(loader.SENSOR_COLUMNS))
        ]
        rows.append([
            f'2024-01-01T00:00:0{row_index}Z',
            *sensor_values,
            str(int(anomalies[row_index]) if anomalies else 0),
            str(int(changepoints[row_index]) if changepoints else 0),
        ])

    fixture = tmp_path / 'windowing.csv'
    fixture.write_text('\n'.join(';'.join(row) for row in rows) + '\n')
    return loader.load_skab_csv(fixture)


def test_build_sensor_windows_flattens_windows_in_default_sensor_order(tmp_path):
    loader = _skab_loader()
    windowing = _windowing()
    frame = _sensor_frame(tmp_path, row_count=4)

    dataset = windowing.build_sensor_windows(frame, window_size=2, stride=2)

    sensor_count = len(loader.SENSOR_COLUMNS)
    assert dataset.features.shape == (2, 2 * sensor_count)
    assert dataset.sensor_columns == tuple(loader.SENSOR_COLUMNS)
    np.testing.assert_array_equal(
        dataset.features[0],
        np.asarray([*range(sensor_count), *range(100, 100 + sensor_count)], dtype=float),
    )
    np.testing.assert_array_equal(
        dataset.features[1],
        np.asarray([*range(200, 200 + sensor_count), *range(300, 300 + sensor_count)], dtype=float),
    )
    assert dataset.window_size == 2
    assert dataset.stride == 2


def test_build_sensor_windows_labels_window_when_any_row_is_anomalous(tmp_path):
    windowing = _windowing()
    frame = _sensor_frame(tmp_path, row_count=4, anomalies=[0, 0, 2, 0])

    dataset = windowing.build_sensor_windows(frame, window_size=2, stride=2)

    assert dataset.labels.tolist() == [0, 1]


def test_build_sensor_windows_tracks_changepoints_without_changing_anomaly_labels(tmp_path):
    windowing = _windowing()
    frame = _sensor_frame(
        tmp_path,
        row_count=4,
        anomalies=[0, 0, 0, 0],
        changepoints=[0, 2, 0, 0],
    )

    dataset = windowing.build_sensor_windows(frame, window_size=2, stride=2)

    assert dataset.labels.tolist() == [0, 0]
    assert dataset.changepoints.tolist() == [1, 0]


def test_build_sensor_windows_uses_final_timestamp_and_drops_incomplete_trailing_window(
    tmp_path,
):
    windowing = _windowing()
    frame = _sensor_frame(tmp_path, row_count=5)

    dataset = windowing.build_sensor_windows(frame, window_size=2, stride=2)

    assert dataset.timestamps.tolist() == [
        '2024-01-01T00:00:01Z',
        '2024-01-01T00:00:03Z',
    ]
    assert len(dataset.features) == 2


@pytest.mark.parametrize(
    ('window_size', 'stride'),
    [(0, 1), (-1, 1), (1, 0), (1, -1), (1.5, 1), (True, 1)],
)
def test_build_sensor_windows_rejects_invalid_window_size_or_stride(
    tmp_path,
    window_size,
    stride,
):
    windowing = _windowing()
    frame = _sensor_frame(tmp_path, row_count=3)

    with pytest.raises(ValueError, match='positive integer'):
        windowing.build_sensor_windows(frame, window_size=window_size, stride=stride)


def test_build_sensor_windows_rejects_missing_sensor_columns(tmp_path):
    loader = _skab_loader()
    windowing = _windowing()
    missing_sensor = loader.SENSOR_COLUMNS[0]
    frame = _sensor_frame(tmp_path, row_count=3).drop(columns=[missing_sensor])

    with pytest.raises(ValueError, match=missing_sensor):
        windowing.build_sensor_windows(frame, window_size=2, stride=1)
