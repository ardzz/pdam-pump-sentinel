from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ml.datasets.skab_loader import SENSOR_COLUMNS


@dataclass(frozen=True)
class WindowedSensorDataset:
    features: np.ndarray
    labels: np.ndarray
    changepoints: np.ndarray
    timestamps: np.ndarray
    sensor_columns: tuple[str, ...]
    window_size: int
    stride: int


def build_sensor_windows(
    frame,
    window_size: int,
    stride: int,
    sensor_columns: Sequence[str] = SENSOR_COLUMNS,
) -> WindowedSensorDataset:
    _validate_positive_integer(window_size, 'window_size')
    _validate_positive_integer(stride, 'stride')

    ordered_sensor_columns = tuple(sensor_columns)
    if not ordered_sensor_columns:
        raise ValueError('sensor_columns must not be empty')

    missing_columns = [
        column
        for column in (*ordered_sensor_columns, 'anomaly', 'datetime')
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f'missing required columns: {", ".join(missing_columns)}')

    sensor_values = frame[list(ordered_sensor_columns)].to_numpy(dtype=float, copy=False)
    anomalies = frame['anomaly'].to_numpy()
    changepoints = (
        frame['changepoint'].to_numpy()
        if 'changepoint' in frame.columns
        else np.zeros(len(frame), dtype=int)
    )
    source_timestamps = frame['datetime'].to_numpy()

    features = []
    labels = []
    window_changepoints = []
    timestamps = []
    for start in range(0, len(frame) - window_size + 1, stride):
        stop = start + window_size
        features.append(sensor_values[start:stop].reshape(-1))
        labels.append(int(np.any(anomalies[start:stop] != 0)))
        window_changepoints.append(int(np.any(changepoints[start:stop] != 0)))
        timestamps.append(_normalize_timestamp(source_timestamps[stop - 1]))

    feature_count = window_size * len(ordered_sensor_columns)
    feature_array = (
        np.vstack(features)
        if features
        else np.empty((0, feature_count), dtype=float)
    )

    return WindowedSensorDataset(
        features=feature_array,
        labels=np.asarray(labels, dtype=int),
        changepoints=np.asarray(window_changepoints, dtype=int),
        timestamps=np.asarray(timestamps, dtype=object),
        sensor_columns=ordered_sensor_columns,
        window_size=window_size,
        stride=stride,
    )


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def _normalize_timestamp(value) -> str:
    isoformat = getattr(value, 'isoformat', None)
    text = str(isoformat()) if callable(isoformat) else str(value)
    return text.replace('+00:00', 'Z')
