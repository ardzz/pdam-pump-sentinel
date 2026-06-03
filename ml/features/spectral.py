from collections.abc import Sequence

import numpy as np

from ml.datasets.skab_loader import SENSOR_COLUMNS
from ml.features.windowing import _normalize_timestamp, _validate_positive_integer


def build_spectral_window_features(
    frame,
    window_size: int,
    stride: int,
    sensor_columns: Sequence[str] = SENSOR_COLUMNS,
    n_bands: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _validate_positive_integer(window_size, 'window_size')
    _validate_positive_integer(stride, 'stride')
    _validate_positive_integer(n_bands, 'n_bands')

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
        window = sensor_values[start:stop]
        features.append(_window_features(window, n_bands))
        labels.append(int(np.any(anomalies[start:stop] != 0)))
        window_changepoints.append(int(np.any(changepoints[start:stop] != 0)))
        timestamps.append(_normalize_timestamp(source_timestamps[stop - 1]))

    feature_count = len(ordered_sensor_columns) * (n_bands + 4)
    feature_array = (
        np.vstack(features)
        if features
        else np.empty((0, feature_count), dtype=float)
    )
    return (
        feature_array,
        np.asarray(labels, dtype=int),
        np.asarray(window_changepoints, dtype=int),
        np.asarray(timestamps, dtype=object),
    )


def _window_features(window: np.ndarray, n_bands: int) -> np.ndarray:
    means = np.mean(window, axis=0)
    stds = np.std(window, axis=0)
    ranges = np.ptp(window, axis=0)

    band_energies = []
    centroids = []
    frequency_bins = np.fft.rfftfreq(window.shape[0])
    for sensor_index in range(window.shape[1]):
        centered = window[:, sensor_index] - means[sensor_index]
        energies = np.abs(np.fft.rfft(centered)) ** 2
        band_energies.extend(float(band.sum()) for band in np.array_split(energies, n_bands))
        energy_sum = float(energies.sum())
        centroids.append(float(np.dot(frequency_bins, energies) / energy_sum) if energy_sum else 0.0)

    return np.asarray([*means, *stds, *ranges, *band_energies, *centroids], dtype=float)


__all__ = ['build_spectral_window_features']
