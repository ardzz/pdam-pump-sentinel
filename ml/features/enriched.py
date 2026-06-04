from collections.abc import Sequence

import numpy as np

from ml.datasets.skab_loader import SENSOR_COLUMNS
from ml.features.spectral import _window_features as _spectral_window_features
from ml.features.windowing import _normalize_timestamp, _validate_positive_integer

_DOMAIN_FEATURE_COLUMNS = (
    'Temperature',
    'Thermocouple',
    'Pressure',
    'Volume Flow RateRMS',
    'Voltage',
    'Current',
)
_DOMAIN_FEATURE_COUNT = 6
_EPS = 1e-12


def build_enriched_window_features(
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
        column for column in (*ordered_sensor_columns, 'anomaly', 'datetime') if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f'missing required columns: {", ".join(missing_columns)}')

    missing_domain_columns = [column for column in _DOMAIN_FEATURE_COLUMNS if column not in ordered_sensor_columns]
    if missing_domain_columns:
        raise ValueError(f'sensor_columns must include domain columns: {", ".join(missing_domain_columns)}')

    sensor_index = {column: index for index, column in enumerate(ordered_sensor_columns)}
    sensor_values = frame[list(ordered_sensor_columns)].to_numpy(dtype=float, copy=False)
    anomalies = frame['anomaly'].to_numpy()
    changepoints = (
        frame['changepoint'].to_numpy() if 'changepoint' in frame.columns else np.zeros(len(frame), dtype=int)
    )
    source_timestamps = frame['datetime'].to_numpy()

    features = []
    labels = []
    window_changepoints = []
    timestamps = []
    for start in range(0, len(frame) - window_size + 1, stride):
        stop = start + window_size
        window = sensor_values[start:stop]
        features.append(_window_features(window, n_bands, sensor_index))
        labels.append(int(np.any(anomalies[start:stop] != 0)))
        window_changepoints.append(int(np.any(changepoints[start:stop] != 0)))
        timestamps.append(_normalize_timestamp(source_timestamps[stop - 1]))

    feature_count = len(ordered_sensor_columns) * (n_bands + 7) + _DOMAIN_FEATURE_COUNT
    feature_array = np.vstack(features) if features else np.empty((0, feature_count), dtype=float)
    return (
        feature_array,
        np.asarray(labels, dtype=int),
        np.asarray(window_changepoints, dtype=int),
        np.asarray(timestamps, dtype=object),
    )


def _window_features(window: np.ndarray, n_bands: int, sensor_index: dict[str, int]) -> np.ndarray:
    spectral_features = _spectral_window_features(window, n_bands)
    mins = np.min(window, axis=0)
    maxs = np.max(window, axis=0)
    rate_of_change = window[-1] - window[0]
    domain_features = _domain_features(np.mean(window, axis=0), sensor_index)
    return np.asarray([*spectral_features, *mins, *maxs, *rate_of_change, *domain_features], dtype=float)


def _domain_features(window_means: np.ndarray, sensor_index: dict[str, int]) -> np.ndarray:
    temperature = window_means[sensor_index['Temperature']]
    thermocouple = window_means[sensor_index['Thermocouple']]
    pressure = window_means[sensor_index['Pressure']]
    flow = window_means[sensor_index['Volume Flow RateRMS']]
    voltage = window_means[sensor_index['Voltage']]
    current = window_means[sensor_index['Current']]
    return np.asarray(
        [
            temperature - thermocouple,
            temperature / (thermocouple + _EPS),
            pressure * flow,
            pressure / (flow + _EPS),
            voltage * current,
            voltage / (current + _EPS),
        ],
        dtype=float,
    )


__all__ = ['build_enriched_window_features']
