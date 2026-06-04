from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from ml.features.enriched import build_enriched_window_features

_SCALER_FILENAME = 'scaler.joblib'
_METADATA_FILENAME = 'metadata.json'
_MODEL_FAMILIES = {'lightgbm', 'xgboost'}
_SPECTRAL_N_BANDS = 4
_DOMAIN_FEATURE_SENSORS = (
    ('Temperature', 'Thermocouple'),
    ('Temperature', 'Thermocouple'),
    ('Pressure', 'Volume Flow RateRMS'),
    ('Pressure', 'Volume Flow RateRMS'),
    ('Voltage', 'Current'),
    ('Voltage', 'Current'),
)


@dataclass(frozen=True)
class SupervisedAnomalyVerdict:
    station: str
    timestamp: str | None
    window_filled: bool
    window_size: int
    model_version: str
    t2_threshold: float
    q_threshold: float
    t2: float | None = None
    q: float | None = None
    score: float | None = None
    anomaly: int | None = None
    top_contributing_sensor: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SupervisedAnomalyInferenceService:
    def __init__(
        self,
        model: Any,
        scaler: Any,
        sensor_columns: Sequence[str],
        window_size: int,
        threshold: float,
        model_family: str,
        model_version: str | None = None,
        spectral_n_bands: int = _SPECTRAL_N_BANDS,
    ) -> None:
        ordered = tuple(sensor_columns)
        family = model_family.lower()
        if not ordered:
            raise ValueError('sensor_columns must not be empty')
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
            raise ValueError('window_size must be a positive integer')
        if family not in _MODEL_FAMILIES:
            raise ValueError(f'model_family must be one of: {", ".join(sorted(_MODEL_FAMILIES))}')
        if not callable(getattr(model, 'predict_proba', None)):
            raise TypeError('model must implement predict_proba()')
        if scaler is not None and not callable(getattr(scaler, 'transform', None)):
            raise TypeError('scaler must implement transform()')
        if isinstance(spectral_n_bands, bool) or not isinstance(spectral_n_bands, int) or spectral_n_bands <= 0:
            raise ValueError('spectral_n_bands must be a positive integer')

        expected_features = _feature_count(len(ordered), spectral_n_bands)
        n_features = int(getattr(model, 'n_features_in_', expected_features))
        if n_features != expected_features:
            raise ValueError(f'model expects {n_features} features but enriched mode produces {expected_features}')
        scaler_features = int(getattr(scaler, 'n_features_in_', expected_features)) if scaler is not None else expected_features
        if scaler_features != expected_features:
            raise ValueError(f'scaler expects {scaler_features} features but enriched mode produces {expected_features}')

        self._model = model
        self._scaler = scaler
        self._sensor_columns = ordered
        self._window_size = window_size
        self._threshold = float(threshold)
        self._model_family = family
        self._model_version = model_version or f'{family}-local'
        self._spectral_n_bands = spectral_n_bands
        self._buffers: dict[str, deque[tuple[str | None, list[float]]]] = {}
        self._top_sensor = _top_sensor_from_model_importance(model, ordered, spectral_n_bands)

    @property
    def sensor_columns(self) -> tuple[str, ...]:
        return self._sensor_columns

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def model_family(self) -> str:
        return self._model_family

    @classmethod
    def from_artifacts(cls, model_dir: str | Path, model_version: str | None = None) -> SupervisedAnomalyInferenceService:
        directory = Path(model_dir)
        metadata = cls._read_metadata(directory / _METADATA_FILENAME)
        model_path = _model_path(directory, metadata['model_family'])
        scaler_path = directory / _SCALER_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(f'{metadata["model_family"]} model artifact not found: {model_path}')

        joblib = import_module('joblib')
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        return cls(
            model,
            scaler,
            metadata['sensor_columns'],
            metadata['window_size'],
            metadata['threshold'],
            metadata['model_family'],
            model_version or metadata['model_version'],
            metadata['spectral_n_bands'],
        )

    @staticmethod
    def _read_metadata(metadata_path: Path) -> dict[str, Any]:
        if not metadata_path.exists():
            raise FileNotFoundError(f'supervised metadata artifact not found: {metadata_path}')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        params = metadata.get('params') or {}
        model_family = str(metadata.get('model_family') or params.get('model_type') or '').lower()
        if model_family not in _MODEL_FAMILIES:
            raise ValueError(f'unsupported supervised model_family: {model_family}')
        sensor_columns = [str(column) for column in metadata.get('sensor_columns') or params.get('sensor_columns') or []]
        if not sensor_columns:
            skab_loader = import_module('ml.datasets.skab_loader')
            sensor_columns = list(skab_loader.SENSOR_COLUMNS)
        raw_window_size = params.get('window_size')
        if raw_window_size is None:
            raise ValueError('supervised metadata must include window_size')
        thresholds = metadata.get('thresholds') or {}
        threshold_value = thresholds.get('threshold', thresholds.get('q_threshold', params.get('threshold')))
        if threshold_value is None:
            raise ValueError('supervised metadata must include a threshold')
        return {
            'model_family': model_family,
            'sensor_columns': sensor_columns,
            'window_size': int(raw_window_size),
            'threshold': float(threshold_value),
            'model_version': str(params.get('model_version') or f'{model_family}-local'),
            'spectral_n_bands': int(params.get('spectral_n_bands') or _SPECTRAL_N_BANDS),
        }

    def reset(self, station: str | None = None) -> None:
        if station is None:
            self._buffers.clear()
        else:
            self._buffers.pop(station, None)

    def observe(self, station: str, timestamp: str | None, sensors: Mapping[str, Any]) -> SupervisedAnomalyVerdict:
        if not isinstance(sensors, Mapping):
            raise TypeError('sensors must be a mapping of sensor name to value')

        row = self._extract_row(sensors)
        buffer = self._buffers.setdefault(station, deque(maxlen=self._window_size))
        buffer.append((timestamp, row))

        if len(buffer) < self._window_size:
            return self._warmup_verdict(station, timestamp)

        feature = self._build_feature(buffer)
        model_feature = self._scaler.transform(feature) if self._scaler is not None else feature
        score = float(_positive_probabilities(self._model, model_feature)[0])
        anomaly = int(score >= self._threshold)

        return SupervisedAnomalyVerdict(
            station=station,
            timestamp=timestamp,
            window_filled=True,
            window_size=self._window_size,
            model_version=self._model_version,
            t2_threshold=self._threshold,
            q_threshold=self._threshold,
            t2=None,
            q=None,
            score=score,
            anomaly=anomaly,
            top_contributing_sensor=self._top_sensor,
        )

    def _extract_row(self, sensors: Mapping[str, Any]) -> list[float]:
        row: list[float] = []
        for column in self._sensor_columns:
            if column not in sensors:
                raise KeyError(f'missing sensor value: {column}')
            row.append(float(sensors[column]))
        return row

    def _warmup_verdict(self, station: str, timestamp: str | None) -> SupervisedAnomalyVerdict:
        return SupervisedAnomalyVerdict(
            station=station,
            timestamp=timestamp,
            window_filled=False,
            window_size=self._window_size,
            model_version=self._model_version,
            t2_threshold=self._threshold,
            q_threshold=self._threshold,
        )

    def _build_feature(self, buffer: deque[tuple[str | None, list[float]]]) -> np.ndarray:
        pandas = import_module('pandas')
        rows = [row for _timestamp, row in buffer]
        timestamps = [timestamp if timestamp is not None else str(index) for index, (timestamp, _row) in enumerate(buffer)]
        frame = pandas.DataFrame(rows, columns=list(self._sensor_columns))
        frame['datetime'] = timestamps
        frame['anomaly'] = 0
        frame['changepoint'] = 0
        features, _labels, _changepoints, _timestamps = build_enriched_window_features(
            frame,
            window_size=self._window_size,
            stride=self._window_size,
            sensor_columns=self._sensor_columns,
            n_bands=self._spectral_n_bands,
        )
        return np.asarray(features, dtype=np.float64)


def _model_path(directory: Path, model_family: str) -> Path:
    preferred = directory / f'{model_family}.joblib'
    if preferred.exists():
        return preferred
    return directory / 'supervised_model.joblib'


def _feature_count(sensor_count: int, n_bands: int) -> int:
    return sensor_count * (n_bands + 7) + len(_DOMAIN_FEATURE_SENSORS)


def _positive_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = [int(value) for value in np.asarray(getattr(model, 'classes_', [0, 1])).reshape(-1)]
    if probabilities.ndim == 1:
        return probabilities.reshape(-1)
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    return np.zeros((probabilities.shape[0],), dtype=float)


def _top_sensor_from_model_importance(model: Any, sensor_columns: tuple[str, ...], n_bands: int) -> str | None:
    raw_importances = getattr(model, 'feature_importances_', None)
    if raw_importances is None:
        return None
    importances = np.asarray(raw_importances, dtype=np.float64).reshape(-1)
    expected_features = _feature_count(len(sensor_columns), n_bands)
    if importances.size != expected_features or not np.isfinite(importances).all():
        return None
    sensor_importances = _aggregate_sensor_importances(importances, sensor_columns, n_bands)
    if float(sensor_importances.sum()) <= 0.0:
        return None
    return sensor_columns[int(np.argmax(sensor_importances))]


def _aggregate_sensor_importances(importances: np.ndarray, sensor_columns: tuple[str, ...], n_bands: int) -> np.ndarray:
    sensor_count = len(sensor_columns)
    per_sensor = np.zeros(sensor_count, dtype=np.float64)
    per_sensor += importances[0:sensor_count]
    per_sensor += importances[sensor_count : 2 * sensor_count]
    per_sensor += importances[2 * sensor_count : 3 * sensor_count]
    band_start = 3 * sensor_count
    band_stop = band_start + sensor_count * n_bands
    per_sensor += importances[band_start:band_stop].reshape(sensor_count, n_bands).sum(axis=1)
    centroid_start = band_stop
    centroid_stop = centroid_start + sensor_count
    per_sensor += importances[centroid_start:centroid_stop]
    extra_start = centroid_stop
    per_sensor += importances[extra_start : extra_start + sensor_count]
    per_sensor += importances[extra_start + sensor_count : extra_start + 2 * sensor_count]
    per_sensor += importances[extra_start + 2 * sensor_count : extra_start + 3 * sensor_count]
    domain_start = extra_start + 3 * sensor_count
    sensor_index = {name: index for index, name in enumerate(sensor_columns)}
    for offset, names in enumerate(_DOMAIN_FEATURE_SENSORS):
        value = importances[domain_start + offset] / len(names)
        for name in names:
            index = sensor_index.get(name)
            if index is not None:
                per_sensor[index] += value
    return per_sensor


__all__ = ['SupervisedAnomalyInferenceService', 'SupervisedAnomalyVerdict']
