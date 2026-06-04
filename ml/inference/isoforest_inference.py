from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from ml.features.spectral import build_spectral_window_features

_MODEL_FILENAME = 'isoforest.joblib'
_SCALER_FILENAME = 'scaler.joblib'
_METADATA_FILENAME = 'metadata.json'
_DEFAULT_MODEL_VERSION = 'isoforest-local'
_FEATURE_MODES = {'raw', 'spectral'}
_SPECTRAL_N_BANDS = 4


@dataclass(frozen=True)
class IsoForestAnomalyVerdict:
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


class IsoForestAnomalyInferenceService:
    def __init__(
        self,
        model: Any,
        scaler: Any,
        sensor_columns: Sequence[str],
        window_size: int,
        threshold: float,
        feature_mode: str = 'raw',
        model_version: str = _DEFAULT_MODEL_VERSION,
        spectral_n_bands: int = _SPECTRAL_N_BANDS,
    ) -> None:
        ordered = tuple(sensor_columns)
        if not ordered:
            raise ValueError('sensor_columns must not be empty')
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
            raise ValueError('window_size must be a positive integer')
        if not callable(getattr(model, 'score_samples', None)):
            raise TypeError('model must implement score_samples()')
        if scaler is not None and not callable(getattr(scaler, 'transform', None)):
            raise TypeError('scaler must implement transform()')
        if feature_mode not in _FEATURE_MODES:
            raise ValueError(f'feature_mode must be one of: {", ".join(sorted(_FEATURE_MODES))}')
        if isinstance(spectral_n_bands, bool) or not isinstance(spectral_n_bands, int) or spectral_n_bands <= 0:
            raise ValueError('spectral_n_bands must be a positive integer')

        expected_features = _feature_count(feature_mode, window_size, len(ordered), spectral_n_bands)
        n_features = int(getattr(model, 'n_features_in_', expected_features))
        if n_features != expected_features:
            raise ValueError(
                f'model expects {n_features} features but feature_mode={feature_mode} produces {expected_features}'
            )
        scaler_features = int(getattr(scaler, 'n_features_in_', expected_features)) if scaler is not None else expected_features
        if scaler_features != expected_features:
            raise ValueError(
                f'scaler expects {scaler_features} features but feature_mode={feature_mode} produces {expected_features}'
            )

        self._model = model
        self._scaler = scaler
        self._sensor_columns = ordered
        self._window_size = window_size
        self._threshold = float(threshold)
        self._feature_mode = feature_mode
        self._model_version = str(model_version)
        self._spectral_n_bands = spectral_n_bands
        self._buffers: dict[str, deque[tuple[str | None, list[float]]]] = {}

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
    def feature_mode(self) -> str:
        return self._feature_mode

    @classmethod
    def from_artifacts(cls, model_dir: str | Path, model_version: str | None = None) -> IsoForestAnomalyInferenceService:
        directory = Path(model_dir)
        model_path = directory / _MODEL_FILENAME
        scaler_path = directory / _SCALER_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(f'Isolation Forest model artifact not found: {model_path}')

        joblib = import_module('joblib')
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        sensor_columns, window_size, threshold, feature_mode, spectral_n_bands, metadata_version = cls._read_metadata(
            directory / _METADATA_FILENAME
        )
        return cls(
            model,
            scaler,
            sensor_columns,
            window_size,
            threshold,
            feature_mode,
            model_version or metadata_version,
            spectral_n_bands,
        )

    @staticmethod
    def _read_metadata(metadata_path: Path) -> tuple[list[str], int, float, str, int, str]:
        if not metadata_path.exists():
            raise FileNotFoundError(f'Isolation Forest metadata artifact not found: {metadata_path}')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        params = metadata.get('params') or {}
        sensor_columns = [str(column) for column in metadata.get('sensor_columns') or []]
        if not sensor_columns:
            sensor_columns = [str(column) for column in params.get('sensor_columns') or []]
        if not sensor_columns:
            skab_loader = import_module('ml.datasets.skab_loader')
            sensor_columns = list(skab_loader.SENSOR_COLUMNS)
        raw_window_size = params.get('window_size')
        if raw_window_size is None:
            raise ValueError('Isolation Forest metadata must include window_size')
        thresholds = metadata.get('thresholds') or {}
        threshold_value = thresholds.get('threshold', thresholds.get('q_threshold', params.get('threshold')))
        if threshold_value is None:
            raise ValueError('Isolation Forest metadata must include a threshold')
        feature_mode = str(params.get('feature_mode') or 'raw').lower()
        spectral_n_bands = int(params.get('spectral_n_bands') or _SPECTRAL_N_BANDS)
        version = str(params.get('model_version') or _DEFAULT_MODEL_VERSION)
        return sensor_columns, int(raw_window_size), float(threshold_value), feature_mode, spectral_n_bands, version

    def reset(self, station: str | None = None) -> None:
        if station is None:
            self._buffers.clear()
        else:
            self._buffers.pop(station, None)

    def observe(self, station: str, timestamp: str | None, sensors: Mapping[str, Any]) -> IsoForestAnomalyVerdict:
        if not isinstance(sensors, Mapping):
            raise TypeError('sensors must be a mapping of sensor name to value')

        row = self._extract_row(sensors)
        buffer = self._buffers.setdefault(station, deque(maxlen=self._window_size))
        buffer.append((timestamp, row))

        if len(buffer) < self._window_size:
            return self._warmup_verdict(station, timestamp)

        feature = self._build_feature(buffer)
        model_feature = self._scaler.transform(feature) if self._scaler is not None else feature
        score = float(-np.asarray(self._model.score_samples(model_feature), dtype=np.float64)[0])
        anomaly = int(score > self._threshold)

        return IsoForestAnomalyVerdict(
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
            top_contributing_sensor=self._top_contributing_sensor(feature),
        )

    def _extract_row(self, sensors: Mapping[str, Any]) -> list[float]:
        row: list[float] = []
        for column in self._sensor_columns:
            if column not in sensors:
                raise KeyError(f'missing sensor value: {column}')
            row.append(float(sensors[column]))
        return row

    def _warmup_verdict(self, station: str, timestamp: str | None) -> IsoForestAnomalyVerdict:
        return IsoForestAnomalyVerdict(
            station=station,
            timestamp=timestamp,
            window_filled=False,
            window_size=self._window_size,
            model_version=self._model_version,
            t2_threshold=self._threshold,
            q_threshold=self._threshold,
        )

    def _build_feature(self, buffer: deque[tuple[str | None, list[float]]]) -> np.ndarray:
        rows = [row for _timestamp, row in buffer]
        if self._feature_mode == 'raw':
            return np.asarray(rows, dtype=np.float64).reshape(1, -1)

        pd = import_module('pandas')
        timestamps = [timestamp if timestamp is not None else str(index) for index, (timestamp, _row) in enumerate(buffer)]
        frame = pd.DataFrame(rows, columns=list(self._sensor_columns))
        frame['datetime'] = timestamps
        frame['anomaly'] = 0
        frame['changepoint'] = 0
        features, _labels, _changepoints, _timestamps = build_spectral_window_features(
            frame,
            window_size=self._window_size,
            stride=self._window_size,
            sensor_columns=self._sensor_columns,
            n_bands=self._spectral_n_bands,
        )
        return np.asarray(features, dtype=np.float64)

    def _top_contributing_sensor(self, feature: np.ndarray) -> str | None:
        scaled = self._scaler.transform(feature) if self._scaler is not None else feature
        magnitudes = np.abs(np.asarray(scaled, dtype=np.float64).reshape(-1))
        sensor_count = len(self._sensor_columns)
        if self._feature_mode == 'raw':
            per_sensor = magnitudes.reshape(self._window_size, sensor_count).sum(axis=0)
        else:
            per_sensor = _spectral_sensor_magnitudes(magnitudes, sensor_count, self._spectral_n_bands)
        return self._sensor_columns[int(np.argmax(per_sensor))]


def _feature_count(feature_mode: str, window_size: int, sensor_count: int, n_bands: int) -> int:
    if feature_mode == 'spectral':
        return sensor_count * (n_bands + 4)
    return window_size * sensor_count


def _spectral_sensor_magnitudes(features: np.ndarray, sensor_count: int, n_bands: int) -> np.ndarray:
    means = features[0:sensor_count]
    stds = features[sensor_count : 2 * sensor_count]
    ranges = features[2 * sensor_count : 3 * sensor_count]
    band_start = 3 * sensor_count
    band_stop = band_start + sensor_count * n_bands
    bands = features[band_start:band_stop].reshape(sensor_count, n_bands).sum(axis=1)
    centroids = features[band_stop : band_stop + sensor_count]
    return means + stds + ranges + bands + centroids


__all__ = ['IsoForestAnomalyInferenceService', 'IsoForestAnomalyVerdict']
