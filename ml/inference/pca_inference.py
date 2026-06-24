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

_DETECTOR_FILENAME = 'pca_detector.joblib'
_METADATA_FILENAME = 'metadata.json'
_DEFAULT_MODEL_VERSION = 'pca-local'
_FEATURE_MODES = {'raw', 'spectral'}
_SPECTRAL_N_BANDS = 4


@dataclass(frozen=True)
class AnomalyVerdict:
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


class PcaAnomalyInferenceService:
    # Raw feature layout MUST match ml.features.windowing.build_sensor_windows:
    # window_size consecutive readings, each ordered by sensor_columns, flattened
    # time-major (W, N).reshape(-1). Spectral mode delegates to
    # ml.features.spectral.build_spectral_window_features. A mismatch silently
    # produces wrong T²/Q scores.

    def __init__(
        self,
        detector: Any,
        sensor_columns: Sequence[str],
        window_size: int,
        model_version: str = _DEFAULT_MODEL_VERSION,
        feature_mode: str = 'raw',
        spectral_n_bands: int = _SPECTRAL_N_BANDS,
    ) -> None:
        ordered = tuple(sensor_columns)
        if not ordered:
            raise ValueError('sensor_columns must not be empty')
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
            raise ValueError('window_size must be a positive integer')
        normalized_feature_mode = str(feature_mode).lower()
        if normalized_feature_mode not in _FEATURE_MODES:
            supported_modes = ', '.join(sorted(_FEATURE_MODES))
            raise ValueError(
                f'unsupported PCA feature_mode: {feature_mode}; supported modes: {supported_modes}'
            )
        if isinstance(spectral_n_bands, bool) or not isinstance(spectral_n_bands, int) or spectral_n_bands <= 0:
            raise ValueError('spectral_n_bands must be a positive integer')

        for attribute in ('transform', 'score_samples'):
            if not callable(getattr(detector, attribute, None)):
                raise TypeError(f'detector must implement {attribute}()')
        if not hasattr(detector, 't2_threshold_') or not hasattr(detector, 'q_threshold_'):
            raise TypeError('detector must expose fitted t2_threshold_ and q_threshold_ attributes')

        expected_features = _feature_count(normalized_feature_mode, window_size, len(ordered), spectral_n_bands)
        n_features = int(getattr(detector, 'n_features_in_', expected_features))
        if n_features != expected_features:
            raise ValueError(
                f'detector expects {n_features} features but feature_mode={normalized_feature_mode} '
                f'produces {expected_features}'
            )

        self._detector = detector
        self._sensor_columns = ordered
        self._window_size = window_size
        self._model_version = str(model_version)
        self._feature_mode = normalized_feature_mode
        self._spectral_n_bands = spectral_n_bands
        self._t2_threshold = float(detector.t2_threshold_)
        self._q_threshold = float(detector.q_threshold_)
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
    def feature_mode(self) -> str:
        return self._feature_mode

    @property
    def spectral_n_bands(self) -> int:
        return self._spectral_n_bands

    @classmethod
    def from_artifacts(cls, model_dir: str | Path, model_version: str | None = None) -> PcaAnomalyInferenceService:
        directory = Path(model_dir)
        detector_path = directory / _DETECTOR_FILENAME
        if not detector_path.exists():
            raise FileNotFoundError(f'detector artifact not found: {detector_path}')

        joblib = import_module('joblib')
        detector = joblib.load(detector_path)
        sensor_columns, window_size, metadata_version, feature_mode, spectral_n_bands = cls._read_metadata(
            directory / _METADATA_FILENAME,
            detector,
        )
        return cls(
            detector,
            sensor_columns,
            window_size,
            model_version=model_version or metadata_version,
            feature_mode=feature_mode,
            spectral_n_bands=spectral_n_bands,
        )

    @staticmethod
    def _read_metadata(metadata_path: Path, detector: Any) -> tuple[list[str], int, str, str, int]:
        sensor_columns = [str(name) for name in getattr(detector, 'feature_names_in_', []) or []]
        window_size: int | None = None
        version = _DEFAULT_MODEL_VERSION
        feature_mode = 'raw'
        spectral_n_bands = _SPECTRAL_N_BANDS

        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            metadata_columns = metadata.get('sensor_columns')
            if metadata_columns:
                sensor_columns = [str(column) for column in metadata_columns]
            params = metadata.get('params') or {}
            window_size = params.get('window_size')
            version = str(params.get('model_version') or params.get('registered_model_name') or version)
            feature_mode = str(params.get('feature_mode') or feature_mode).lower()
            spectral_n_bands = int(params.get('spectral_n_bands') or spectral_n_bands)

        if not sensor_columns:
            skab_loader = import_module('ml.datasets.skab_loader')
            sensor_columns = list(skab_loader.SENSOR_COLUMNS)

        if feature_mode == 'spectral' and window_size is None:
            raise ValueError('PCA spectral metadata must include window_size')

        if window_size is None:
            n_features = int(getattr(detector, 'n_features_in_', len(sensor_columns)))
            window_size = max(1, n_features // len(sensor_columns))

        return sensor_columns, int(window_size), version, feature_mode, spectral_n_bands

    def reset(self, station: str | None = None) -> None:
        if station is None:
            self._buffers.clear()
        else:
            self._buffers.pop(station, None)

    def observe(self, station: str, timestamp: str | None, sensors: Mapping[str, Any]) -> AnomalyVerdict:
        if not isinstance(sensors, Mapping):
            raise TypeError('sensors must be a mapping of sensor name to value')

        row = self._extract_row(sensors)
        buffer = self._buffers.setdefault(station, deque(maxlen=self._window_size))
        buffer.append((timestamp, row))

        if len(buffer) < self._window_size:
            return self._warmup_verdict(station, timestamp)

        feature = self._build_feature(buffer)
        statistics = np.asarray(self._detector.transform(feature), dtype=np.float64)[0]
        t2 = float(statistics[0])
        q = float(statistics[1])
        score = float(np.asarray(self._detector.score_samples(feature), dtype=np.float64)[0])
        anomaly = int(t2 > self._t2_threshold or q > self._q_threshold)

        return AnomalyVerdict(
            station=station,
            timestamp=timestamp,
            window_filled=True,
            window_size=self._window_size,
            model_version=self._model_version,
            t2_threshold=self._t2_threshold,
            q_threshold=self._q_threshold,
            t2=t2,
            q=q,
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

    def _warmup_verdict(self, station: str, timestamp: str | None) -> AnomalyVerdict:
        return AnomalyVerdict(
            station=station,
            timestamp=timestamp,
            window_filled=False,
            window_size=self._window_size,
            model_version=self._model_version,
            t2_threshold=self._t2_threshold,
            q_threshold=self._q_threshold,
        )

    def _build_feature(self, buffer: deque[tuple[str | None, list[float]]]) -> np.ndarray:
        rows = [row for _timestamp, row in buffer]
        if self._feature_mode == 'raw':
            return np.asarray(rows, dtype=np.float64).reshape(1, -1)

        pd = import_module('pandas')
        timestamps = [
            timestamp if timestamp is not None else str(index)
            for index, (timestamp, _row) in enumerate(buffer)
        ]
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
        pca = getattr(self._detector, 'pca_', None)
        if pca is None:
            return None
        scaler = getattr(self._detector, 'scaler_', None)
        scaled = scaler.transform(feature) if scaler is not None else feature
        reconstructed = pca.inverse_transform(pca.transform(scaled))
        residual_squared = np.asarray((scaled - reconstructed) ** 2, dtype=np.float64)
        sensor_count = len(self._sensor_columns)
        if self._feature_mode == 'raw':
            per_sensor = residual_squared.reshape(self._window_size, sensor_count).sum(axis=0)
        else:
            per_sensor = _spectral_sensor_residuals(
                residual_squared.reshape(-1),
                sensor_count,
                self._spectral_n_bands,
            )
        return self._sensor_columns[int(np.argmax(per_sensor))]


def _feature_count(feature_mode: str, window_size: int, sensor_count: int, n_bands: int) -> int:
    if feature_mode == 'spectral':
        return sensor_count * (n_bands + 4)
    return window_size * sensor_count


def _spectral_sensor_residuals(features: np.ndarray, sensor_count: int, n_bands: int) -> np.ndarray:
    means = features[0:sensor_count]
    stds = features[sensor_count : 2 * sensor_count]
    ranges = features[2 * sensor_count : 3 * sensor_count]
    band_start = 3 * sensor_count
    band_stop = band_start + sensor_count * n_bands
    bands = features[band_start:band_stop].reshape(sensor_count, n_bands).sum(axis=1)
    centroids = features[band_stop : band_stop + sensor_count]
    return means + stds + ranges + bands + centroids


__all__ = ['AnomalyVerdict', 'PcaAnomalyInferenceService']
