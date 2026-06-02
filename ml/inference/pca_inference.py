from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

_DETECTOR_FILENAME = 'pca_detector.joblib'
_METADATA_FILENAME = 'metadata.json'
_DEFAULT_MODEL_VERSION = 'pca-local'


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
    # Feature layout MUST match ml.features.windowing.build_sensor_windows:
    # window_size consecutive readings, each ordered by sensor_columns, flattened
    # time-major (W, N).reshape(-1). A mismatch silently produces wrong T²/Q scores.

    def __init__(
        self,
        detector: Any,
        sensor_columns: Sequence[str],
        window_size: int,
        model_version: str = _DEFAULT_MODEL_VERSION,
    ) -> None:
        ordered = tuple(sensor_columns)
        if not ordered:
            raise ValueError('sensor_columns must not be empty')
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
            raise ValueError('window_size must be a positive integer')

        for attribute in ('transform', 'score_samples'):
            if not callable(getattr(detector, attribute, None)):
                raise TypeError(f'detector must implement {attribute}()')
        if not hasattr(detector, 't2_threshold_') or not hasattr(detector, 'q_threshold_'):
            raise TypeError('detector must expose fitted t2_threshold_ and q_threshold_ attributes')

        expected_features = window_size * len(ordered)
        n_features = int(getattr(detector, 'n_features_in_', expected_features))
        if n_features != expected_features:
            raise ValueError(
                f'detector expects {n_features} features but window_size={window_size} '
                f'x {len(ordered)} sensors = {expected_features}'
            )

        self._detector = detector
        self._sensor_columns = ordered
        self._window_size = window_size
        self._model_version = str(model_version)
        self._t2_threshold = float(detector.t2_threshold_)
        self._q_threshold = float(detector.q_threshold_)
        self._buffers: dict[str, deque[list[float]]] = {}

    @property
    def sensor_columns(self) -> tuple[str, ...]:
        return self._sensor_columns

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def model_version(self) -> str:
        return self._model_version

    @classmethod
    def from_artifacts(cls, model_dir: str | Path, model_version: str | None = None) -> PcaAnomalyInferenceService:
        directory = Path(model_dir)
        detector_path = directory / _DETECTOR_FILENAME
        if not detector_path.exists():
            raise FileNotFoundError(f'detector artifact not found: {detector_path}')

        joblib = import_module('joblib')
        detector = joblib.load(detector_path)
        sensor_columns, window_size, metadata_version = cls._read_metadata(directory / _METADATA_FILENAME, detector)
        return cls(detector, sensor_columns, window_size, model_version or metadata_version)

    @staticmethod
    def _read_metadata(metadata_path: Path, detector: Any) -> tuple[list[str], int, str]:
        sensor_columns = [str(name) for name in getattr(detector, 'feature_names_in_', []) or []]
        window_size: int | None = None
        version = _DEFAULT_MODEL_VERSION

        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            metadata_columns = metadata.get('sensor_columns')
            if metadata_columns:
                sensor_columns = [str(column) for column in metadata_columns]
            params = metadata.get('params') or {}
            window_size = params.get('window_size')
            version = str(params.get('registered_model_name') or version)

        if not sensor_columns:
            skab_loader = import_module('ml.datasets.skab_loader')
            sensor_columns = list(skab_loader.SENSOR_COLUMNS)

        if window_size is None:
            n_features = int(getattr(detector, 'n_features_in_', len(sensor_columns)))
            window_size = max(1, n_features // len(sensor_columns))

        return sensor_columns, int(window_size), version

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
        buffer.append(row)

        if len(buffer) < self._window_size:
            return self._warmup_verdict(station, timestamp)

        feature = np.asarray(buffer, dtype=np.float64).reshape(1, -1)
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

    def _top_contributing_sensor(self, feature: np.ndarray) -> str | None:
        pca = getattr(self._detector, 'pca_', None)
        if pca is None:
            return None
        scaler = getattr(self._detector, 'scaler_', None)
        scaled = scaler.transform(feature) if scaler is not None else feature
        reconstructed = pca.inverse_transform(pca.transform(scaled))
        residual_squared = np.asarray((scaled - reconstructed) ** 2, dtype=np.float64)
        per_sensor = residual_squared.reshape(self._window_size, len(self._sensor_columns)).sum(axis=0)
        return self._sensor_columns[int(np.argmax(per_sensor))]


__all__ = ['AnomalyVerdict', 'PcaAnomalyInferenceService']
