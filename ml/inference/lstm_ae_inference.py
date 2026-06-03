from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

_MODEL_FILENAME = 'lstm_ae.keras'
_SCALER_FILENAME = 'scaler.joblib'
_METADATA_FILENAME = 'metadata.json'
_DEFAULT_MODEL_VERSION = 'lstm-ae-local'


@dataclass(frozen=True)
class LstmAeAnomalyVerdict:
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


class LstmAeAnomalyInferenceService:
    def __init__(
        self,
        model: Any,
        scaler: Any,
        sensor_columns: Sequence[str],
        window_size: int,
        threshold: float,
        model_version: str = _DEFAULT_MODEL_VERSION,
    ) -> None:
        ordered = tuple(sensor_columns)
        if not ordered:
            raise ValueError('sensor_columns must not be empty')
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
            raise ValueError('window_size must be a positive integer')
        if not callable(getattr(model, 'predict', None)):
            raise TypeError('model must implement predict()')
        if not callable(getattr(scaler, 'transform', None)):
            raise TypeError('scaler must implement transform()')

        self._validate_model_shape(model, window_size, len(ordered))
        self._model = model
        self._scaler = scaler
        self._sensor_columns = ordered
        self._window_size = window_size
        self._threshold = float(threshold)
        self._model_version = str(model_version)
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

    @property
    def threshold(self) -> float:
        return self._threshold

    @classmethod
    def from_artifacts(cls, model_dir: str | Path, model_version: str | None = None) -> LstmAeAnomalyInferenceService:
        directory = Path(model_dir)
        model_path = directory / _MODEL_FILENAME
        scaler_path = directory / _SCALER_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(f'LSTM-AE model artifact not found: {model_path}')
        if not scaler_path.exists():
            raise FileNotFoundError(f'LSTM-AE scaler artifact not found: {scaler_path}')

        keras = import_module('keras')
        joblib = import_module('joblib')
        model = keras.saving.load_model(str(model_path), compile=False)
        scaler = joblib.load(scaler_path)
        sensor_columns, window_size, threshold, metadata_version = cls._read_metadata(directory / _METADATA_FILENAME)
        return cls(model, scaler, sensor_columns, window_size, threshold, model_version or metadata_version)

    @staticmethod
    def _read_metadata(metadata_path: Path) -> tuple[list[str], int, float, str]:
        if not metadata_path.exists():
            raise FileNotFoundError(f'LSTM-AE metadata artifact not found: {metadata_path}')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        sensor_columns = [str(column) for column in metadata.get('sensor_columns') or []]
        params = metadata.get('params') or {}
        if not sensor_columns:
            sensor_columns = [str(column) for column in params.get('sensor_columns') or []]
        if not sensor_columns:
            skab_loader = import_module('ml.datasets.skab_loader')
            sensor_columns = list(skab_loader.SENSOR_COLUMNS)
        raw_window_size = params.get('window_size')
        if raw_window_size is None:
            raise ValueError('LSTM-AE metadata must include window_size')
        window_size = int(raw_window_size)
        thresholds = metadata.get('thresholds') or {}
        threshold_value = thresholds.get('threshold', thresholds.get('q_threshold', params.get('threshold')))
        if threshold_value is None:
            raise ValueError('LSTM-AE metadata must include a threshold')
        version = str(params.get('model_version') or _DEFAULT_MODEL_VERSION)
        return sensor_columns, window_size, float(threshold_value), version

    def reset(self, station: str | None = None) -> None:
        if station is None:
            self._buffers.clear()
        else:
            self._buffers.pop(station, None)

    def observe(self, station: str, timestamp: str | None, sensors: Mapping[str, Any]) -> LstmAeAnomalyVerdict:
        if not isinstance(sensors, Mapping):
            raise TypeError('sensors must be a mapping of sensor name to value')

        row = self._extract_row(sensors)
        buffer = self._buffers.setdefault(station, deque(maxlen=self._window_size))
        buffer.append(row)

        if len(buffer) < self._window_size:
            return self._warmup_verdict(station, timestamp)

        window = np.asarray(buffer, dtype=np.float64)
        scaled = self._scaler.transform(window).reshape(1, self._window_size, len(self._sensor_columns))
        reconstructed = np.asarray(self._model.predict(scaled, verbose=0), dtype=np.float64)
        residual = np.abs(scaled - reconstructed)
        score = float(np.mean(residual))
        anomaly = int(score > self._threshold)
        per_sensor = residual.reshape(self._window_size, len(self._sensor_columns)).mean(axis=0)

        return LstmAeAnomalyVerdict(
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
            top_contributing_sensor=self._sensor_columns[int(np.argmax(per_sensor))],
        )

    def _extract_row(self, sensors: Mapping[str, Any]) -> list[float]:
        row: list[float] = []
        for column in self._sensor_columns:
            if column not in sensors:
                raise KeyError(f'missing sensor value: {column}')
            row.append(float(sensors[column]))
        return row

    def _warmup_verdict(self, station: str, timestamp: str | None) -> LstmAeAnomalyVerdict:
        return LstmAeAnomalyVerdict(
            station=station,
            timestamp=timestamp,
            window_filled=False,
            window_size=self._window_size,
            model_version=self._model_version,
            t2_threshold=self._threshold,
            q_threshold=self._threshold,
        )

    @staticmethod
    def _validate_model_shape(model: Any, window_size: int, sensor_count: int) -> None:
        input_shape = getattr(model, 'input_shape', None)
        if not isinstance(input_shape, tuple) or len(input_shape) < 3:
            return
        expected = (window_size, sensor_count)
        actual = tuple(input_shape[-2:])
        if actual != expected:
            raise ValueError(f'model expects window shape {actual} but metadata declares {expected}')


__all__ = ['LstmAeAnomalyInferenceService', 'LstmAeAnomalyVerdict']
