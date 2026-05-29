from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any


def build_anomaly_topic(station: str) -> str:
    return f'factory/skab/{station}/anomaly'


def build_anomaly_payload(station: str, telemetry: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(telemetry, Mapping):
        raise TypeError('telemetry payload must be a mapping')

    sensors = telemetry.get('sensors')
    if sensors is not None and not isinstance(sensors, Mapping):
        raise ValueError('telemetry payload sensors must be a mapping when provided')

    labels = telemetry.get('labels', {})
    return {
        'station': station,
        'anomaly': _demo_anomaly_flag(labels),
        'source_timestamp': telemetry.get('timestamp'),
    }


def _demo_anomaly_flag(labels: Any) -> int:
    if not isinstance(labels, Mapping):
        return 0

    raw_flag = labels.get('anomaly', 0)
    if isinstance(raw_flag, bool):
        return int(raw_flag)
    if isinstance(raw_flag, Real):
        return int(raw_flag != 0)

    try:
        return int(str(raw_flag))
    except (TypeError, ValueError):
        return 0
