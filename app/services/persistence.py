from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from routemq.telemetry import (  # type: ignore[reportMissingImports]
    TelemetryPoint,
    telemetry,
)

logger = logging.getLogger('PDAM.persistence')

LATEST_READING_KEY = 'pumpad:latest:reading:{station}'
LATEST_ANOMALY_KEY = 'pumpad:latest:anomaly:{station}'


async def persist_telemetry(
    station: str,
    reading_payload: Mapping[str, Any],
    anomaly_payload: Mapping[str, Any],
) -> None:
    await _persist_latest(station, reading_payload, anomaly_payload)
    await _persist_history(station, reading_payload, anomaly_payload)


async def _persist_latest(
    station: str,
    reading_payload: Mapping[str, Any],
    anomaly_payload: Mapping[str, Any],
) -> None:
    try:
        from routemq.redis_manager import redis_manager  # type: ignore[reportMissingImports]

        if not redis_manager.is_enabled():
            return
        await redis_manager.set_json(LATEST_READING_KEY.format(station=station), dict(reading_payload))
        await redis_manager.set_json(LATEST_ANOMALY_KEY.format(station=station), dict(anomaly_payload))
    except Exception:
        logger.exception('failed to persist latest reading to Redis', extra={'station': station})


async def _persist_history(
    station: str,
    reading_payload: Mapping[str, Any],
    anomaly_payload: Mapping[str, Any],
) -> None:
    if not telemetry.settings.enabled:
        return
    measurements: dict[str, Any] = {}
    sensors = reading_payload.get('sensors')
    if isinstance(sensors, Mapping):
        measurements.update(sensors)
    for name in ('score', 'anomaly'):
        value = anomaly_payload.get(name)
        if value is not None:
            measurements[name] = value
    if not measurements:
        return
    attributes: dict[str, str] = {}
    source_timestamp = reading_payload.get('timestamp')
    if source_timestamp is not None:
        attributes['source_timestamp'] = str(source_timestamp)
    try:
        point = TelemetryPoint(
            device_id=station,
            observed_at=None,
            measurements=measurements,
            tags={'station': station},
            attributes=attributes,
        )
        await telemetry.write(point)
    except Exception:
        logger.exception('failed to emit telemetry history point', extra={'station': station})
