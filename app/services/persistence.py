from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger('PDAM.persistence')

LATEST_READING_KEY = 'pumpad:latest:reading:{station}'
LATEST_ANOMALY_KEY = 'pumpad:latest:anomaly:{station}'

_history_enabled = False


def enable_history_persistence(enabled: bool) -> None:
    global _history_enabled
    _history_enabled = bool(enabled)


def history_persistence_enabled() -> bool:
    return _history_enabled


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
    if not _history_enabled:
        return
    try:
        from routemq.model import Model  # type: ignore[reportMissingImports]

        from app.models.sensor_reading import SensorReading

        await Model.create(
            SensorReading,
            station=station,
            source_timestamp=reading_payload.get('timestamp'),
            sensors=dict(reading_payload.get('sensors') or {}),
            score=anomaly_payload.get('score'),
            anomaly=anomaly_payload.get('anomaly'),
        )
    except Exception:
        logger.exception('failed to persist sensor reading to MySQL', extra={'station': station})
