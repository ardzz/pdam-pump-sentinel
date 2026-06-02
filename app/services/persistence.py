from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from routemq.telemetry import (  # type: ignore[reportMissingImports]
    Measurement,
    SchemaValidationResult,
    TelemetryAdapter,
    TelemetryHealthStatus,
    TelemetryPoint,
    WriteResult,
)

logger = logging.getLogger('PDAM.persistence')

LATEST_READING_KEY = 'pumpad:latest:reading:{station}'
LATEST_ANOMALY_KEY = 'pumpad:latest:anomaly:{station}'

_history_enabled = False


def enable_history_persistence(enabled: bool) -> None:
    global _history_enabled
    _history_enabled = bool(enabled)


def history_persistence_enabled() -> bool:
    return _history_enabled


class SensorReadingTelemetryAdapter(TelemetryAdapter):
    backend = 'mysql'

    async def write_many(self, points: Sequence[TelemetryPoint]) -> WriteResult:
        from routemq.model import Model  # type: ignore[reportMissingImports]

        from app.models.sensor_reading import SensorReading

        written = 0
        for point in points:
            await Model.create(
                SensorReading,
                station=point.device_id,
                source_timestamp=point.attributes.get('source_timestamp'),
                sensors={name: Measurement.from_value(measurement).value for name, measurement in point.measurements.items()},
                score=point.attributes.get('score'),
                anomaly=point.attributes.get('anomaly'),
            )
            written += 1
        return WriteResult(accepted=len(points), written=written)

    async def validate_schema(self) -> SchemaValidationResult:
        return SchemaValidationResult()

    async def health_check(self) -> TelemetryHealthStatus:
        return TelemetryHealthStatus(ok=True, backend=self.backend)

    async def close(self) -> None:
        return None


_history_adapter = SensorReadingTelemetryAdapter()


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
    sensors = reading_payload.get('sensors') or {}
    if not sensors:
        return
    try:
        point = TelemetryPoint(
            device_id=station,
            observed_at=None,
            measurements=dict(sensors),
            tags={'station': station},
            attributes={
                'source_timestamp': reading_payload.get('timestamp'),
                'score': anomaly_payload.get('score'),
                'anomaly': anomaly_payload.get('anomaly'),
            },
        )
        await _history_adapter.write_many([point])
    except Exception:
        logger.exception('failed to persist sensor reading to MySQL', extra={'station': station})
