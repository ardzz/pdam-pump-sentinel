from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from routemq.telemetry import (  # type: ignore[reportMissingImports]
    TelemetryPoint,
    telemetry,
)
from routemq.tsdb.telemetry_adapters import (  # type: ignore[reportMissingImports]
    CLICKHOUSE_TELEMETRY_COLUMNS,
    ClickHouseTelemetryAdapter,
)

from app.observability.metrics import PERSISTENCE_WRITES

logger = logging.getLogger('PDAM.persistence')

LATEST_READING_KEY = 'pumpad:latest:reading:{station}'
LATEST_ANOMALY_KEY = 'pumpad:latest:anomaly:{station}'
ANOMALY_SCORE_MEASUREMENT = 'anomaly_score'


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
        PERSISTENCE_WRITES.labels(target='redis', result='success').inc()
    except Exception:
        PERSISTENCE_WRITES.labels(target='redis', result='error').inc()
        logger.exception('failed to persist latest reading to Redis', extra={'station': station})


async def _persist_history(
    station: str,
    reading_payload: Mapping[str, Any],
    anomaly_payload: Mapping[str, Any],
) -> None:
    if not telemetry.settings.enabled:
        return
    sensor_values: dict[str, Any] = {}
    sensors = reading_payload.get('sensors')
    if isinstance(sensors, Mapping):
        sensor_values.update(sensors)
    if not sensor_values and anomaly_payload.get('score') is None:
        return
    attributes: dict[str, str] = {}
    source_timestamp = reading_payload.get('timestamp')
    if source_timestamp is not None:
        attributes['source_timestamp'] = str(source_timestamp)
    try:
        if sensor_values:
            point = TelemetryPoint(
                device_id=station,
                observed_at=None,
                measurements=sensor_values,
                tags={'station': station},
                attributes=attributes,
            )
            await telemetry.write(point)
            if isinstance(telemetry.adapter, ClickHouseTelemetryAdapter):
                PERSISTENCE_WRITES.labels(target='clickhouse', result='success').inc()
        await _persist_anomaly_score_row(station, anomaly_payload, attributes)
    except Exception:
        if isinstance(telemetry.adapter, ClickHouseTelemetryAdapter):
            PERSISTENCE_WRITES.labels(target='clickhouse', result='error').inc()
        logger.exception('failed to emit telemetry history point', extra={'station': station})


async def _persist_anomaly_score_row(
    station: str,
    anomaly_payload: Mapping[str, Any],
    attributes: Mapping[str, str],
) -> None:
    score = anomaly_payload.get('score')
    if score is None:
        return
    adapter = telemetry.adapter
    if not isinstance(adapter, ClickHouseTelemetryAdapter):
        return
    if adapter._client is None:
        await adapter.connect()
    anomaly = anomaly_payload.get('anomaly')
    now = datetime.now(UTC)
    row = {
        'observed_at': now,
        'ingested_at': now,
        'device_id': station,
        'measurement': ANOMALY_SCORE_MEASUREMENT,
        'value_float': float(score),
        'value_int': int(anomaly) if anomaly is not None else None,
        'value_string': None,
        'value_bool': None,
        'unit': None,
        'quality': None,
        'tags': {'station': station},
        'attributes': dict(attributes),
        'metadata': {},
    }
    await adapter._client.insert(
        adapter.table,
        [tuple(row.get(column) for column in CLICKHOUSE_TELEMETRY_COLUMNS)],
        column_names=list(CLICKHOUSE_TELEMETRY_COLUMNS),
    )
    PERSISTENCE_WRITES.labels(target='clickhouse', result='success').inc()
