from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from app.models.anomaly_event import (
    build_anomaly_payload,
    build_anomaly_payload_from_verdict,
    build_anomaly_topic,
)
from app.services.inference import get_inference_service
from app.services.persistence import persist_telemetry


class Controller:
    @staticmethod
    async def ingest(station: str, payload: Any, client: Any) -> dict[str, Any]:
        anomaly_payload = _score_payload(station, payload)
        client.publish(build_anomaly_topic(station), json.dumps(anomaly_payload), qos=1)
        reading_payload = payload if isinstance(payload, Mapping) else {}
        await persist_telemetry(station, reading_payload, anomaly_payload)
        return {'accepted': True, 'station': station}


def _score_payload(station: str, payload: Any) -> dict[str, Any]:
    service = get_inference_service()
    sensors = payload.get('sensors') if isinstance(payload, Mapping) else None

    if service is not None and isinstance(sensors, Mapping) and sensors:
        timestamp = payload.get('timestamp') if isinstance(payload, Mapping) else None
        verdict = service.observe(station, timestamp, sensors)
        return build_anomaly_payload_from_verdict(verdict)

    return build_anomaly_payload(station, payload)
