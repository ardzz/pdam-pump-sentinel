from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

from app.models.anomaly_event import (
    build_anomaly_payload,
    build_anomaly_payload_from_verdict,
    build_anomaly_topic,
)
from app.observability.metrics import (
    ANOMALY_EVENTS,
    ANOMALY_SCORE,
    INFERENCE_EVENTS,
    INFERENCE_LATENCY,
    TELEMETRY_FRESHNESS,
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
        t0 = time.perf_counter()
        try:
            verdict = service.observe(station, timestamp, sensors)
        except Exception:
            INFERENCE_EVENTS.labels(station=station, model_version='', result='error').inc()
            raise
        elapsed = time.perf_counter() - t0
        model_version = str(verdict.model_version)
        INFERENCE_EVENTS.labels(station=station, model_version=model_version, result='success').inc()
        INFERENCE_LATENCY.labels(station=station, model_version=str(verdict.model_version)).observe(elapsed)
        if verdict.score is not None:
            score = float(verdict.score)
            ANOMALY_SCORE.labels(station=station).observe(score)
            if verdict.anomaly:
                ANOMALY_EVENTS.labels(station=station, severity=_severity(score), model_version=model_version).inc()
        TELEMETRY_FRESHNESS.labels(station=station).set(0.0)
        return build_anomaly_payload_from_verdict(verdict)

    return build_anomaly_payload(station, payload)


def _severity(score: float) -> str:
    if score < 0.5:
        return 'low'
    if score < 1.0:
        return 'medium'
    return 'high'
