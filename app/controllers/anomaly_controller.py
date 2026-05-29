from __future__ import annotations

import json
from typing import Any

from app.models.anomaly_event import build_anomaly_payload, build_anomaly_topic


class Controller:
    @staticmethod
    async def ingest(station: str, payload: Any, client: Any) -> dict[str, Any]:
        anomaly_payload = build_anomaly_payload(station, payload)
        client.publish(build_anomaly_topic(station), json.dumps(anomaly_payload), qos=1)
        return {'accepted': True, 'station': station}
