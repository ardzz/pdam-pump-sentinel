from __future__ import annotations

from app.models.operator_label import parse_operator_label_payload
from app.services.persistence import persist_operator_label


class LabelController:
    @staticmethod
    async def ingest(station: str, payload: object, client: object) -> dict[str, bool | int | str]:
        _ = client
        label_payload = parse_operator_label_payload(payload)
        await persist_operator_label(station, label_payload)
        return {'accepted': True, 'station': station, 'label': label_payload.label}
