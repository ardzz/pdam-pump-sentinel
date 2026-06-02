from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from routemq.middleware import Middleware  # type: ignore[reportMissingImports]


class ValidateTelemetryMiddleware(Middleware):
    def __init__(self) -> None:
        super().__init__()

    async def handle(
        self,
        context: dict[str, Any],
        next_handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        payload = context.get('payload')
        if not isinstance(payload, Mapping):
            raise ValueError('telemetry payload must be a JSON object')

        sensors = payload.get('sensors')
        if not isinstance(sensors, Mapping) or not sensors:
            raise ValueError('telemetry payload must include a non-empty sensors object')

        return await next_handler(context)
