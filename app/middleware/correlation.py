from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from routemq.middleware import Middleware  # type: ignore[reportMissingImports]


class CorrelationLoggingMiddleware(Middleware):
    def __init__(self, logger: logging.Logger | None = None) -> None:
        super().__init__()
        self._logger = logger or logging.getLogger('PDAM.telemetry')

    async def handle(
        self,
        context: dict[str, Any],
        next_handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        self._logger.debug(
            'telemetry received',
            extra={
                'correlation_id': context.get('correlation_id'),
                'mqtt_topic': context.get('topic'),
            },
        )
        return await next_handler(context)
