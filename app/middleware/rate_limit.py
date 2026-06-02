from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from typing import Any

from routemq.middleware import Middleware  # type: ignore[reportMissingImports]


class InMemoryRateLimitMiddleware(Middleware):
    def __init__(
        self,
        max_messages: int = 50,
        window_seconds: float = 1.0,
        key: str = 'station',
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        if max_messages <= 0:
            raise ValueError('max_messages must be positive')
        if window_seconds <= 0:
            raise ValueError('window_seconds must be positive')
        self._max_messages = max_messages
        self._window_seconds = float(window_seconds)
        self._key = key
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def handle(
        self,
        context: dict[str, Any],
        next_handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        identity = str((context.get('params') or {}).get(self._key, 'global'))
        now = self._clock()
        bucket = self._hits[identity]
        while bucket and now - bucket[0] >= self._window_seconds:
            bucket.popleft()

        if len(bucket) >= self._max_messages:
            return {'accepted': False, 'reason': 'rate_limited', self._key: identity}

        bucket.append(now)
        return await next_handler(context)
