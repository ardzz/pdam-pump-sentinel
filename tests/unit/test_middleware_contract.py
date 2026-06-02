import pytest

from app.middleware import (
    CorrelationLoggingMiddleware,
    InMemoryRateLimitMiddleware,
    ValidateTelemetryMiddleware,
)


def _context(payload, params=None):
    return {
        'topic': 'factory/skab/ipa_01/telemetry',
        'payload': payload,
        'params': params or {'station': 'ipa_01'},
        'client': object(),
        'correlation_id': 'cid-1',
    }


async def _passthrough(context):
    return {'handled': True, 'station': context['params'].get('station')}


async def test_validate_middleware_accepts_valid_telemetry():
    middleware = ValidateTelemetryMiddleware()
    context = _context({'sensors': {'Pressure': 2.1}, 'timestamp': 't0'})

    result = await middleware.handle(context, _passthrough)

    assert result == {'handled': True, 'station': 'ipa_01'}


async def test_validate_middleware_rejects_missing_sensors():
    middleware = ValidateTelemetryMiddleware()

    with pytest.raises(ValueError):
        await middleware.handle(_context({'timestamp': 't0'}), _passthrough)
    with pytest.raises(ValueError):
        await middleware.handle(_context(['not', 'a', 'mapping']), _passthrough)


async def test_rate_limit_middleware_blocks_over_threshold_then_recovers():
    clock = {'now': 0.0}
    middleware = InMemoryRateLimitMiddleware(max_messages=2, window_seconds=1.0, clock=lambda: clock['now'])
    context = _context({'sensors': {'Pressure': 2.1}})

    assert (await middleware.handle(context, _passthrough))['handled'] is True
    assert (await middleware.handle(context, _passthrough))['handled'] is True
    blocked = await middleware.handle(context, _passthrough)
    assert blocked == {'accepted': False, 'reason': 'rate_limited', 'station': 'ipa_01'}

    clock['now'] = 1.5
    assert (await middleware.handle(context, _passthrough))['handled'] is True


async def test_rate_limit_middleware_isolates_stations():
    clock = {'now': 0.0}
    middleware = InMemoryRateLimitMiddleware(max_messages=1, window_seconds=1.0, clock=lambda: clock['now'])

    first = await middleware.handle(_context({'sensors': {'Pressure': 1.0}}, {'station': 'a'}), _passthrough)
    second = await middleware.handle(_context({'sensors': {'Pressure': 1.0}}, {'station': 'b'}), _passthrough)

    assert first['handled'] is True and second['handled'] is True


async def test_correlation_middleware_passes_through():
    middleware = CorrelationLoggingMiddleware()
    context = _context({'sensors': {'Pressure': 2.1}})

    result = await middleware.handle(context, _passthrough)

    assert result == {'handled': True, 'station': 'ipa_01'}
