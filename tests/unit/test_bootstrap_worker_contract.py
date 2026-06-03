from routemq.job_registry import discover_and_register_jobs  # type: ignore[reportMissingImports]
from routemq.redis_manager import redis_manager  # type: ignore[reportMissingImports]
from routemq.router import Router  # type: ignore[reportMissingImports]

from bootstrap.app import Application


def _application() -> Application:
    return Application(router=Router(), show_banner=False, log_to_console=False)


async def test_initialize_connections_skips_redis_when_disabled(monkeypatch):
    calls = []

    async def fake_initialize():
        calls.append(True)
        return True

    application = _application()
    monkeypatch.setattr(redis_manager, 'initialize', fake_initialize, raising=True)
    monkeypatch.delenv('ENABLE_REDIS', raising=False)
    monkeypatch.setenv('ENABLE_TELEMETRY', 'false')

    await application._initialize_connections()

    assert calls == []


async def test_initialize_connections_initializes_redis_when_enabled(monkeypatch):
    calls = []

    async def fake_initialize():
        calls.append(True)
        return True

    application = _application()
    monkeypatch.setattr(redis_manager, 'initialize', fake_initialize, raising=True)
    monkeypatch.setenv('ENABLE_REDIS', 'true')
    monkeypatch.setenv('ENABLE_TELEMETRY', 'false')

    await application._initialize_connections()

    assert calls == [True]


def test_discover_and_register_jobs_returns_list():
    result = discover_and_register_jobs('app.jobs')

    assert isinstance(result, list)
