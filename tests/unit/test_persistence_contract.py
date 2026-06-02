
from routemq.settings import TelemetrySettings  # type: ignore[reportMissingImports]
from routemq.telemetry import telemetry  # type: ignore[reportMissingImports]

from app.services import persistence


class FakeRedisManager:
    def __init__(self, enabled=True):
        self._enabled = enabled
        self.calls = []

    def is_enabled(self):
        return self._enabled

    async def set_json(self, key, value, ex=None, px=None, nx=False, xx=False):
        self.calls.append({'key': key, 'value': value})
        return True


def _reading():
    return {'station': 'ipa_01', 'timestamp': 't0', 'sensors': {'Pressure': 2.1}}


def _anomaly():
    return {'station': 'ipa_01', 'score': 0.9, 'anomaly': 1, 'status': 'ok'}


async def test_persist_latest_writes_reading_and_anomaly_to_redis(monkeypatch):
    fake = FakeRedisManager(enabled=True)
    import routemq.redis_manager as redis_manager_module

    monkeypatch.setattr(redis_manager_module, 'redis_manager', fake, raising=True)

    await persistence.persist_telemetry('ipa_01', _reading(), _anomaly())

    keys = [call['key'] for call in fake.calls]
    assert keys == ['pumpad:latest:reading:ipa_01', 'pumpad:latest:anomaly:ipa_01']


async def test_persist_latest_skips_when_redis_disabled(monkeypatch):
    fake = FakeRedisManager(enabled=False)
    import routemq.redis_manager as redis_manager_module

    monkeypatch.setattr(redis_manager_module, 'redis_manager', fake, raising=True)

    await persistence.persist_telemetry('ipa_01', _reading(), _anomaly())

    assert fake.calls == []


async def test_history_persistence_disabled_by_default(monkeypatch):
    created = []

    async def fake_create(model_class, **kwargs):
        created.append(kwargs)

    import routemq.model as model_module

    monkeypatch.setattr(model_module.Model, 'create', classmethod(lambda cls, mc, **kw: fake_create(mc, **kw)), raising=True)
    monkeypatch.setattr(persistence, '_history_enabled', False, raising=True)

    await persistence.persist_telemetry('ipa_01', _reading(), _anomaly())

    assert created == []


async def test_history_persistence_when_enabled(monkeypatch):
    created = []

    async def fake_create(cls, model_class, **kwargs):
        created.append({'model': model_class.__name__, **kwargs})

    import routemq.model as model_module

    monkeypatch.setattr(model_module.Model, 'create', classmethod(fake_create), raising=True)
    persistence.enable_history_persistence(True)
    await telemetry.start(
        adapter=persistence.SensorReadingTelemetryAdapter(),
        settings=TelemetrySettings(enabled=True),
    )
    try:
        await persistence.persist_telemetry('ipa_01', _reading(), _anomaly())
        assert created == []
        await telemetry.flush()
    finally:
        await telemetry.close()
        persistence.enable_history_persistence(False)

    assert len(created) == 1
    assert created[0]['model'] == 'SensorReading'
    assert created[0]['station'] == 'ipa_01'
    assert created[0]['anomaly'] == 1
