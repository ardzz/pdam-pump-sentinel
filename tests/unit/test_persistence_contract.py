from routemq.settings import TelemetrySettings  # type: ignore[reportMissingImports]

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
    points = []

    async def fake_write(point):
        points.append(point)

    monkeypatch.setattr(persistence.telemetry, 'settings', TelemetrySettings(enabled=False), raising=True)
    monkeypatch.setattr(persistence.telemetry, 'write', fake_write, raising=True)

    await persistence.persist_telemetry('ipa_01', _reading(), _anomaly())

    assert points == []


async def test_history_persistence_when_enabled(monkeypatch):
    points = []

    async def fake_write(point):
        points.append(point)

    monkeypatch.setattr(persistence.telemetry, 'settings', TelemetrySettings(enabled=True), raising=True)
    monkeypatch.setattr(persistence.telemetry, 'write', fake_write, raising=True)

    await persistence.persist_telemetry('ipa_01', _reading(), _anomaly())

    assert len(points) == 1
    point = points[0]
    assert point.device_id == 'ipa_01'
    assert point.measurements['Pressure'].value == 2.1
    assert point.measurements['score'].value == 0.9
    assert point.measurements['anomaly'].value == 1
    assert point.tags == {'station': 'ipa_01'}
    assert point.attributes['source_timestamp'] == 't0'
