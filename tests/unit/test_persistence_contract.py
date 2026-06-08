from routemq.settings import TelemetrySettings  # type: ignore[reportMissingImports]
from routemq.tsdb.telemetry_adapters import ClickHouseTelemetryAdapter  # type: ignore[reportMissingImports]

from app.observability.metrics import PERSISTENCE_WRITES
from app.services import persistence


class FakeRedisManager:
    def __init__(self, enabled=True, fail=False):
        self._enabled = enabled
        self.fail = fail
        self.calls = []

    def is_enabled(self):
        return self._enabled

    async def set_json(self, key, value, ex=None, px=None, nx=False, xx=False):
        if self.fail:
            raise RuntimeError('redis unavailable')
        self.calls.append({'key': key, 'value': value})
        return True


class FakeClickHouseClient:
    def __init__(self):
        self.inserts = []

    async def insert(self, table, data, column_names):
        self.inserts.append({'table': table, 'data': data, 'column_names': column_names})


def _reading():
    return {'station': 'ipa_01', 'timestamp': 't0', 'sensors': {'Pressure': 2.1}}


def _anomaly():
    return {'station': 'ipa_01', 'score': 0.9, 'anomaly': 1, 'status': 'ok'}


async def test_persist_latest_writes_reading_and_anomaly_to_redis(monkeypatch):
    fake = FakeRedisManager(enabled=True)
    import routemq.redis_manager as redis_manager_module

    monkeypatch.setattr(redis_manager_module, 'redis_manager', fake, raising=True)

    PERSISTENCE_WRITES.clear()
    try:
        await persistence.persist_telemetry('ipa_01', _reading(), _anomaly())

        keys = [call['key'] for call in fake.calls]
        assert keys == ['pumpad:latest:reading:ipa_01', 'pumpad:latest:anomaly:ipa_01']
        assert PERSISTENCE_WRITES.labels(target='redis', result='success')._value.get() == 1
    finally:
        PERSISTENCE_WRITES.clear()


async def test_persist_latest_records_redis_errors(monkeypatch):
    fake = FakeRedisManager(enabled=True, fail=True)
    import routemq.redis_manager as redis_manager_module

    monkeypatch.setattr(redis_manager_module, 'redis_manager', fake, raising=True)
    monkeypatch.setattr(persistence.telemetry, 'settings', TelemetrySettings(enabled=False), raising=True)

    PERSISTENCE_WRITES.clear()
    try:
        await persistence.persist_telemetry('ipa_01', _reading(), _anomaly())

        assert fake.calls == []
        assert PERSISTENCE_WRITES.labels(target='redis', result='error')._value.get() == 1
    finally:
        PERSISTENCE_WRITES.clear()


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
    fake_clickhouse = FakeClickHouseClient()
    adapter = ClickHouseTelemetryAdapter('http://localhost:8123/default')
    adapter._client = fake_clickhouse

    async def fake_write(point):
        points.append(point)

    monkeypatch.setattr(persistence.telemetry, 'settings', TelemetrySettings(enabled=True), raising=True)
    monkeypatch.setattr(persistence.telemetry, 'write', fake_write, raising=True)
    monkeypatch.setattr(persistence.telemetry, 'adapter', adapter, raising=True)

    PERSISTENCE_WRITES.clear()
    try:
        await persistence.persist_telemetry('ipa_01', _reading(), _anomaly())

        assert len(points) == 1
        point = points[0]
        assert point.device_id == 'ipa_01'
        assert point.measurements['Pressure'].value == 2.1
        assert 'score' not in point.measurements
        assert 'anomaly' not in point.measurements
        assert point.tags == {'station': 'ipa_01'}
        assert point.attributes['source_timestamp'] == 't0'
        assert len(fake_clickhouse.inserts) == 1
        insert = fake_clickhouse.inserts[0]
        row = dict(zip(insert['column_names'], insert['data'][0]))
        assert row['device_id'] == 'ipa_01'
        assert row['measurement'] == 'anomaly_score'
        assert row['value_float'] == 0.9
        assert row['value_int'] == 1
        assert row['tags'] == {'station': 'ipa_01'}
        assert row['attributes']['source_timestamp'] == 't0'
        assert PERSISTENCE_WRITES.labels(target='clickhouse', result='success')._value.get() == 2
    finally:
        PERSISTENCE_WRITES.clear()


async def test_history_persistence_skips_anomaly_score_when_score_is_none(monkeypatch):
    points = []
    fake_clickhouse = FakeClickHouseClient()
    adapter = ClickHouseTelemetryAdapter('http://localhost:8123/default')
    adapter._client = fake_clickhouse

    async def fake_write(point):
        points.append(point)

    monkeypatch.setattr(persistence.telemetry, 'settings', TelemetrySettings(enabled=True), raising=True)
    monkeypatch.setattr(persistence.telemetry, 'write', fake_write, raising=True)
    monkeypatch.setattr(persistence.telemetry, 'adapter', adapter, raising=True)

    await persistence.persist_telemetry('ipa_01', _reading(), {'station': 'ipa_01', 'score': None, 'anomaly': None})

    assert len(points) == 1
    assert points[0].measurements['Pressure'].value == 2.1
    assert fake_clickhouse.inserts == []
