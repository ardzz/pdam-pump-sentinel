from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from dashboard import data


class FakeRedis:
    def __init__(self, payloads: dict[str, str | bytes]):
        self.payloads = payloads
        self.keys: list[str] = []

    def get(self, key: str):
        self.keys.append(key)
        return self.payloads.get(key)


class FakeClickHouseClient:
    def __init__(self, result: Any = None):
        self.result = result
        self.calls: list[dict[str, Any]] = []
        self.inserts: list[dict[str, Any]] = []

    def query(self, query: str, parameters: dict[str, Any]):
        self.calls.append({'query': query, 'parameters': parameters})
        return self.result

    def insert(self, table: str, data: Any, column_names: Any = None):
        self.inserts.append({'table': table, 'data': data, 'column_names': column_names})


class FailingInsertClickHouseClient(FakeClickHouseClient):
    def insert(self, table: str, data: Any, column_names: Any = None):
        raise RuntimeError('clickhouse insert failed')


class FakeRedisWriter:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def set(self, key: str, value: str, ex: int | None = None):
        self.calls.append({'key': key, 'value': value, 'ex': ex})
        return True


def _raise_clickhouse():
    raise RuntimeError('clickhouse unavailable')


class FakeNamedResult:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def named_results(self):
        return iter(self.rows)


class FakeRowResult:
    column_names = ('observed_at', 'measurement', 'value_float', 'value_int')

    def __init__(self, rows: list[tuple[Any, ...]]):
        self.result_rows = rows


class FakeMlflowClient:
    def __init__(self, versions: list[Any]):
        self.versions = versions
        self.filters: list[str] = []

    def search_model_versions(self, filter_string: str):
        self.filters.append(filter_string)
        return self.versions


def _raise():
    raise data.redis.RedisError('offline')


@pytest.fixture(autouse=True)
def clear_dashboard_data_caches():
    for reader_name in (
        'get_latest_reading',
        'get_latest_anomaly',
        'get_drift_result',
        'get_retrain_result',
        'get_active_model',
        'get_anomaly_history',
        'get_model_versions',
        'list_stations',
    ):
        getattr(data, reader_name).clear()
    yield
    for reader_name in (
        'get_latest_reading',
        'get_latest_anomaly',
        'get_drift_result',
        'get_retrain_result',
        'get_active_model',
        'get_anomaly_history',
        'get_model_versions',
        'list_stations',
    ):
        getattr(data, reader_name).clear()


@pytest.fixture(autouse=True)
def _clear_streamlit_caches():
    cached_readers = (
        data.get_latest_reading,
        data.get_latest_anomaly,
        data.get_drift_result,
        data.get_retrain_result,
        data.get_active_model,
        data.get_anomaly_history,
        data.get_model_versions,
        data.list_stations,
    )
    for reader in cached_readers:
        reader.clear()
    yield
    for reader in cached_readers:
        reader.clear()


@pytest.mark.parametrize(
    ('reader_name', 'args', 'key', 'payload'),
    [
        ('get_latest_reading', ('ipa_01',), 'pumpad:latest:reading:ipa_01', {'station': 'ipa_01', 'flow': 1.2}),
        ('get_latest_anomaly', ('ipa_01',), 'pumpad:latest:anomaly:ipa_01', {'station': 'ipa_01', 'score': 0.91}),
        ('get_drift_result', (), 'pumpad:drift:result', {'dataset_drift': True, 'drift_share': 0.75}),
        ('get_retrain_result', (), 'pumpad:retrain:result', {'promoted': False, 'reason': 'guardrail'}),
        (
            'get_active_model',
            (),
            'pumpad:active:model',
            {
                'registered_model_name': 'PumpAD',
                'alias': 'champion',
                'name': 'PumpAD',
                'version': '1',
                'activated_at': '2026-06-04T00:00:00+00:00',
            },
        ),
    ],
)
def test_redis_readers_parse_json(monkeypatch, reader_name, args, key, payload):
    fake = FakeRedis({key: json.dumps(payload).encode('utf-8')})
    monkeypatch.setattr(data, '_redis_client', lambda: fake)

    assert getattr(data, reader_name)(*args) == payload
    assert fake.keys == [key]


@pytest.mark.parametrize(
    ('reader_name', 'args'),
    [
        ('get_latest_reading', ('ipa_01',)),
        ('get_latest_anomaly', ('ipa_01',)),
        ('get_drift_result', ()),
        ('get_retrain_result', ()),
        ('get_active_model', ()),
    ],
)
def test_redis_readers_return_none_when_factory_fails(monkeypatch, reader_name, args):
    monkeypatch.setattr(data, '_redis_client', _raise)

    assert getattr(data, reader_name)(*args) is None


def test_redis_readers_return_none_for_parse_failure(monkeypatch):
    fake = FakeRedis({'pumpad:latest:reading:ipa_01': '{bad-json'})
    monkeypatch.setattr(data, '_redis_client', lambda: fake)

    assert data.get_latest_reading('ipa_01') is None


def test_get_anomaly_history_uses_clickhouse_named_results(monkeypatch):
    rows = [
        {'observed_at': '2026-06-03T00:00:00', 'measurement': 'anomaly_score', 'value_float': 0.91, 'value_int': 1},
    ]
    fake = FakeClickHouseClient(FakeNamedResult(rows))
    monkeypatch.setattr(data, '_clickhouse_client', lambda: fake)

    assert data.get_anomaly_history('ipa_01', limit=2) == rows
    assert fake.calls[0]['parameters'] == {'d': 'ipa_01', 'n': 2}
    assert 'telemetry_observations' in fake.calls[0]['query']


def test_get_anomaly_history_supports_clickhouse_result_rows(monkeypatch):
    fake = FakeClickHouseClient(FakeRowResult([('2026-06-03T00:00:00', 'anomaly_score', 0.91, 1)]))
    monkeypatch.setattr(data, '_clickhouse_client', lambda: fake)

    assert data.get_anomaly_history('ipa_01', limit=1) == [
        {'observed_at': '2026-06-03T00:00:00', 'measurement': 'anomaly_score', 'value_float': 0.91, 'value_int': 1}
    ]


def test_get_anomaly_history_returns_empty_when_factory_fails(monkeypatch):
    monkeypatch.setattr(data, '_clickhouse_client', _raise)

    assert data.get_anomaly_history('ipa_01') == []


def test_get_model_versions_returns_version_aliases_and_run_id(monkeypatch):
    versions = [
        SimpleNamespace(version=2, aliases=('champion', 'candidate'), run_id='run-2'),
        SimpleNamespace(version='1', aliases=None, run_id=None),
    ]
    fake = FakeMlflowClient(versions)
    monkeypatch.setattr(data, '_mlflow_client', lambda: fake)

    assert data.get_model_versions() == [
        {'version': '2', 'aliases': ['champion', 'candidate'], 'run_id': 'run-2'},
        {'version': '1', 'aliases': [], 'run_id': None},
    ]
    assert fake.filters == ["name='PumpAD'"]


def test_get_model_versions_returns_empty_when_factory_fails(monkeypatch):
    monkeypatch.setattr(data, '_mlflow_client', _raise)

    assert data.get_model_versions() == []


def test_list_stations_defaults(monkeypatch):
    monkeypatch.delenv('STATIONS', raising=False)

    assert data.list_stations() == ['ipa_01']


def test_list_stations_parses_comma_separated_env(monkeypatch):
    monkeypatch.setenv('STATIONS', 'ipa_01, ipa_02,,ipa_03 ')

    assert data.list_stations() == ['ipa_01', 'ipa_02', 'ipa_03']


def test_timestamp_age_seconds_parses_iso_timestamp():
    now = datetime(2026, 6, 8, 0, 1, tzinfo=timezone.utc)

    assert data.timestamp_age_seconds('2026-06-08T00:00:00+00:00', now=now) == 60
    assert data.timestamp_age_seconds('not-a-timestamp', now=now) is None


def test_get_observability_snapshot_classifies_green_state(monkeypatch):
    now = datetime(2026, 6, 8, 0, 1, tzinfo=timezone.utc)
    fake = FakeRedis(
        {
            'pumpad:latest:reading:ipa_01': json.dumps({'station': 'ipa_01', 'timestamp': '2026-06-08T00:00:30+00:00'}),
            'pumpad:latest:anomaly:ipa_01': json.dumps({'station': 'ipa_01', 'score': 0.1, 'source_timestamp': '2026-06-08T00:00:30+00:00'}),
            'pumpad:drift:result': json.dumps({'dataset_drift': False, 'timestamp': '2026-06-08T00:00:00+00:00'}),
            'pumpad:retrain:result': json.dumps({'success': True, 'version': '2'}),
            'pumpad:active:model': json.dumps({'name': 'PumpAD', 'version': '2', 'activated_at': '2026-06-08T00:00:00+00:00'}),
        }
    )
    monkeypatch.setattr(data, '_redis_client', lambda: fake)

    snapshot = data.get_observability_snapshot('ipa_01', now=now)

    assert snapshot['state'] == 'GREEN'
    assert snapshot['components'] == {'telemetry': 'GREEN', 'drift_report': 'GREEN', 'active_model': 'GREEN'}
    assert snapshot['telemetry_age_seconds'] == 30
    assert snapshot['drift_report_age_seconds'] == 60
    assert snapshot['active_model_age_seconds'] == 60
    assert snapshot['retrain_result'] == 'SUCCESS'


def test_get_observability_snapshot_flags_stale_and_drift(monkeypatch):
    now = datetime(2026, 6, 8, 0, 10, tzinfo=timezone.utc)
    fake = FakeRedis(
        {
            'pumpad:latest:reading:ipa_01': json.dumps({'station': 'ipa_01', 'timestamp': '2026-06-08T00:00:00+00:00'}),
            'pumpad:drift:result': json.dumps({'dataset_drift': True, 'timestamp': '2026-06-08T00:00:00+00:00'}),
            'pumpad:retrain:result': json.dumps({'promoted': False, 'reason': 'guardrail'}),
            'pumpad:active:model': json.dumps({'name': 'PumpAD', 'version': '2', 'activated_at': '2026-06-02T00:00:00+00:00'}),
        }
    )
    monkeypatch.setattr(data, '_redis_client', lambda: fake)

    snapshot = data.get_observability_snapshot('ipa_01', now=now)

    assert snapshot['state'] == 'RED'
    assert snapshot['components']['telemetry'] == 'RED'
    assert snapshot['components']['drift_report'] == 'RED'
    assert snapshot['components']['active_model'] == 'DEGRADED'
    assert snapshot['drift_detected'] is True
    assert snapshot['retrain_result'] == 'REJECTED'


OPERATOR_ACTION_HISTORY_COLUMNS = [
    'action_id',
    'station',
    'source_timestamp',
    'action_type',
    'operator_id',
    'note',
    'reason',
    'mute_until',
    'payload_json',
    'created_at',
]


def test_record_operator_action_history_shapes_ack_row(monkeypatch):
    fake = FakeClickHouseClient()
    monkeypatch.setattr(data, '_clickhouse_client', lambda: fake)
    now = datetime(2026, 6, 8, 0, 0, tzinfo=timezone.utc)

    result = data._record_operator_action_history(
        'ack',
        'ipa_01',
        {
            '_ts': '2026-06-08T00:00:00+00:00',
            'source_timestamp': '2026-06-08T00:00:00+00:00',
            'operator_id': 'op-1',
            'reason': 'confirmed leak signature',
        },
        now=now,
    )

    assert result is True
    assert len(fake.inserts) == 1
    insert = fake.inserts[0]
    assert insert['table'] == 'operator_actions'
    assert insert['column_names'] == OPERATOR_ACTION_HISTORY_COLUMNS
    assert len(insert['data']) == 1
    row = dict(zip(insert['column_names'], insert['data'][0]))
    assert row['station'] == 'ipa_01'
    assert row['action_type'] == 'ack'
    assert row['source_timestamp'] == '2026-06-08T00:00:00+00:00'
    assert row['operator_id'] == 'op-1'
    assert row['reason'] == 'confirmed leak signature'
    assert row['note'] is None
    assert row['mute_until'] is None
    assert row['created_at'] == now
    assert isinstance(row['action_id'], str) and row['action_id']
    assert json.loads(row['payload_json']) == {
        'source_timestamp': '2026-06-08T00:00:00+00:00',
        'operator_id': 'op-1',
        'reason': 'confirmed leak signature',
    }


def test_record_operator_action_history_shapes_mute_row(monkeypatch):
    fake = FakeClickHouseClient()
    monkeypatch.setattr(data, '_clickhouse_client', lambda: fake)
    now = datetime(2026, 6, 8, 0, 0, tzinfo=timezone.utc)

    result = data._record_operator_action_history(
        'mute',
        'ipa_01',
        {'source_timestamp': '2026-06-08T00:00:00+00:00', 'operator_id': 'op-2'},
        ttl_seconds=900,
        now=now,
    )

    assert result is True
    insert = fake.inserts[0]
    assert insert['table'] == 'operator_actions'
    row = dict(zip(insert['column_names'], insert['data'][0]))
    assert row['action_type'] == 'mute'
    assert row['mute_until'] == now + timedelta(seconds=900)
    assert row['note'] is None
    assert row['reason'] is None


def test_record_operator_action_history_shapes_note_row(monkeypatch):
    fake = FakeClickHouseClient()
    monkeypatch.setattr(data, '_clickhouse_client', lambda: fake)
    now = datetime(2026, 6, 8, 0, 0, tzinfo=timezone.utc)

    result = data._record_operator_action_history(
        'note',
        'ipa_01',
        {'timestamp': '2026-06-08T00:00:00+00:00', 'note': 'pump bearing noise', 'operator_id': 'op-3'},
        now=now,
    )

    assert result is True
    row = dict(zip(fake.inserts[0]['column_names'], fake.inserts[0]['data'][0]))
    assert row['action_type'] == 'note'
    assert row['note'] == 'pump bearing noise'
    assert row['source_timestamp'] == '2026-06-08T00:00:00+00:00'
    assert row['mute_until'] is None


def test_record_operator_action_history_falls_back_to_legacy_operator(monkeypatch):
    fake = FakeClickHouseClient()
    monkeypatch.setattr(data, '_clickhouse_client', lambda: fake)
    now = datetime(2026, 6, 8, 0, 0, tzinfo=timezone.utc)

    result = data._record_operator_action_history(
        'ack',
        'ipa_01',
        {'source_timestamp': '2026-06-08T00:00:00+00:00', 'operator': 'legacy'},
        now=now,
    )

    assert result is True
    row = dict(zip(fake.inserts[0]['column_names'], fake.inserts[0]['data'][0]))
    assert row['operator_id'] == 'legacy'


def test_record_operator_action_history_returns_false_when_client_unavailable(monkeypatch):
    monkeypatch.setattr(data, '_clickhouse_client', _raise_clickhouse)

    assert data._record_operator_action_history('ack', 'ipa_01', {'source_timestamp': 't0'}) is False


def test_record_operator_action_history_returns_false_when_insert_fails(monkeypatch):
    fake = FailingInsertClickHouseClient()
    monkeypatch.setattr(data, '_clickhouse_client', lambda: fake)

    assert data._record_operator_action_history('note', 'ipa_01', {'source_timestamp': 't0', 'note': 'x'}) is False
    assert fake.inserts == []


def test_record_operator_action_writes_redis_state_and_history(monkeypatch):
    redis_fake = FakeRedisWriter()
    ch_fake = FakeClickHouseClient()
    monkeypatch.setattr(data, '_redis_client', lambda: redis_fake)
    monkeypatch.setattr(data, '_clickhouse_client', lambda: ch_fake)

    result = data.record_operator_action(
        'ack',
        'ipa_01',
        {'_ts': '2026-06-08T00:00:00+00:00', 'operator_id': 'op-1', 'note': 'Acked from UI'},
        30,
    )

    assert result.ok is True
    assert result.history_ok is True
    assert bool(result) is True
    assert redis_fake.calls[0]['key'] == 'pumpad:anomaly:ack:ipa_01:2026-06-08T00:00:00+00:00'
    assert redis_fake.calls[0]['ex'] == 30
    assert json.loads(redis_fake.calls[0]['value']) == {'operator_id': 'op-1', 'note': 'Acked from UI'}
    assert len(ch_fake.inserts) == 1
    insert = ch_fake.inserts[0]
    assert insert['table'] == 'operator_actions'
    assert insert['column_names'] == OPERATOR_ACTION_HISTORY_COLUMNS
    row = dict(zip(insert['column_names'], insert['data'][0]))
    assert row['station'] == 'ipa_01'
    assert row['action_type'] == 'ack'
    assert row['source_timestamp'] == '2026-06-08T00:00:00+00:00'
    assert row['operator_id'] == 'op-1'


def test_record_operator_action_widget_ack_payload_records_dashboard_operator(monkeypatch):
    redis_fake = FakeRedisWriter()
    ch_fake = FakeClickHouseClient()
    monkeypatch.setattr(data, '_redis_client', lambda: redis_fake)
    monkeypatch.setattr(data, '_clickhouse_client', lambda: ch_fake)

    result = data.record_operator_action(
        'ack',
        'ipa_01',
        {
            '_ts': '2026-06-08T00:00:00+00:00',
            'operator_id': 'dashboard',
            'ack_at': '2026-06-08T00:00:01+00:00',
            'note': 'Acked from UI',
        },
        30,
    )

    assert result.ok is True
    assert result.history_ok is True
    row = dict(zip(ch_fake.inserts[0]['column_names'], ch_fake.inserts[0]['data'][0]))
    assert row['operator_id'] == 'dashboard'


@pytest.mark.parametrize(
    ('kind', 'ttl_seconds', 'payload', 'expected_key'),
    [
        ('ack', 30, {'_ts': '2026-06-08T00:00:00+00:00'}, 'pumpad:anomaly:ack:ipa_01:2026-06-08T00:00:00+00:00'),
        ('mute', 900, {'source_timestamp': '2026-06-08T00:00:00+00:00'}, 'pumpad:anomaly:mute:ipa_01'),
        ('note', 30, {'_ts': '2026-06-08T00:00:00+00:00', 'note': 'bearing'}, 'pumpad:anomaly:note:ipa_01:2026-06-08T00:00:00+00:00'),
    ],
)
def test_record_operator_action_persists_each_kind(monkeypatch, kind, ttl_seconds, payload, expected_key):
    redis_fake = FakeRedisWriter()
    ch_fake = FakeClickHouseClient()
    monkeypatch.setattr(data, '_redis_client', lambda: redis_fake)
    monkeypatch.setattr(data, '_clickhouse_client', lambda: ch_fake)

    result = data.record_operator_action(kind, 'ipa_01', payload, ttl_seconds)

    assert result.ok is True
    assert result.history_ok is True
    assert redis_fake.calls[0]['key'] == expected_key
    assert redis_fake.calls[0]['ex'] == ttl_seconds
    row = dict(zip(ch_fake.inserts[0]['column_names'], ch_fake.inserts[0]['data'][0]))
    assert row['action_type'] == kind


def test_record_operator_action_history_failure_keeps_redis_success(monkeypatch):
    redis_fake = FakeRedisWriter()
    ch_fake = FailingInsertClickHouseClient()
    monkeypatch.setattr(data, '_redis_client', lambda: redis_fake)
    monkeypatch.setattr(data, '_clickhouse_client', lambda: ch_fake)

    result = data.record_operator_action('mute', 'ipa_01', {'source_timestamp': 't0'}, 900)

    assert result.ok is True
    assert result.history_ok is False
    assert bool(result) is True
    assert redis_fake.calls[0]['key'] == 'pumpad:anomaly:mute:ipa_01'
    assert ch_fake.inserts == []


def test_record_operator_action_redis_failure_skips_history(monkeypatch):
    ch_fake = FakeClickHouseClient()
    monkeypatch.setattr(data, '_redis_client', _raise)
    monkeypatch.setattr(data, '_clickhouse_client', lambda: ch_fake)

    result = data.record_operator_action('ack', 'ipa_01', {'_ts': 't0'}, 30)

    assert result.ok is False
    assert result.history_ok is False
    assert bool(result) is False
    assert ch_fake.inserts == []
    assert data.get_last_error() == 'offline'
