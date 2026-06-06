from __future__ import annotations

import json
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
    def __init__(self, result: Any):
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def query(self, query: str, parameters: dict[str, Any]):
        self.calls.append({'query': query, 'parameters': parameters})
        return self.result


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
