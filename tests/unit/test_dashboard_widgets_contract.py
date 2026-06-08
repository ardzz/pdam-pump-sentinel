from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from dashboard import data, widgets


class FakeRedis:
    def __init__(self):
        self.calls = []

    def set(self, key, value, ex=None):
        self.calls.append({'key': key, 'value': value, 'ex': ex})
        return True


@pytest.mark.parametrize(
    ('timestamp', 'threshold', 'expected'),
    [
        ((datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat(), 60, 'color:green'),
        ((datetime.now(timezone.utc) - timedelta(minutes=4)).isoformat(), 60, 'STALE'),
        (None, 60, 'No data'),
    ],
)
def test_freshness_pill_renders_state(monkeypatch, timestamp, threshold, expected):
    rendered = []
    monkeypatch.setattr(widgets.st, 'markdown', lambda body, **_kwargs: rendered.append(body))

    widgets.freshness_pill(timestamp, threshold_seconds=threshold)

    assert expected in rendered[0]


def test_record_operator_action_writes_ack_key(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(data, '_redis_client', lambda: fake)

    ok = data.record_operator_action(
        'ack',
        'ipa_01',
        {'_ts': '2026-06-06T00:00:00+00:00', 'operator': 'dashboard', 'ack_at': 'now', 'note': 'Acked from UI'},
        30,
    )

    assert ok is True
    assert fake.calls[0]['key'] == 'pumpad:anomaly:ack:ipa_01:2026-06-06T00:00:00+00:00'
    assert fake.calls[0]['ex'] == 30
    assert json.loads(fake.calls[0]['value']) == {'operator': 'dashboard', 'ack_at': 'now', 'note': 'Acked from UI'}


@pytest.mark.parametrize(
    ('mlflow_ok', 'redis_ok', 'clickhouse_ok', 'mqtt_ok', 'active_ok', 'telemetry_ok', 'expected'),
    [
        (True, True, True, True, True, True, 'GREEN'),
        (True, True, True, True, False, True, 'DEGRADED'),
        (True, False, True, True, True, True, 'RED'),
    ],
)
def test_render_global_status_banner_composite_state(
    monkeypatch,
    mlflow_ok,
    redis_ok,
    clickhouse_ok,
    mqtt_ok,
    active_ok,
    telemetry_ok,
    expected,
):
    monkeypatch.setattr(widgets.st, 'markdown', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(widgets.st, 'caption', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(widgets, '_probe_mlflow', lambda: (mlflow_ok, 'mlflow'))
    monkeypatch.setattr(widgets, '_probe_redis', lambda: (redis_ok, 'redis'))
    monkeypatch.setattr(widgets, '_probe_clickhouse', lambda: (clickhouse_ok, 'clickhouse'))
    monkeypatch.setattr(widgets, '_probe_mqtt', lambda: (mqtt_ok, 'mqtt'))
    monkeypatch.setattr(widgets, '_probe_active_model', lambda: (active_ok, 'active'))
    monkeypatch.setattr(widgets, '_probe_telemetry_freshness', lambda _station: (telemetry_ok, 'telemetry'))
    monkeypatch.setattr(data, '_last_error', None)

    assert widgets.render_global_status_banner('ipa_01') == expected


def test_render_global_status_banner_reports_actionable_failed_checks(monkeypatch):
    captions = []
    monkeypatch.setattr(widgets.st, 'markdown', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(widgets.st, 'caption', lambda body: captions.append(body))
    monkeypatch.setattr(
        widgets,
        'collect_status_checks',
        lambda _station: {
            'MLflow': (True, 'mlflow ok'),
            'Redis': (True, 'redis ok'),
            'ClickHouse': (True, 'clickhouse ok'),
            'MQTT': (True, 'mqtt ok'),
            'Active model': (True, 'active ok'),
            'Telemetry': (False, 'telemetry stale degraded age 600s'),
        },
    )
    monkeypatch.setattr(data, '_last_error', None)

    assert widgets.render_global_status_banner('ipa_01') == 'DEGRADED'
    assert any('Action needed: Telemetry: telemetry stale degraded age 600s' in caption for caption in captions)


def test_composite_state_treats_critical_telemetry_as_red():
    assert (
        widgets._composite_state(
            {
                'MLflow': (True, 'mlflow ok'),
                'Redis': (True, 'redis ok'),
                'ClickHouse': (True, 'clickhouse ok'),
                'MQTT': (True, 'mqtt ok'),
                'Active model': (True, 'active ok'),
                'Telemetry': (False, 'telemetry stale critical age 1200s'),
            }
        )
        == 'RED'
    )


def test_service_probe_urls_are_configurable(monkeypatch):
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'http://mlflow.local:5001')
    monkeypatch.setenv('TELEMETRY_URL', 'http://user:pass@clickhouse.local:18124/default')
    monkeypatch.setenv('MQTT_HOST', 'mqtt.local')
    monkeypatch.setenv('MQTT_PORT', '11884')

    assert widgets._mlflow_health_url() == 'http://mlflow.local:5001/health'
    assert widgets._clickhouse_ping_url() == 'http://clickhouse.local:18124/ping'
    assert widgets._mqtt_endpoint() == ('mqtt.local', 11884)


def test_service_probe_explicit_urls_override_base_config(monkeypatch):
    monkeypatch.setenv('DASHBOARD_MLFLOW_HEALTH_URL', 'http://mlflow-proxy/ready')
    monkeypatch.setenv('DASHBOARD_CLICKHOUSE_PING_URL', 'http://clickhouse-proxy/ping')

    assert widgets._mlflow_health_url() == 'http://mlflow-proxy/ready'
    assert widgets._clickhouse_ping_url() == 'http://clickhouse-proxy/ping'
