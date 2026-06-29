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


class _FakeColumn:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class FakeStreamlit:
    def __init__(self, clicked_labels=(), note_text=''):
        self.clicked = set(clicked_labels)
        self.note_text = note_text
        self.session_state: dict[str, object] = {}
        self.successes: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def columns(self, count):
        return tuple(_FakeColumn() for _ in range(count))

    def button(self, label, key=None):
        return label in self.clicked

    def text_input(self, label, key=None):
        return self.note_text

    def success(self, message):
        self.successes.append(message)

    def warning(self, message):
        self.warnings.append(message)

    def error(self, message):
        self.errors.append(message)


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
    monkeypatch.setattr(data, '_record_operator_action_history', lambda *args, **kwargs: True)

    result = data.record_operator_action(
        'ack',
        'ipa_01',
        {'_ts': '2026-06-06T00:00:00+00:00', 'operator_id': 'dashboard', 'ack_at': 'now', 'note': 'Acked from UI'},
        30,
    )

    assert bool(result) is True
    assert result.ok is True
    assert result.history_ok is True
    assert fake.calls[0]['key'] == 'pumpad:anomaly:ack:ipa_01:2026-06-06T00:00:00+00:00'
    assert fake.calls[0]['ex'] == 30
    assert json.loads(fake.calls[0]['value']) == {'operator_id': 'dashboard', 'ack_at': 'now', 'note': 'Acked from UI'}


def test_write_operator_action_full_success_shows_no_warning(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(widgets, 'st', fake_st)
    monkeypatch.setattr(
        data,
        'record_operator_action',
        lambda *args, **kwargs: data.OperatorActionResult(ok=True, history_ok=True),
    )

    widgets._write_operator_action('ack', 'ipa_01', {}, 30, 'Acknowledged anomaly.')

    assert fake_st.successes == ['Acknowledged anomaly.']
    assert fake_st.warnings == []
    assert fake_st.errors == []


def test_write_operator_action_history_failure_surfaces_warning(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(widgets, 'st', fake_st)
    monkeypatch.setattr(
        data,
        'record_operator_action',
        lambda *args, **kwargs: data.OperatorActionResult(ok=True, history_ok=False),
    )

    widgets._write_operator_action('mute', 'ipa_01', {}, 900, 'Muted station for 15 minutes.')

    assert fake_st.successes == ['Muted station for 15 minutes.']
    assert len(fake_st.warnings) == 1
    assert fake_st.errors == []


def test_write_operator_action_redis_failure_shows_error(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(widgets, 'st', fake_st)
    monkeypatch.setattr(
        data,
        'record_operator_action',
        lambda *args, **kwargs: data.OperatorActionResult(ok=False, history_ok=False),
    )

    widgets._write_operator_action('ack', 'ipa_01', {}, 30, 'Acknowledged anomaly.')

    assert fake_st.successes == []
    assert fake_st.warnings == []
    assert fake_st.errors == ['Operator action failed. Redis is unavailable.']


@pytest.mark.parametrize(
    ('clicked', 'expected_kind'),
    [
        ({'Acknowledge'}, 'ack'),
        ({'Mute 15m'}, 'mute'),
        ({'Add note', 'Submit note'}, 'note'),
    ],
)
def test_operator_action_buttons_route_expected_kind(monkeypatch, clicked, expected_kind):
    captured: list[str] = []
    monkeypatch.setattr(widgets, '_write_operator_action', lambda kind, *args, **kwargs: captured.append(kind))
    fake_st = FakeStreamlit(clicked_labels=clicked, note_text='bearing noise')
    monkeypatch.setattr(widgets, 'st', fake_st)

    widgets.operator_action_buttons({'source_timestamp': '2026-06-06T00:00:00+00:00'}, 'ipa_01')

    assert captured == [expected_kind]


def test_operator_action_buttons_submits_note_payload_as_note(monkeypatch):
    captured: list[tuple[str, str, dict[str, object], int]] = []
    monkeypatch.setattr(
        widgets,
        '_write_operator_action',
        lambda kind, station, payload, ttl, message: captured.append((kind, station, payload, ttl)),
    )
    fake_st = FakeStreamlit(clicked_labels={'Add note', 'Submit note'}, note_text='bearing noise')
    monkeypatch.setattr(widgets, 'st', fake_st)

    widgets.operator_action_buttons({'source_timestamp': '2026-06-06T00:00:00+00:00'}, 'ipa_01')

    assert len(captured) == 1
    kind, station, payload, ttl = captured[0]
    assert kind == 'note'
    assert station == 'ipa_01'
    assert payload['note'] == 'bearing noise'
    assert payload['operator_id'] == 'dashboard'
    assert ttl == widgets.ACK_TTL_SECONDS


@pytest.mark.parametrize(
    ('clicked', 'expected_kind'),
    [
        ({'Acknowledge'}, 'ack'),
        ({'Mute 15m'}, 'mute'),
        ({'Add note', 'Submit note'}, 'note'),
    ],
)
def test_operator_action_buttons_payloads_include_operator_id(monkeypatch, clicked, expected_kind):
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        widgets,
        '_write_operator_action',
        lambda kind, station, payload, ttl, message: captured.append(payload),
    )
    fake_st = FakeStreamlit(clicked_labels=clicked, note_text='bearing noise')
    monkeypatch.setattr(widgets, 'st', fake_st)

    widgets.operator_action_buttons({'source_timestamp': '2026-06-06T00:00:00+00:00'}, 'ipa_01')

    assert len(captured) == 1
    assert captured[0]['operator_id'] == 'dashboard'
    assert 'operator' not in captured[0]


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
