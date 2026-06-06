from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import streamlit as st

from dashboard import data

ACK_TTL_SECONDS = 30 * 24 * 60 * 60


def render_global_status_banner(station: str | None = None) -> str:
    checks = collect_status_checks(station)
    state = _composite_state(checks)
    color = {'GREEN': 'green', 'DEGRADED': 'orange', 'RED': 'red'}[state]
    summary = ' · '.join(f'{name}: {"OK" if ok else "FAIL"}' for name, (ok, _detail) in checks.items())

    st.markdown(f'### :{color}[{state}] Operator Console Health')
    st.caption(summary)
    redis_error = data.get_last_error()
    if redis_error:
        st.caption(f'Redis unreachable: {redis_error}')
    return state


def freshness_pill(timestamp_iso: str | None, threshold_seconds: int = 60) -> None:
    parsed = _parse_timestamp(timestamp_iso)
    if parsed is None:
        st.markdown('<span style="color:gray">No data</span>', unsafe_allow_html=True)
        return

    age_seconds = max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))
    relative = _relative_short(age_seconds)
    if age_seconds <= threshold_seconds:
        st.markdown(f'<span style="color:green">Updated {relative}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span style="color:red">Updated {relative} — STALE</span>', unsafe_allow_html=True)


def last_updated_caption(ts: str | None) -> None:
    parsed = _parse_timestamp(ts)
    if parsed is None:
        st.caption('Last updated: N/A')
        return
    age_seconds = max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))
    st.caption(f'Last updated: {_relative_short(age_seconds)}')


def operator_action_buttons(anomaly_payload: dict[str, Any], station: str) -> None:
    iso_ts = _event_iso_ts(anomaly_payload)
    ack_payload = {
        '_ts': iso_ts,
        'operator': 'dashboard',
        'ack_at': datetime.now(timezone.utc).isoformat(),
        'note': 'Acked from UI',
    }
    button_suffix = f'{station}:{iso_ts}'

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button('Acknowledge', key=f'ack-{button_suffix}'):
            _write_operator_action('ack', station, ack_payload, ACK_TTL_SECONDS, 'Acknowledged anomaly.')
    with c2:
        if st.button('Mute 15m', key=f'mute-{button_suffix}'):
            until_ts = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
            mute_payload = {'until_ts': until_ts, 'operator': 'dashboard', 'reason': 'Muted from UI'}
            _write_operator_action('mute', station, mute_payload, 15 * 60, 'Muted station for 15 minutes.')
    with c3:
        note_key = f'note-open-{button_suffix}'
        if st.button('Add note', key=f'add-note-{button_suffix}'):
            st.session_state[note_key] = True

    if st.session_state.get(note_key):
        note = st.text_input('Operator note', key=f'note-text-{button_suffix}')
        if st.button('Submit note', key=f'submit-note-{button_suffix}'):
            note_payload = {
                '_ts': iso_ts,
                'operator': 'dashboard',
                'ack_at': datetime.now(timezone.utc).isoformat(),
                'note': note or 'Operator note submitted from UI',
            }
            _write_operator_action('ack', station, note_payload, ACK_TTL_SECONDS, 'Saved operator note.')


def collect_status_checks(station: str | None = None) -> dict[str, tuple[bool, str]]:
    return {
        'MLflow': _probe_mlflow(),
        'Redis': _probe_redis(),
        'ClickHouse': _probe_clickhouse(),
        'MQTT': _probe_mqtt(),
        'Active model': _probe_active_model(),
        'Telemetry': _probe_telemetry_freshness(station),
    }


def _write_operator_action(kind: str, station: str, payload: dict[str, Any], ttl_seconds: int, success_message: str) -> None:
    try:
        ok = data.record_operator_action(kind, station, payload, ttl_seconds)
    except Exception as exc:
        st.error(f'Operator action failed: {exc}')
        return
    if ok:
        st.success(success_message)
    else:
        st.error('Operator action failed. Redis is unavailable.')


def _probe_mlflow() -> tuple[bool, str]:
    return _http_probe('http://localhost:5000/health')


def _probe_clickhouse() -> tuple[bool, str]:
    return _http_probe('http://localhost:18123/ping')


def _probe_mqtt() -> tuple[bool, str]:
    return _tcp_probe('localhost', 11883)


def _probe_redis() -> tuple[bool, str]:
    try:
        client = data._redis_client()
        return bool(client.ping()), 'Redis ping OK'
    except Exception as exc:
        return False, str(exc)


def _probe_active_model() -> tuple[bool, str]:
    active = data.get_active_model()
    if not active:
        return False, 'pumpad:active:model missing'
    activated_at = active.get('activated_at') or active.get('activated_ts') or active.get('updated_at')
    parsed = _parse_timestamp(None if activated_at is None else str(activated_at))
    if parsed is None:
        return False, 'active model timestamp missing'
    age = datetime.now(timezone.utc) - parsed
    return age < timedelta(hours=24), f'active model age {int(age.total_seconds() // 60)}m'


def _probe_telemetry_freshness(station: str | None) -> tuple[bool, str]:
    if not station:
        return False, 'station not selected'
    history = data.get_anomaly_history(station, limit=1)
    if not history:
        return False, 'no recent observations'
    latest = history[0].get('observed_at')
    parsed = _parse_timestamp(str(latest) if latest is not None else None)
    if parsed is None:
        return False, 'latest observation timestamp invalid'
    age = datetime.now(timezone.utc) - parsed
    return age < timedelta(minutes=5), f'latest observation age {int(age.total_seconds())}s'


def _http_probe(url: str) -> tuple[bool, str]:
    try:
        request = Request(url, method='GET')
        with urlopen(request, timeout=0.5) as response:
            return 200 <= response.status < 400, f'HTTP {response.status}'
    except (OSError, URLError) as exc:
        return False, str(exc)


def _tcp_probe(host: str, port: int) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True, f'TCP {host}:{port} reachable'
    except OSError as exc:
        return False, str(exc)


def _composite_state(checks: dict[str, tuple[bool, str]]) -> str:
    if all(ok for ok, _detail in checks.values()):
        return 'GREEN'
    service_down = any(not checks[name][0] for name in ('MLflow', 'Redis', 'ClickHouse', 'MQTT'))
    return 'RED' if service_down else 'DEGRADED'


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _relative_short(seconds: int) -> str:
    if seconds < 60:
        return f'{seconds}s ago'
    minutes = seconds // 60
    if minutes < 60:
        return f'{minutes}m ago'
    hours = minutes // 60
    if hours < 48:
        return f'{hours}h ago'
    return f'{hours // 24}d ago'


def _event_iso_ts(payload: dict[str, Any]) -> str:
    raw = payload.get('source_timestamp') or payload.get('timestamp') or payload.get('observed_at')
    parsed = _parse_timestamp(str(raw) if raw is not None else None)
    if parsed is None:
        return datetime.now(timezone.utc).isoformat()
    return parsed.isoformat()


def _loads_json(value: str | bytes | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


__all__ = [
    'collect_status_checks',
    'freshness_pill',
    'last_updated_caption',
    'operator_action_buttons',
    'render_global_status_banner',
]
