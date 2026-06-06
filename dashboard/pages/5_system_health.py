from __future__ import annotations

import importlib
import socket
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

import streamlit as st
from streamlit_autorefresh import st_autorefresh


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _http_probe(url: str) -> tuple[bool, str]:
    try:
        request = Request(url, method='GET')
        with urlopen(request, timeout=0.5) as response:
            return 200 <= response.status < 500, f'HTTP {response.status}'
    except (OSError, URLError) as exc:
        return False, str(exc)


def _tcp_probe(host: str, port: int) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True, f'TCP {host}:{port} reachable'
    except OSError as exc:
        return False, str(exc)


def _redis_manager_ping() -> tuple[bool, str] | None:
    for module_name in ('app.redis_manager', 'app.core.redis_manager', 'bootstrap.redis_manager'):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        manager = getattr(module, 'redis_manager', None)
        if manager is None:
            continue

        try:
            ping = getattr(manager, 'ping', None)
            if callable(ping):
                return bool(ping()), 'redis_manager ping'

            client = getattr(manager, 'client', None)
            client_ping = getattr(client, 'ping', None)
            if callable(client_ping):
                return bool(client_ping()), 'redis_manager client ping'
        except Exception as exc:
            return False, str(exc)

    return None


def _redis_probe() -> tuple[bool, str]:
    manager_result = _redis_manager_ping()
    if manager_result is not None:
        return manager_result
    return _tcp_probe('localhost', 6379)


def _service_card(name: str, ok: bool, detail: str, checked_at: str) -> None:
    color = 'green' if ok else 'red'
    status = 'ONLINE' if ok else 'OFFLINE'
    with st.container(border=True):
        st.markdown(f'### :{color}[{name}]')
        st.metric('Status', status)
        st.caption(f'Last checked: {checked_at}')
        st.caption(detail)


def _checks() -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    mlflow_ok, mlflow_detail = _http_probe('http://localhost:5000/health')
    redis_ok, redis_detail = _redis_probe()
    clickhouse_ok, clickhouse_detail = _http_probe('http://localhost:18123/ping')
    mqtt_ok, mqtt_detail = _tcp_probe('localhost', 11883)

    checks.append(('MLflow', mlflow_ok, mlflow_detail))
    checks.append(('Redis', redis_ok, redis_detail))
    checks.append(('ClickHouse', clickhouse_ok, clickhouse_detail))
    checks.append(('MQTT', mqtt_ok, mqtt_detail))
    return checks


st_autorefresh(interval=10 * 1000, key='system-health-refresh')

st.title('System Health')
st.caption('Service reachability checks refresh every 10 seconds.')

checked = _now_iso()
for column, (service_name, service_ok, service_detail) in zip(st.columns(4), _checks(), strict=False):
    with column:
        _service_card(service_name, service_ok, service_detail, checked)
