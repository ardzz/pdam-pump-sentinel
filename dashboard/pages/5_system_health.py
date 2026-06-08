from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from dashboard import data, widgets


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _service_card(name: str, ok: bool, detail: str, checked_at: str) -> None:
    color = 'green' if ok else 'red'
    status = 'ONLINE' if ok else 'OFFLINE'
    with st.container(border=True):
        st.markdown(f'### :{color}[{name}]')
        st.metric('Status', status)
        st.caption(f'Last checked: {checked_at}')
        st.caption(detail)


def _age_label(value: object) -> str:
    if value is None:
        return 'N/A'
    try:
        seconds = int(float(str(value)))
    except (TypeError, ValueError):
        return 'N/A'
    if seconds < 60:
        return f'{seconds}s'
    minutes = seconds // 60
    if minutes < 60:
        return f'{minutes}m'
    hours = minutes // 60
    if hours < 48:
        return f'{hours}h'
    return f'{hours // 24}d'


st_autorefresh(interval=10 * 1000, key='system-health-refresh')

st.title('System Health')
stations = data.list_stations()
station = stations[0] if stations else None
widgets.render_global_status_banner(station=station)
st.caption('Service reachability checks refresh every 10 seconds.')

checked = _now_iso()
if station:
    snapshot = data.get_observability_snapshot(station)
    st.subheader('App Metric Freshness')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Telemetry Age', _age_label(snapshot.get('telemetry_age_seconds')))
    c2.metric('Drift Report Age', _age_label(snapshot.get('drift_report_age_seconds')))
    c3.metric('Active Model Age', _age_label(snapshot.get('active_model_age_seconds')))
    c4.metric('Overall State', snapshot['state'])

checks = widgets.collect_status_checks(station)
st.subheader('Service Checks')
check_items = list(checks.items())
for start in range(0, len(check_items), 3):
    for column, (service_name, (service_ok, service_detail)) in zip(st.columns(3), check_items[start : start + 3], strict=False):
        with column:
            _service_card(service_name, service_ok, service_detail, checked)
