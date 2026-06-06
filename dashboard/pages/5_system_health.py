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


st_autorefresh(interval=10 * 1000, key='system-health-refresh')

st.title('System Health')
stations = data.list_stations()
station = stations[0] if stations else None
widgets.render_global_status_banner(station=station)
st.caption('Service reachability checks refresh every 10 seconds.')

checked = _now_iso()
checks = widgets.collect_status_checks(station)
check_items = list(checks.items())
for start in range(0, len(check_items), 3):
    for column, (service_name, (service_ok, service_detail)) in zip(st.columns(3), check_items[start : start + 3], strict=False):
        with column:
            _service_card(service_name, service_ok, service_detail, checked)
