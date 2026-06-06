from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard import data, widgets


def _station_from_state(stations: list[str]) -> str | None:
    if not stations:
        return None
    current = st.session_state.get('overview_station')
    if current in stations:
        return str(current)
    return stations[0]


def _model_name(active: dict[str, Any] | None) -> str:
    if not active:
        return 'N/A'
    return str(active.get('name') or active.get('registered_model_name') or 'N/A')


def _model_version(active: dict[str, Any] | None) -> str:
    if not active:
        return 'N/A'
    return str(active.get('version') or active.get('mlflow_version') or 'N/A')


def _drift_status(drift: dict[str, Any] | None) -> str:
    if not drift:
        return 'N/A'
    metrics = drift.get('metrics', {}) if isinstance(drift.get('metrics'), dict) else {}
    drift_detected = bool(drift.get('dataset_drift', metrics.get('drift_detected', False)))
    return 'DRIFT' if drift_detected else 'STABLE'


def _score(value: Any) -> str:
    try:
        return f'{float(value):.4f}'
    except (TypeError, ValueError):
        return 'N/A'


stations = data.list_stations()
station = _station_from_state(stations)

st.title('PDAM Pump Sentinel')
widgets.render_global_status_banner(station=station)
st.caption('Consolidated MLOps overview for pump anomaly detection and retraining operations.')

if not station:
    st.warning('No pump stations found.')
    st.stop()

active = data.get_active_model()
reading = data.get_latest_reading(station)
anomaly = data.get_latest_anomaly(station)
drift = data.get_drift_result()
retrain = data.get_retrain_result()

k1, k2, k3, k4 = st.columns(4)
with k1:
    version = _model_version(active)
    st.metric('Active Champion', f'{_model_name(active)} v{version}' if version != 'N/A' else _model_name(active))
with k2:
    st.metric('Last Anomaly', _score(anomaly.get('score') if anomaly else None))
    st.caption(f"Timestamp: {(anomaly or {}).get('source_timestamp', 'N/A')}")
with k3:
    st.metric('Drift Status', _drift_status(drift))
with k4:
    if retrain:
        retrain_status = 'SUCCESS' if retrain.get('success') else 'FAILED'
        st.metric('Last Retrain', retrain_status)
        st.caption(f"Version: {retrain.get('version', 'N/A')}")
    else:
        st.metric('Last Retrain', 'N/A')
        st.caption('Version: N/A')

st.divider()

selected_index = stations.index(station)
station = st.selectbox('Select Pump Station', options=stations, index=selected_index, key='overview_station')
reading = data.get_latest_reading(station)
st.caption(f"Last reading: {(reading or {}).get('timestamp', 'N/A')}")

st.subheader('What to do next')
next_cols = st.columns(5)
with next_cols[0]:
    st.markdown('**Live Sensors**')
    st.caption('Watch current telemetry, anomaly status, and the latest score trend.')
with next_cols[1]:
    st.markdown('**Anomaly History**')
    st.caption('Filter recent anomaly scores and inspect historical telemetry rows.')
with next_cols[2]:
    st.markdown('**Model Registry**')
    st.caption('Confirm the active champion, alias, and registered model versions.')
with next_cols[3]:
    st.markdown('**Drift & Training**')
    st.caption('Review drift status and the latest retraining outcome before action.')
with next_cols[4]:
    st.markdown('**System Health**')
    st.caption('Check MLflow, Redis, ClickHouse, and MQTT service reachability.')
