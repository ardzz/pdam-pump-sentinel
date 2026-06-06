from __future__ import annotations

import streamlit as st

from dashboard import data, widgets

st.title('Operator Runbook')
stations = data.list_stations()
widgets.render_global_status_banner(station=stations[0] if stations else None)

with st.expander('Anomaly storm'):
    st.markdown(
        '- Confirm the storm scope: station, first seen time, current anomaly score, and top contributing sensor.\n'
        '- Check Grafana telemetry panels for sensor spikes, missing data, replay jobs, and MQTT ingress lag.\n'
        '- Compare MLflow active champion version with the last known stable champion and recent deployments.\n'
        '- Acknowledge the current event in the dashboard, mute only if operators are already investigating.\n'
        '- Escalate to operations lead and ML owner if anomalies persist for more than 15 minutes.\n\n'
        '**Last reviewed:** 2026-06-06'
    )

with st.expander('Drift detected'):
    st.markdown(
        '- Open the Drift & Training page and verify the latest drift score, drifted features, and report path.\n'
        '- Check Grafana for sensor calibration changes, maintenance windows, and sustained operating regime shifts.\n'
        '- Check MLflow for the active model age and any recent candidate promotion or rollback.\n'
        '- Trigger retraining only after confirming the drift is operationally valid, not telemetry corruption.\n'
        '- Escalate to ML owner when drift remains high across two consecutive analysis windows.\n\n'
        '**Last reviewed:** 2026-06-06'
    )

with st.expander('Retrain failure'):
    st.markdown(
        '- Review the latest retrain result error, duration, candidate version, and promotion guardrail status.\n'
        '- Check MLflow experiment runs for failed artifacts, missing metrics, and registry write failures.\n'
        '- Check Grafana service panels for Redis, ClickHouse, and worker resource pressure during retraining.\n'
        '- Keep the current champion active unless model registry metadata is missing or corrupted.\n'
        '- Escalate to ML owner and platform owner if the next scheduled retry fails.\n\n'
        '**Last reviewed:** 2026-06-06'
    )

with st.expander('Service degraded'):
    st.markdown(
        '- Use the System Health page to identify whether MLflow, Redis, ClickHouse, MQTT, model freshness, or telemetry freshness failed.\n'
        '- Check Grafana infrastructure panels for container restarts, disk pressure, network errors, and queue backlog.\n'
        '- Confirm MLflow /health, ClickHouse /ping, Redis ping, and MQTT TCP reachability from the host.\n'
        '- If telemetry is stale, verify MQTT ingestion first, then ClickHouse writes, then dashboard cache state.\n'
        '- Escalate to platform owner when a core service remains RED for more than 5 minutes.\n\n'
        '**Last reviewed:** 2026-06-06'
    )
