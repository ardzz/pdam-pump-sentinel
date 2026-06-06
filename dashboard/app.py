import streamlit as st

st.set_page_config(
    page_title='Pump Sentinel Dashboard',
    page_icon=':material/shield:',
    layout='wide',
    initial_sidebar_state='expanded',
)

pages = [
    st.Page('pages/0_overview.py', title='Overview', icon=':material/dashboard:', default=True),
    st.Page('pages/1_live_sensors.py', title='Live Sensors', icon=':material/sensors:'),
    st.Page('pages/2_anomaly_history.py', title='Anomaly History', icon=':material/history:'),
    st.Page('pages/3_model_registry.py', title='Model Registry', icon=':material/smart_toy:'),
    st.Page('pages/4_drift_reports.py', title='Drift & Training', icon=':material/trending_down:'),
    st.Page('pages/5_system_health.py', title='System Health', icon=':material/monitor_heart:'),
    st.Page('pages/6_runbook.py', title='Operator Runbook', icon=':material/menu_book:'),
]

pg = st.navigation(pages)
pg.run()
