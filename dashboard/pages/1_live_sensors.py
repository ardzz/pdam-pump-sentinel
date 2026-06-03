import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data

st.title('Live Sensor Monitoring')

stations = data.list_stations()
station = st.selectbox('Select Pump Station', options=stations)

if not station:
    st.warning('No pump stations found.')
    st.stop()

col1, col2, col3, col4 = st.columns(4)

reading = data.get_latest_reading(station)
anomaly = data.get_latest_anomaly(station)

if reading:
    with col1:
        st.metric('Current (A)', reading.get('Current', 0))
    with col2:
        st.metric('Voltage (V)', reading.get('Voltage', 0))
    with col3:
        st.metric('Pressure (bar)', reading.get('Pressure', 0))
    with col4:
        st.metric('Temperature (°C)', reading.get('Temperature', 0))
else:
    st.info('Waiting for sensor data...')

st.divider()

if anomaly:
    status_color = 'red' if anomaly.get('anomaly', 0) == 1 else 'green'
    status_text = 'ANOMALY DETECTED' if anomaly.get('anomaly', 0) == 1 else 'NORMAL'

    st.subheader('Anomaly Status')
    st.markdown(f'### :{status_color}[{status_text}]')

    ac1, ac2 = st.columns(2)
    with ac1:
        st.metric('Anomaly Score', round(anomaly.get('score', 0), 4))
    with ac2:
        st.metric('Model Version', anomaly.get('model_version', 'N/A'))
else:
    st.info('No anomaly data available yet.')

history = data.get_anomaly_history(station, limit=50)
if history:
    st.subheader('Recent Anomaly Score Trend')
    df_hist = pd.DataFrame(history)
    df_hist['observed_at'] = pd.to_datetime(df_hist['observed_at'])

    score_data = df_hist[df_hist['measurement'] == 'anomaly_score']
    if not score_data.empty:
        fig = px.line(score_data, x='observed_at', y='value_float',
                     title='Anomaly Score (Last 50)',
                     labels={'value_float': 'Score', 'observed_at': 'Time'})
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Historical score trend data not found.')
