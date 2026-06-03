import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data

st.title('Anomaly History')

stations = data.list_stations()
station = st.selectbox('Select Pump Station', options=stations)

if not station:
    st.warning('No pump stations found.')
    st.stop()

limit = st.slider('Number of records to fetch', 100, 1000, 500)
history = data.get_anomaly_history(station, limit=limit)

if not history:
    st.info('No historical data found for this station.')
    st.stop()

df = pd.DataFrame(history)
df['observed_at'] = pd.to_datetime(df['observed_at'])

st.subheader('Anomaly Score Timeline')
score_df = df[df['measurement'] == 'anomaly_score']
if not score_df.empty:
    fig = px.scatter(score_df, x='observed_at', y='value_float',
                     color='value_int',
                     title='Anomaly Scores over Time',
                     labels={'value_float': 'Score', 'observed_at': 'Time', 'value_int': 'Is Anomaly'})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info('No anomaly score measurements found.')

st.subheader('Raw Telemetry Logs')
st.dataframe(df.sort_values('observed_at', ascending=False), use_container_width=True)
