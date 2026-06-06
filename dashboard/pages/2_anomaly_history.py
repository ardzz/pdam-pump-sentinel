from typing import cast

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

df = cast(pd.DataFrame, pd.DataFrame(history))
df['observed_at'] = pd.to_datetime(df['observed_at'])

st.subheader('Filters')
anomaly_only = st.toggle('Anomaly only', value=False)

min_time = df['observed_at'].min().to_pydatetime()
max_time = df['observed_at'].max().to_pydatetime()
if min_time < max_time:
    start_time, end_time = st.slider(
        'Time range',
        min_value=min_time,
        max_value=max_time,
        value=(min_time, max_time),
        format='YYYY-MM-DD HH:mm',
    )
    time_mask = cast(pd.Series, (df['observed_at'] >= pd.Timestamp(start_time)) & (df['observed_at'] <= pd.Timestamp(end_time)))
    df = cast(pd.DataFrame, df.loc[time_mask])
else:
    st.caption(f'Time range: {min_time.isoformat()}')

if anomaly_only:
    anomaly_mask = cast(pd.Series, df['value_int']).fillna(0).astype(int) == 1
    df = cast(pd.DataFrame, df.loc[anomaly_mask])

score_mask = cast(pd.Series, df['measurement']) == 'anomaly_score'
score_df = cast(pd.DataFrame, df.loc[score_mask])
score_anomaly_mask = cast(pd.Series, score_df['value_int']).fillna(0).astype(int) == 1
anomaly_count = int(score_anomaly_mask.sum()) if not score_df.empty else 0
score_count = len(score_df)
anomaly_rate = anomaly_count / score_count if score_count else 0
average_score = score_df['value_float'].mean() if not score_df.empty else None
max_score = score_df['value_float'].max() if not score_df.empty else None

m1, m2, m3, m4 = st.columns(4)
m1.metric('Total Anomalies', anomaly_count)
m2.metric('Anomaly Rate', f'{anomaly_rate:.1%}')
m3.metric('Average Score', f'{average_score:.4f}' if average_score is not None else 'N/A')
m4.metric('Max Score', f'{max_score:.4f}' if max_score is not None else 'N/A')

st.subheader('Anomaly Score Timeline')
if not score_df.empty:
    fig = px.scatter(
        score_df,
        x='observed_at',
        y='value_float',
        color='value_int',
        title='Anomaly Scores over Time',
        labels={'value_float': 'Score', 'observed_at': 'Time', 'value_int': 'Is Anomaly'},
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info('No anomaly score measurements found.')

st.subheader('Raw Telemetry Logs')
st.dataframe(df.sort_values('observed_at', ascending=False), use_container_width=True)
