from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data, widgets


def _parse_dt(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_anomaly(row: dict) -> bool:
    try:
        return int(row.get('value_int') or 0) == 1
    except (TypeError, ValueError):
        return False


def _anomaly_count(rows: list[dict], start: datetime, end: datetime) -> int:
    total = 0
    for row in rows:
        observed_at = _parse_dt(row.get('observed_at'))
        if observed_at is not None and start <= observed_at < end and _is_anomaly(row):
            total += 1
    return total


def _window_metric(rows: list[dict], now: datetime, window: timedelta) -> tuple[int, int]:
    current = _anomaly_count(rows, now - window, now)
    previous = _anomaly_count(rows, now - (window * 2), now - window)
    return current, current - previous


def _severity(value: object) -> str:
    try:
        score = float(cast(Any, value) or 0)
    except (TypeError, ValueError):
        score = 0.0
    if score < 0.5:
        return 'low'
    if score < 1.0:
        return 'medium'
    return 'high'


def _iso_ts(value: object) -> str:
    parsed = _parse_dt(value)
    return parsed.isoformat() if parsed else str(value or 'unknown')


def _row_payload(row: pd.Series) -> dict:
    return {
        'observed_at': _iso_ts(row.get('observed_at')),
        'measurement': str(row.get('measurement', 'N/A')),
        'value_float': float(cast(Any, row.get('value_float')) or 0),
        'value_int': int(cast(Any, row.get('value_int')) or 0),
        'severity': str(row.get('severity', 'low')),
    }


def _redis_json(key: str) -> dict | None:
    try:
        value = data._redis_client().get(key)
    except Exception:
        return None
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None

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
df['severity'] = df['value_float'].apply(_severity)

now = datetime.now(timezone.utc)
last_1h, delta_1h = _window_metric(history, now, timedelta(hours=1))
last_24h, delta_24h = _window_metric(history, now, timedelta(hours=24))
last_7d, delta_7d = _window_metric(history, now, timedelta(days=7))
total_fetched = sum(1 for row in history if _is_anomaly(row))

k1, k2, k3, k4 = st.columns(4)
k1.metric('Last 1 hour', last_1h, delta=delta_1h)
k2.metric('Last 24 hours', last_24h, delta=delta_24h)
k3.metric('Last 7 days', last_7d, delta=delta_7d)
k4.metric('Total fetched', total_fetched)

severity_counts = cast(pd.Series, df.loc[cast(pd.Series, df['value_int']).fillna(0).astype(int) == 1, 'severity']).value_counts()
s1, s2, s3 = st.columns(3)
s1.metric('Low severity', int(cast(Any, severity_counts.get('low', 0)) or 0))
s2.metric('Medium severity', int(cast(Any, severity_counts.get('medium', 0)) or 0))
s3.metric('High severity', int(cast(Any, severity_counts.get('high', 0)) or 0))

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

m1, m2, m3 = st.columns(3)
m1.metric('Anomaly Rate', f'{anomaly_rate:.1%}')
m2.metric('Average Score', f'{average_score:.4f}' if average_score is not None else 'N/A')
m3.metric('Max Score', f'{max_score:.4f}' if max_score is not None else 'N/A')

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

st.subheader('Anomaly Drilldown')
drilldown_mask = cast(pd.Series, df['value_int']).fillna(0).astype(int) == 1
drilldown_df = cast(pd.DataFrame, df.loc[drilldown_mask]).sort_values('observed_at', ascending=False).head(20)
if drilldown_df.empty:
    st.info('No anomaly rows available for drilldown.')
else:
    options = list(drilldown_df.index)

    def _label(index: object) -> str:
        row = drilldown_df.loc[index]
        return f"{_iso_ts(row.get('observed_at'))} — score {float(row.get('value_float') or 0):.4f}"

    selected_index = st.selectbox('Select anomaly event', options=options, format_func=_label)
    selected_row = drilldown_df.loc[selected_index]
    selected_ts = _iso_ts(selected_row.get('observed_at'))
    latest_payload = data.get_latest_anomaly(station) if selected_index == options[0] else None
    anomaly_payload = latest_payload or _row_payload(selected_row)
    anomaly_payload.setdefault('source_timestamp', selected_ts)
    ack_payload = _redis_json(f'pumpad:anomaly:ack:{station}:{selected_ts}')

    with st.container(border=True):
        st.markdown('**Anomaly payload**')
        st.json(anomaly_payload)
        st.markdown('**Operator acknowledgement**')
        if ack_payload:
            st.json(ack_payload)
        else:
            st.caption('No acknowledgement recorded.')
        widgets.operator_action_buttons(anomaly_payload, station)

st.subheader('Raw Telemetry Logs')
st.dataframe(df.sort_values('observed_at', ascending=False), use_container_width=True)
