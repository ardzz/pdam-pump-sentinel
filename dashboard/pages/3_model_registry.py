from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from dashboard import data


def _status_pill(text: str, tone: str) -> None:
    color = {'green': 'green', 'yellow': 'orange', 'red': 'red'}.get(tone, 'gray')
    st.markdown(f'### :{color}[{text}]')


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _relative_time(value: str | None) -> str:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return 'N/A'

    seconds = max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))
    if seconds < 60:
        return f'{seconds} seconds ago'
    minutes = seconds // 60
    if minutes < 60:
        return f'{minutes} minutes ago'
    hours = minutes // 60
    return f'{hours} hours ago'


st.title('Model Registry')

active = data.get_active_model()
if active:
    st.subheader('Active Model (Champion)')
    name = active.get('name') or active.get('registered_model_name') or 'N/A'
    version = active.get('version') or active.get('mlflow_version') or 'N/A'
    alias = active.get('alias') or 'champion'
    activated_at = active.get('activated_at')
    healthy = bool(version != 'N/A' and activated_at)

    _status_pill('Healthy active champion' if healthy else 'Unknown active champion', 'green' if healthy else 'yellow')
    st.markdown(f'### {name} :blue[{alias}]')

    c1, c2, c3 = st.columns(3)
    c1.metric('Model Name', name)
    c2.metric('Version', version)
    c3.metric('Activated At', _relative_time(activated_at))
    c3.caption(f'ISO: {activated_at or "N/A"}')
else:
    st.info('No active model metadata found in cache.')

st.divider()

st.subheader('Registered Versions')
versions = data.get_model_versions()
if versions:
    df_v = pd.DataFrame(versions)
    st.table(df_v)
else:
    st.info('No models found in registry.')
