import pandas as pd
import streamlit as st

from dashboard import data

st.title('Model Registry')

active = data.get_active_model()
if active:
    st.subheader('Active Model (Champion)')
    c1, c2, c3 = st.columns(3)
    c1.metric('Model Name', active.get('name', 'N/A'))
    c2.metric('Version', active.get('version', 'N/A'))
    c3.metric('Activated At', active.get('activated_at', 'N/A'))
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
