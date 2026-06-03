
import streamlit as st

from dashboard import data

st.title('Drift & Retrain Reports')

drift = data.get_drift_result()
retrain = data.get_retrain_result()

st.subheader('Model Drift Analysis')
if drift:
    st.info(f"Last analyzed: {drift.get('timestamp', 'N/A')}")

    metrics = drift.get('metrics', {})
    c1, c2, c3 = st.columns(3)

    drift_detected = metrics.get('drift_detected', False)
    status_color = 'red' if drift_detected else 'green'
    status_text = 'DRIFT DETECTED' if drift_detected else 'STABLE'

    c1.metric('Status', status_text, delta=None, delta_color='inverse' if drift_detected else 'normal')
    c2.metric('Drift Score', round(metrics.get('drift_score', 0), 4))
    c3.metric('Method', drift.get('method', 'N/A'))

    if drift.get('report_path'):
        st.caption(f"Full report available at: {drift['report_path']}")

    with st.expander('Raw Drift Metadata'):
        st.json(drift)
else:
    st.warning('No drift analysis results found.')

st.divider()

st.subheader('Latest Retraining Job')
if retrain:
    st.info(f"Finished at: {retrain.get('finished_at', 'N/A')}")

    rc1, rc2, rc3 = st.columns(3)
    success = retrain.get('success', False)
    rc1.metric('Result', 'SUCCESS' if success else 'FAILED')
    rc2.metric('New Version', retrain.get('version', 'N/A'))
    rc3.metric('Duration (s)', round(retrain.get('duration_sec', 0), 2))

    if not success and retrain.get('error'):
        st.error(f"Error: {retrain['error']}")

    with st.expander('Retrain Metadata'):
        st.json(retrain)
else:
    st.info('No retraining history available.')
