import streamlit as st

from dashboard import data


def _drift_detected(drift: dict) -> bool:
    metrics = drift.get('metrics', {}) if isinstance(drift.get('metrics'), dict) else {}
    return bool(drift.get('dataset_drift', metrics.get('drift_detected', False)))


def _drift_share(drift: dict) -> float:
    metrics = drift.get('metrics', {}) if isinstance(drift.get('metrics'), dict) else {}
    raw_value = drift.get('drift_share', metrics.get('drift_score', 0))
    try:
        value = float(raw_value or 0)
    except (TypeError, ValueError):
        value = 0
    return min(1.0, max(0.0, value))


def _age_label(seconds: float | None) -> str:
    if seconds is None:
        return 'N/A'
    value = int(seconds)
    if value < 60:
        return f'{value}s'
    minutes = value // 60
    if minutes < 60:
        return f'{minutes}m'
    hours = minutes // 60
    if hours < 48:
        return f'{hours}h'
    return f'{hours // 24}d'


def _duration_seconds(retrain: dict) -> float:
    raw = retrain.get('duration_seconds', retrain.get('duration_sec', 0))
    try:
        return float(raw or 0)
    except (TypeError, ValueError):
        return 0.0


st.title('Drift & Retrain Reports')

drift = data.get_drift_result()
retrain = data.get_retrain_result()

st.subheader('Model Drift Analysis')
if drift:
    drift_ts = drift.get('timestamp') or drift.get('finished_at') or drift.get('created_at')
    drift_age = data.timestamp_age_seconds(drift_ts)
    st.info(f"Last analyzed: {drift_ts or 'N/A'}")

    metrics = drift.get('metrics', {}) if isinstance(drift.get('metrics'), dict) else {}
    c1, c2, c3, c4 = st.columns(4)

    drift_detected = _drift_detected(drift)
    status_text = 'DRIFT DETECTED' if drift_detected else 'STABLE'
    share = _drift_share(drift)

    if drift_detected:
        st.error('DRIFT detected. Review feature drift before trusting fresh inference windows.')
    else:
        st.success('STABLE. Latest drift report is within the expected operating band.')

    c1.metric('Status', status_text, delta=None, delta_color='inverse' if drift_detected else 'normal')
    c2.metric('Drift Share', f'{share:.0%}')
    if drift.get('n_drifted') is not None or drift.get('n_features') is not None:
        c3.metric('Features Drifted', f"{drift.get('n_drifted', 'N/A')}/{drift.get('n_features', 'N/A')}")
    else:
        c3.metric('Method', drift.get('method', 'N/A'))
    c4.metric('Drift Report Age', _age_label(drift_age))

    if drift_age is None:
        st.warning('Drift report timestamp is missing; verify the drift job wrote metadata.')
    elif drift_age > 24 * 3600:
        st.warning('Drift report is stale. Run the drift job before making model decisions.')

    st.progress(share)
    st.caption(f'Drift share: {share:.2f}')

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
    rc3.metric('Duration (s)', round(_duration_seconds(retrain), 2))

    if drift and _drift_detected(drift):
        st.caption(f"Retrain linkage: drift report recommends reviewing candidate outcome `{retrain.get('reason', 'N/A')}`.")

    if not success and retrain.get('error'):
        st.error(f"Error: {retrain['error']}")

    with st.expander('Retrain Metadata'):
        st.json(retrain)
else:
    st.info('No retraining history available.')
