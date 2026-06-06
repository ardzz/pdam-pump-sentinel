from __future__ import annotations

from typing import Any

from app.observability.metrics import (
    ANOMALY_SCORE,
    DRIFT_DETECTED,
    DRIFT_SHARE,
    INFERENCE_LATENCY,
    MODEL_INFO,
    RETRAINING_JOBS,
    TELEMETRY_FRESHNESS,
    set_model_info,
)


def test_metrics_exist_with_expected_names_and_labels() -> None:
    assert _public_metric_name(MODEL_INFO) == 'pumpad_model_info'
    assert tuple(MODEL_INFO._labelnames) == ('name', 'version', 'alias', 'model_dir', 'run_id')

    assert _public_metric_name(INFERENCE_LATENCY) == 'pumpad_inference_latency_seconds'
    assert tuple(INFERENCE_LATENCY._labelnames) == ('station', 'model_version')

    assert _public_metric_name(ANOMALY_SCORE) == 'pumpad_anomaly_score'
    assert tuple(ANOMALY_SCORE._labelnames) == ('station',)

    assert _public_metric_name(DRIFT_SHARE) == 'pumpad_drift_share'
    assert tuple(DRIFT_SHARE._labelnames) == ()

    assert _public_metric_name(DRIFT_DETECTED) == 'pumpad_drift_detected'
    assert tuple(DRIFT_DETECTED._labelnames) == ()

    assert _public_metric_name(TELEMETRY_FRESHNESS) == 'pumpad_telemetry_freshness_seconds'
    assert tuple(TELEMETRY_FRESHNESS._labelnames) == ('station',)

    assert _public_metric_name(RETRAINING_JOBS) == 'pumpad_retraining_jobs_total'
    assert tuple(RETRAINING_JOBS._labelnames) == ('result',)


def test_set_model_info_resets_to_latest_labelset() -> None:
    MODEL_INFO.clear()
    try:
        set_model_info(
            {
                'name': 'PumpAD',
                'version': '1',
                'alias': 'champion',
                'model_dir': '/models/v1',
                'run_id': 'run-1',
            }
        )
        assert tuple(MODEL_INFO._metrics) == (('PumpAD', '1', 'champion', '/models/v1', 'run-1'),)

        set_model_info(
            {
                'name': 'PumpAD',
                'version': '2',
                'alias': 'champion',
                'model_dir': '/models/v2',
                'run_id': 'run-2',
            }
        )

        assert tuple(MODEL_INFO._metrics) == (('PumpAD', '2', 'champion', '/models/v2', 'run-2'),)
        assert MODEL_INFO.labels('PumpAD', '2', 'champion', '/models/v2', 'run-2')._value.get() == 1
    finally:
        MODEL_INFO.clear()


def test_retraining_jobs_accepts_expected_results() -> None:
    for result in ('promoted', 'rejected', 'error'):
        RETRAINING_JOBS.labels(result=result).inc(0)


def _public_metric_name(metric: Any) -> str:
    name = metric._name
    if metric._type == 'counter' and not name.endswith('_total'):
        return f'{name}_total'
    return name
