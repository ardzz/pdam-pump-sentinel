from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from app.observability import metrics
from app.observability.metrics import (
    ACTIVE_MODEL_AGE,
    ANOMALY_EVENTS,
    ANOMALY_SCORE,
    DRIFT_DETECTED,
    DRIFT_REPORT_AGE,
    DRIFT_SHARE,
    INFERENCE_EVENTS,
    INFERENCE_LATENCY,
    MODEL_INFO,
    OBSERVABILITY_BUILD_INFO,
    OBSERVABILITY_SCHEMA_VERSION,
    PERSISTENCE_WRITES,
    RETRAIN_DURATION,
    RETRAINING_JOBS,
    TELEMETRY_FRESHNESS,
    render_prometheus_client_metrics,
    set_active_model_age,
    set_model_info,
)


class FakeRedisClient:
    def __init__(self, payloads: dict[str, dict[str, Any]]):
        self.payloads = payloads

    def get(self, key: str):
        payload = self.payloads.get(key)
        return None if payload is None else json.dumps(payload)

    def scan_iter(self, match: str, count: int = 20):
        prefix = match.rstrip('*')
        return [key for key in self.payloads if key.startswith(prefix)]


class FakeRedisModule:
    def __init__(self, payloads: dict[str, dict[str, Any]]):
        self.payloads = payloads

    def Redis(self, **_kwargs):
        return FakeRedisClient(self.payloads)


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

    assert _public_metric_name(INFERENCE_EVENTS) == 'pumpad_inference_events_total'
    assert tuple(INFERENCE_EVENTS._labelnames) == ('station', 'model_version', 'result')

    assert _public_metric_name(PERSISTENCE_WRITES) == 'pumpad_persistence_writes_total'
    assert tuple(PERSISTENCE_WRITES._labelnames) == ('target', 'result')

    assert _public_metric_name(ANOMALY_EVENTS) == 'pumpad_anomaly_events_total'
    assert tuple(ANOMALY_EVENTS._labelnames) == ('station', 'severity', 'model_version')

    assert _public_metric_name(DRIFT_REPORT_AGE) == 'pumpad_drift_report_age_seconds'
    assert tuple(DRIFT_REPORT_AGE._labelnames) == ()

    assert _public_metric_name(RETRAIN_DURATION) == 'pumpad_retrain_duration_seconds'
    assert tuple(RETRAIN_DURATION._labelnames) == ('result',)

    assert _public_metric_name(ACTIVE_MODEL_AGE) == 'pumpad_active_model_age_seconds'
    assert tuple(ACTIVE_MODEL_AGE._labelnames) == ('name', 'version', 'alias')

    assert _public_metric_name(OBSERVABILITY_BUILD_INFO) == 'pumpad_observability_build_info'
    assert tuple(OBSERVABILITY_BUILD_INFO._labelnames) == ('schema_version',)


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


def test_set_active_model_age_resets_to_latest_labelset() -> None:
    ACTIVE_MODEL_AGE.clear()
    try:
        set_active_model_age(
            {
                'name': 'PumpAD',
                'version': '1',
                'alias': 'champion',
                'activated_at': '2026-06-08T00:00:00+00:00',
            }
        )
        assert tuple(ACTIVE_MODEL_AGE._metrics) == (('PumpAD', '1', 'champion'),)

        set_active_model_age(
            {
                'name': 'PumpAD',
                'version': '2',
                'alias': 'champion',
                'activated_at': '2026-06-08T00:00:00+00:00',
            }
        )
        assert tuple(ACTIVE_MODEL_AGE._metrics) == (('PumpAD', '2', 'champion'),)
    finally:
        ACTIVE_MODEL_AGE.clear()


def test_render_prometheus_client_metrics_exposes_pumpad_families() -> None:
    MODEL_INFO.clear()
    set_model_info({'name': 'PumpAD', 'version': '6', 'alias': 'champion'})
    DRIFT_SHARE.set(0.09)
    DRIFT_DETECTED.set(1)
    TELEMETRY_FRESHNESS.labels(station='ipa_01').set(5)
    ANOMALY_SCORE.labels(station='ipa_01').observe(3.1)
    INFERENCE_LATENCY.labels(station='ipa_01', model_version='6').observe(0.05)
    RETRAINING_JOBS.labels(result='promoted').inc(0)
    INFERENCE_EVENTS.labels(station='ipa_01', model_version='6', result='success').inc(0)
    PERSISTENCE_WRITES.labels(target='redis', result='success').inc(0)
    ANOMALY_EVENTS.labels(station='ipa_01', severity='high', model_version='6').inc(0)
    DRIFT_REPORT_AGE.set(30)
    RETRAIN_DURATION.labels(result='promoted').observe(10)
    ACTIVE_MODEL_AGE.labels(name='PumpAD', version='6', alias='champion').set(60)

    text = render_prometheus_client_metrics().decode('utf-8')

    assert 'pumpad_model_info' in text
    assert 'pumpad_drift_share' in text
    assert 'pumpad_drift_detected' in text
    assert 'pumpad_telemetry_freshness_seconds' in text
    assert 'pumpad_anomaly_score_bucket' in text
    assert 'pumpad_inference_latency_seconds_bucket' in text
    assert 'pumpad_retraining_jobs_total' in text
    assert 'pumpad_inference_events_total' in text
    assert 'pumpad_persistence_writes_total' in text
    assert 'pumpad_anomaly_events_total' in text
    assert 'pumpad_drift_report_age_seconds' in text
    assert 'pumpad_retrain_duration_seconds_bucket' in text
    assert 'pumpad_active_model_age_seconds' in text
    assert 'pumpad_observability_build_info' in text
    assert OBSERVABILITY_SCHEMA_VERSION in text


def test_refresh_metrics_from_redis_state_uses_payload_evidence_without_fake_latency(monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    payloads = {
        'pumpad:drift:result': {
            'timestamp': (now - timedelta(seconds=90)).isoformat(),
            'dataset_drift': True,
            'drift_share': 0.75,
        },
        'pumpad:retrain:result': {
            'finished_at': now.isoformat(),
            'success': True,
            'promoted': True,
            'version': '2',
            'duration_seconds': 12.5,
            'reason': 'promoted',
        },
        'pumpad:latest:anomaly:ipa_01': {
            'timestamp': now.isoformat(),
            'source_timestamp': now.isoformat(),
            'score': 2.0,
            'model_version': '2',
        },
    }
    monkeypatch.setitem(sys.modules, 'redis', FakeRedisModule(payloads))
    monkeypatch.setattr(metrics, '_LAST_ANOMALY_SIGNATURE', None)
    monkeypatch.setattr(metrics, '_LAST_RETRAIN_SIGNATURE', None)
    INFERENCE_LATENCY.clear()
    RETRAIN_DURATION.clear()

    metrics.refresh_metrics_from_redis_state()
    text = render_prometheus_client_metrics().decode('utf-8')

    assert DRIFT_SHARE._value.get() == 0.75
    assert DRIFT_DETECTED._value.get() == 1.0
    assert DRIFT_REPORT_AGE._value.get() >= 0.0
    assert 'pumpad_retrain_duration_seconds_sum{result="promoted"} 12.5' in text
    assert ('ipa_01', '2') not in INFERENCE_LATENCY._metrics


def _public_metric_name(metric: Any) -> str:
    name = metric._name
    if metric._type == 'counter' and not name.endswith('_total'):
        return f'{name}_total'
    return name
