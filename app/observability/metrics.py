from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

try:
    from prometheus_client import (  # type: ignore[reportMissingImports]
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _PROMETHEUS_CLIENT_AVAILABLE = True
except ModuleNotFoundError:
    from ._prometheus_fallback import Counter, Gauge, Histogram  # type: ignore[reportMissingImports]

    REGISTRY = None  # type: ignore[assignment]
    generate_latest = None  # type: ignore[assignment]
    _PROMETHEUS_CLIENT_AVAILABLE = False

logger = logging.getLogger('PDAM.observability')

_LAST_ANOMALY_SIGNATURE: str | None = None
_LAST_RETRAIN_SIGNATURE: str | None = None
OBSERVABILITY_SCHEMA_VERSION = '2026-06-08.portfolio-observability.v1'

MODEL_INFO = Gauge(
    'pumpad_model_info',
    'Active champion model metadata',
    ['name', 'version', 'alias', 'model_dir', 'run_id'],
)
INFERENCE_LATENCY = Histogram(
    'pumpad_inference_latency_seconds',
    'Inference latency per telemetry observation',
    ['station', 'model_version'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
ANOMALY_SCORE = Histogram(
    'pumpad_anomaly_score',
    'Anomaly score distribution per station',
    ['station'],
    buckets=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 50.0),
)
DRIFT_SHARE = Gauge('pumpad_drift_share', 'Share of drifted features (0..1)')
DRIFT_DETECTED = Gauge('pumpad_drift_detected', 'Dataset drift boolean as 0/1')
TELEMETRY_FRESHNESS = Gauge(
    'pumpad_telemetry_freshness_seconds',
    'Seconds since last accepted observation per station',
    ['station'],
)
RETRAINING_JOBS = Counter(
    'pumpad_retraining_jobs_total',
    'Retraining job completion outcomes',
    ['result'],
)
INFERENCE_EVENTS = Counter(
    'pumpad_inference_events_total',
    'Inference event outcomes per station and model version',
    ['station', 'model_version', 'result'],
)
PERSISTENCE_WRITES = Counter(
    'pumpad_persistence_writes_total',
    'Persistence write outcomes for Redis and ClickHouse targets',
    ['target', 'result'],
)
ANOMALY_EVENTS = Counter(
    'pumpad_anomaly_events_total',
    'Anomaly events grouped by station, severity, and model version',
    ['station', 'severity', 'model_version'],
)
DRIFT_REPORT_AGE = Gauge('pumpad_drift_report_age_seconds', 'Seconds since the latest drift report')
RETRAIN_DURATION = Histogram(
    'pumpad_retrain_duration_seconds',
    'Retraining job duration by outcome',
    ['result'],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0),
)
ACTIVE_MODEL_AGE = Gauge(
    'pumpad_active_model_age_seconds',
    'Seconds since the active champion model was activated',
    ['name', 'version', 'alias'],
)
OBSERVABILITY_BUILD_INFO = Gauge(
    'pumpad_observability_build_info',
    'Observability schema compatibility marker',
    ['schema_version'],
)
OBSERVABILITY_BUILD_INFO.labels(schema_version=OBSERVABILITY_SCHEMA_VERSION).set(1)


def set_model_info(payload: Mapping[str, Any] | None) -> None:
    values = payload or {}
    labels = {
        'name': _label_value(values.get('name') or values.get('registered_model_name')),
        'version': _label_value(values.get('version') or values.get('mlflow_version')),
        'alias': _label_value(values.get('alias')),
        'model_dir': _label_value(values.get('model_dir')),
        'run_id': _label_value(values.get('run_id')),
    }
    try:
        MODEL_INFO.clear()
        MODEL_INFO.labels(**labels).set(1)
    except Exception:
        logger.warning('could not update model info metric', exc_info=True)


def set_active_model_age(payload: Mapping[str, Any] | None) -> None:
    values = payload or {}
    activated_at = _parse_timestamp(values.get('activated_at'))
    if activated_at is None:
        return
    labels = {
        'name': _label_value(values.get('name') or values.get('registered_model_name')),
        'version': _label_value(values.get('version') or values.get('mlflow_version')),
        'alias': _label_value(values.get('alias')),
    }
    age = max(0.0, (datetime.now(timezone.utc) - activated_at).total_seconds())
    try:
        ACTIVE_MODEL_AGE.clear()
        ACTIVE_MODEL_AGE.labels(**labels).set(age)
    except Exception:
        logger.warning('could not update active model age metric', exc_info=True)


def render_prometheus_client_metrics() -> bytes:
    if not _PROMETHEUS_CLIENT_AVAILABLE or REGISTRY is None or generate_latest is None:
        return b''
    refresh_metrics_from_redis_state()
    return generate_latest(REGISTRY)


def refresh_metrics_from_redis_state() -> None:
    try:
        import redis  # type: ignore[reportMissingImports]

        client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            decode_responses=True,
        )
        active_model = _redis_json(client, 'pumpad:active:model')
        if active_model:
            set_model_info(active_model)
            set_active_model_age(active_model)

        drift = _redis_json(client, 'pumpad:drift:result')
        if drift:
            DRIFT_SHARE.set(_float(drift.get('drift_share')))
            DRIFT_DETECTED.set(1.0 if drift.get('dataset_drift') else 0.0)
            _record_drift_report_age(drift)

        retrain = _redis_json(client, 'pumpad:retrain:result')
        if retrain:
            _record_retrain_metric(retrain)

        for key in client.scan_iter(match='pumpad:latest:anomaly:*', count=20):
            station = str(key).rsplit(':', 1)[-1]
            anomaly = _redis_json(client, str(key))
            if not anomaly:
                continue
            _record_anomaly_metrics(station, anomaly)
            _record_freshness(station, anomaly.get('source_timestamp') or anomaly.get('timestamp'))

        for key in client.scan_iter(match='pumpad:latest:reading:*', count=20):
            station = str(key).rsplit(':', 1)[-1]
            reading = _redis_json(client, str(key))
            if reading:
                _record_freshness(station, reading.get('timestamp') or reading.get('observed_at'))
    except Exception:
        logger.debug('could not refresh observability metrics from Redis state', exc_info=True)


def _redis_json(client: Any, key: str) -> dict[str, Any] | None:
    value = client.get(key)
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    try:
        payload = json.loads(value)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _record_anomaly_metrics(station: str, payload: Mapping[str, Any]) -> None:
    global _LAST_ANOMALY_SIGNATURE

    signature = f'{station}:{payload.get("source_timestamp") or payload.get("timestamp")}:{payload.get("score")}:{payload.get("model_version")}'
    if signature == _LAST_ANOMALY_SIGNATURE:
        return
    score = payload.get('score')
    if score is not None:
        ANOMALY_SCORE.labels(station=station).observe(_float(score))
    _LAST_ANOMALY_SIGNATURE = signature


def _record_drift_report_age(payload: Mapping[str, Any]) -> None:
    timestamp = payload.get('timestamp') or payload.get('finished_at') or payload.get('created_at')
    parsed = _parse_timestamp(timestamp)
    if parsed is None:
        return
    age = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())
    DRIFT_REPORT_AGE.set(age)


def _record_freshness(station: str, value: Any) -> None:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return
    age = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())
    TELEMETRY_FRESHNESS.labels(station=station).set(age)


def _record_retrain_metric(payload: Mapping[str, Any]) -> None:
    global _LAST_RETRAIN_SIGNATURE

    signature = f'{payload.get("finished_at")}:{payload.get("version")}:{payload.get("promoted")}:{payload.get("success")}:{payload.get("reason")}'
    if signature == _LAST_RETRAIN_SIGNATURE:
        return
    if payload.get('promoted'):
        result = 'promoted'
    elif payload.get('success') is False or payload.get('error'):
        result = 'error'
    else:
        result = 'rejected'
    RETRAINING_JOBS.labels(result=result).inc()
    duration_seconds = payload.get('duration_seconds')
    if duration_seconds is not None:
        RETRAIN_DURATION.labels(result=result).observe(max(0.0, _float(duration_seconds)))
    _LAST_RETRAIN_SIGNATURE = signature


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _label_value(value: Any) -> str:
    return '' if value is None else str(value)
