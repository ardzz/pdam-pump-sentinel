from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

try:
    from prometheus_client import Counter, Gauge, Histogram  # type: ignore[reportMissingImports]
except ModuleNotFoundError:
    from ._prometheus_fallback import Counter, Gauge, Histogram  # type: ignore[reportMissingImports]

logger = logging.getLogger('PDAM.observability')

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


def _label_value(value: Any) -> str:
    return '' if value is None else str(value)
