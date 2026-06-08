from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from routemq.job import Job  # type: ignore[reportMissingImports]
from routemq.queue import dispatch  # type: ignore[reportMissingImports]
from routemq.redis_manager import redis_manager  # type: ignore[reportMissingImports]

from app.jobs.retraining_job import RetrainingJob
from app.observability.annotations import post_annotation
from app.observability.metrics import DRIFT_DETECTED, DRIFT_REPORT_AGE, DRIFT_SHARE
from ml.datasets.skab_loader import SENSOR_COLUMNS, load_skab_csv
from ml.monitoring.drift_check import DriftResult, check_drift

logger = logging.getLogger(__name__)

DRIFT_RESULT_KEY = 'pumpad:drift:result'


@Job.register
class DriftReportJob(Job):
    queue = 'mlops'

    async def handle(self) -> None:
        reference, current = _load_drift_frames(self)
        result = check_drift(
            reference,
            current,
            SENSOR_COLUMNS,
            drift_share=float(os.getenv('DRIFT_SHARE_THRESHOLD', '0.5')),
        )
        payload = _drift_payload(result)
        DRIFT_SHARE.set(float(result.drift_share))
        DRIFT_DETECTED.set(1.0 if result.dataset_drift else 0.0)
        DRIFT_REPORT_AGE.set(0.0)
        if result.dataset_drift:
            post_annotation(
                text=f'Drift detected (share={result.drift_share:.3f}, n_drifted={result.n_drifted})',
                tags=['drift'],
            )

        logger.info(
            'pumpad drift report completed: dataset_drift=%s drift_share=%s',
            result.dataset_drift,
            result.drift_share,
        )
        await _write_redis_json(DRIFT_RESULT_KEY, payload)
        if result.dataset_drift:
            await dispatch(RetrainingJob())


def _load_drift_frames(job):
    reference_path = os.getenv('PUMPAD_DRIFT_REFERENCE_PATH')
    current_path = os.getenv('PUMPAD_DRIFT_CURRENT_PATH')
    if not reference_path or not current_path:
        raise ValueError('PUMPAD_DRIFT_REFERENCE_PATH and PUMPAD_DRIFT_CURRENT_PATH must be set')
    return load_skab_csv(Path(reference_path)), load_skab_csv(Path(current_path))


def _drift_payload(result: DriftResult) -> dict[str, Any]:
    payload = {
        'timestamp': datetime.now(UTC).isoformat(),
        'method': str(getattr(result, 'method', 'evidently') or 'evidently'),
        'threshold': float(getattr(result, 'threshold', os.getenv('DRIFT_SHARE_THRESHOLD', '0.5'))),
        'dataset_drift': bool(result.dataset_drift),
        'drift_share': float(result.drift_share),
        'n_drifted': int(result.n_drifted),
        'n_features': int(result.n_features),
    }
    report_path = getattr(result, 'report_path', None)
    if report_path:
        payload['report_path'] = str(report_path)
    return payload


async def _write_redis_json(key: str, value: Mapping[str, Any]) -> None:
    try:
        if redis_manager.is_enabled():
            await redis_manager.set_json(key, value)
    except Exception:
        logger.warning('could not write redis key %s', key, exc_info=True)
