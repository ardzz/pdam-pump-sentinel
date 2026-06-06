from __future__ import annotations

import asyncio
import os
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[reportMissingImports]
from routemq.queue import dispatch  # type: ignore[reportMissingImports]

from app.jobs.drift_report_job import DriftReportJob
from app.jobs.retraining_job import RetrainingJob

RETRAIN_JOB_ID = 'pumpad-retrain'
DRIFT_JOB_ID = 'pumpad-drift'


class RetrainScheduler:
    def __init__(self, interval_minutes: int | None = None):
        if interval_minutes is None:
            interval_minutes = int(os.getenv('RETRAIN_INTERVAL_MINUTES', '30'))
        self.interval_minutes = interval_minutes
        self._scheduler: Any | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._scheduler is not None:
            return
        scheduler = AsyncIOScheduler(event_loop=loop, timezone='UTC')
        scheduler.add_job(
            self._dispatch_retraining,
            trigger='interval',
            minutes=self.interval_minutes,
            id=RETRAIN_JOB_ID,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        scheduler.start()
        self._scheduler = scheduler

    async def _dispatch_retraining(self) -> None:
        await dispatch(RetrainingJob())

    def shutdown(self) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None


class DriftScheduler:
    def __init__(self, interval_minutes: int | None = None):
        if interval_minutes is None:
            interval_minutes = int(os.getenv('DRIFT_INTERVAL_MINUTES', '15'))
        self.interval_minutes = interval_minutes
        self._scheduler: Any | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._scheduler is not None:
            return
        scheduler = AsyncIOScheduler(event_loop=loop, timezone='UTC')
        scheduler.add_job(
            self._dispatch_drift_report,
            trigger='interval',
            minutes=self.interval_minutes,
            id=DRIFT_JOB_ID,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        scheduler.start()
        self._scheduler = scheduler

    async def _dispatch_drift_report(self) -> None:
        await dispatch(DriftReportJob())

    def shutdown(self) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None


__all__ = ['RETRAIN_JOB_ID', 'DRIFT_JOB_ID', 'RetrainScheduler', 'DriftScheduler']
