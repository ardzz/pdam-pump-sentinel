from __future__ import annotations

import asyncio

from app.jobs.retraining_job import RetrainingJob
from ml.monitoring import scheduler as scheduler_module
from ml.monitoring.scheduler import RETRAIN_JOB_ID, RetrainScheduler


class FakeScheduler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.jobs = []
        self.started = False
        self.shutdown_calls = []

    def add_job(self, func, **kwargs):
        self.jobs.append((func, kwargs))

    def start(self):
        self.started = True

    def shutdown(self, wait=True):
        self.shutdown_calls.append(wait)


def test_retrain_scheduler_registers_interval_job(monkeypatch):
    created = {}

    def factory(**kwargs):
        scheduler = FakeScheduler(**kwargs)
        created['scheduler'] = scheduler
        return scheduler

    monkeypatch.setattr(scheduler_module, 'AsyncIOScheduler', factory, raising=True)
    loop = asyncio.new_event_loop()
    try:
        retrain_scheduler = RetrainScheduler(interval_minutes=15)
        retrain_scheduler.start(loop)
        scheduler = created['scheduler']

        assert scheduler.kwargs.get('event_loop') is loop
        assert scheduler.started is True
        assert len(scheduler.jobs) == 1
        _, job_kwargs = scheduler.jobs[0]
        assert job_kwargs['trigger'] == 'interval'
        assert job_kwargs['minutes'] == 15
        assert job_kwargs['id'] == RETRAIN_JOB_ID
        assert job_kwargs['coalesce'] is True
        assert job_kwargs['max_instances'] == 1

        retrain_scheduler.start(loop)
        assert len(created) == 1

        retrain_scheduler.shutdown()
        assert scheduler.shutdown_calls == [False]
    finally:
        loop.close()


async def test_retrain_scheduler_dispatches_retraining_job(monkeypatch):
    dispatched = []

    async def fake_dispatch(job):
        dispatched.append(job)

    monkeypatch.setattr(scheduler_module, 'dispatch', fake_dispatch, raising=True)
    retrain_scheduler = RetrainScheduler()

    await retrain_scheduler._dispatch_retraining()

    assert len(dispatched) == 1
    assert isinstance(dispatched[0], RetrainingJob)
