from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from routemq.job import Job  # type: ignore[reportMissingImports]

from app.jobs import drift_report_job, retraining_job
from app.jobs.drift_report_job import DriftReportJob
from app.jobs.retraining_job import RetrainingJob
from app.observability.metrics import ACTIVE_MODEL_AGE, DRIFT_REPORT_AGE, RETRAIN_DURATION, RETRAINING_JOBS
from ml.monitoring.drift_check import DriftResult


class FakeRedisManager:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.values = {}

    def is_enabled(self):
        return self.enabled

    async def set_json(self, key, value, ex=None, px=None, nx=False, xx=False):
        self.values[key] = value
        return True


async def test_retraining_job_promotes_challenger(monkeypatch, tmp_path):
    challenger_dir = tmp_path / 'challenger'
    fake_redis = FakeRedisManager(enabled=True)
    sentinel = object()
    inference_calls = []
    mlflow_calls = []
    train_calls = []

    def train(config):
        train_calls.append(config)
        return SimpleNamespace(
            metrics={'f1': 0.9, 'false_alarm_rate': 0.1},
            output_dir=challenger_dir,
            run_id='run-9',
        )

    def load_service(model_dir):
        inference_calls.append(model_dir)
        return sentinel

    def set_service(service):
        inference_calls.append(service)

    def promote_alias(model_name='PumpAD'):
        mlflow_calls.append(model_name)
        return '9'

    monkeypatch.setattr(retraining_job, 'train_pca_from_skab', train, raising=True)
    monkeypatch.setattr(retraining_job, '_read_champion_metrics', lambda: {'f1': 0.8, 'false_alarm_rate': 0.1})
    monkeypatch.setattr(retraining_job, 'load_inference_service_from_artifacts', load_service, raising=True)
    monkeypatch.setattr(retraining_job, 'set_inference_service', set_service, raising=True)
    monkeypatch.setattr(retraining_job, 'redis_manager', fake_redis, raising=True)
    monkeypatch.setattr(retraining_job, '_promote_mlflow_champion_alias', promote_alias, raising=True)

    ACTIVE_MODEL_AGE.clear()
    RETRAIN_DURATION.clear()
    RETRAINING_JOBS.clear()
    try:
        await RetrainingJob().handle()

        config = train_calls[0]
        assert config.register_model is True
        assert config.log_mlflow is True
        assert config.alias == 'challenger'
        assert config.registered_model_name == 'PumpAD'
        assert inference_calls == [challenger_dir, sentinel]
        assert mlflow_calls == ['PumpAD']
        active_model = fake_redis.values['pumpad:active:model']
        assert active_model['registered_model_name'] == 'PumpAD'
        assert active_model['alias'] == 'champion'
        assert active_model['model_dir'] == str(challenger_dir)
        assert active_model['metrics'] == {'f1': 0.9, 'false_alarm_rate': 0.1}
        assert active_model['mlflow_version'] == '9'
        assert active_model['name'] == 'PumpAD'
        assert active_model['version'] == '9'
        activated_at = datetime.fromisoformat(active_model['activated_at'])
        assert activated_at.tzinfo is not None
        assert activated_at.utcoffset() == timezone.utc.utcoffset(None)
        assert fake_redis.values['pumpad:retrain:result']['promoted'] is True
        assert fake_redis.values['pumpad:retrain:result']['success'] is True
        assert fake_redis.values['pumpad:retrain:result']['version'] == '9'
        assert fake_redis.values['pumpad:retrain:result']['run_id'] == 'run-9'
        assert fake_redis.values['pumpad:retrain:result']['started_at']
        assert fake_redis.values['pumpad:retrain:result']['finished_at']
        assert fake_redis.values['pumpad:retrain:result']['duration_seconds'] >= 0
        assert fake_redis.values['pumpad:retrain:result']['metrics'] == {'f1': 0.9, 'false_alarm_rate': 0.1}
        assert RETRAINING_JOBS.labels(result='promoted')._value.get() == 1
        assert _metric_sample_value(
            RETRAIN_DURATION,
            'pumpad_retrain_duration_seconds_count',
            {'result': 'promoted'},
        ) == 1
        assert ACTIVE_MODEL_AGE.labels(name='PumpAD', version='9', alias='champion')._value.get() == 0.0
    finally:
        ACTIVE_MODEL_AGE.clear()
        RETRAIN_DURATION.clear()
        RETRAINING_JOBS.clear()


def test_active_model_version_uses_alias_label_without_mlflow_version():
    assert retraining_job._active_model_version(None, 'champion') == 'champion (local)'


async def test_retraining_job_rejects_challenger(monkeypatch, tmp_path):
    fake_redis = FakeRedisManager(enabled=True)
    inference_calls = []
    mlflow_calls = []

    def train(config):
        return SimpleNamespace(
            metrics={'f1': 0.9, 'false_alarm_rate': 0.1},
            output_dir=tmp_path / 'challenger',
        )

    def load_service(model_dir):
        inference_calls.append(model_dir)
        return object()

    def promote_alias(model_name='PumpAD'):
        mlflow_calls.append(model_name)
        return '9'

    monkeypatch.setattr(retraining_job, 'train_pca_from_skab', train, raising=True)
    monkeypatch.setattr(retraining_job, '_read_champion_metrics', lambda: {'f1': 0.95, 'false_alarm_rate': 0.1})
    monkeypatch.setattr(retraining_job, 'load_inference_service_from_artifacts', load_service, raising=True)
    monkeypatch.setattr(retraining_job, 'set_inference_service', lambda service: inference_calls.append(service), raising=True)
    monkeypatch.setattr(retraining_job, 'redis_manager', fake_redis, raising=True)
    monkeypatch.setattr(retraining_job, '_promote_mlflow_champion_alias', promote_alias, raising=True)

    RETRAIN_DURATION.clear()
    RETRAINING_JOBS.clear()
    try:
        await RetrainingJob().handle()

        assert inference_calls == []
        assert mlflow_calls == []
        assert 'pumpad:active:model' not in fake_redis.values
        assert fake_redis.values['pumpad:retrain:result']['promoted'] is False
        assert fake_redis.values['pumpad:retrain:result']['success'] is True
        assert fake_redis.values['pumpad:retrain:result']['version'] == 'challenger'
        assert fake_redis.values['pumpad:retrain:result']['duration_seconds'] >= 0
        assert 'must exceed' in fake_redis.values['pumpad:retrain:result']['reason']
        assert RETRAINING_JOBS.labels(result='rejected')._value.get() == 1
        assert _metric_sample_value(
            RETRAIN_DURATION,
            'pumpad_retrain_duration_seconds_count',
            {'result': 'rejected'},
        ) == 1
    finally:
        RETRAIN_DURATION.clear()
        RETRAINING_JOBS.clear()


async def test_drift_report_job_dispatches_retraining_on_drift(monkeypatch):
    fake_redis = FakeRedisManager(enabled=True)
    dispatched = []
    frames = (object(), object())

    def load_frames(job):
        return frames

    def check(reference, current, columns, drift_share=0.5):
        assert (reference, current) == frames
        return DriftResult(dataset_drift=True, drift_share=1.0, n_drifted=8, n_features=8, threshold=0.5)

    async def dispatch(job):
        dispatched.append(job)

    monkeypatch.setattr(drift_report_job, '_load_drift_frames', load_frames, raising=True)
    monkeypatch.setattr(drift_report_job, 'check_drift', check, raising=True)
    monkeypatch.setattr(drift_report_job, 'dispatch', dispatch, raising=True)
    monkeypatch.setattr(drift_report_job, 'redis_manager', fake_redis, raising=True)

    DRIFT_REPORT_AGE.set(99.0)

    await DriftReportJob().handle()

    assert len(dispatched) == 1
    assert isinstance(dispatched[0], RetrainingJob)
    assert fake_redis.values['pumpad:drift:result'] | {'timestamp': '<normalized>'} == {
        'timestamp': '<normalized>',
        'method': 'evidently',
        'threshold': 0.5,
        'dataset_drift': True,
        'drift_share': 1.0,
        'n_drifted': 8,
        'n_features': 8,
    }
    assert DRIFT_REPORT_AGE._value.get() == 0.0


async def test_drift_report_job_skips_dispatch_without_drift(monkeypatch):
    fake_redis = FakeRedisManager(enabled=True)
    dispatched = []

    monkeypatch.setattr(drift_report_job, '_load_drift_frames', lambda job: (object(), object()), raising=True)
    monkeypatch.setattr(
        drift_report_job,
        'check_drift',
        lambda reference, current, columns, drift_share=0.5: DriftResult(
            dataset_drift=False,
            drift_share=0.0,
            n_drifted=0,
            n_features=8,
        ),
        raising=True,
    )
    monkeypatch.setattr(drift_report_job, 'dispatch', lambda job: dispatched.append(job), raising=True)
    monkeypatch.setattr(drift_report_job, 'redis_manager', fake_redis, raising=True)

    await DriftReportJob().handle()

    assert dispatched == []
    assert fake_redis.values['pumpad:drift:result']['dataset_drift'] is False
    assert fake_redis.values['pumpad:drift:result']['method'] == 'evidently'
    assert fake_redis.values['pumpad:drift:result']['threshold'] == 0.5
    assert fake_redis.values['pumpad:drift:result']['timestamp']


async def test_retraining_job_records_error_payload(monkeypatch):
    fake_redis = FakeRedisManager(enabled=True)

    def train(config):
        raise RuntimeError('training failed')

    monkeypatch.setattr(retraining_job, 'train_pca_from_skab', train, raising=True)
    monkeypatch.setattr(retraining_job, 'redis_manager', fake_redis, raising=True)

    RETRAIN_DURATION.clear()
    RETRAINING_JOBS.clear()
    try:
        try:
            await RetrainingJob().handle()
        except RuntimeError:
            pass
        else:
            raise AssertionError('expected retraining failure')

        payload = fake_redis.values['pumpad:retrain:result']
        assert payload['success'] is False
        assert payload['promoted'] is False
        assert payload['reason'] == 'error'
        assert payload['version'] == 'unknown'
        assert payload['error'] == 'training failed'
        assert payload['duration_seconds'] >= 0
        assert RETRAINING_JOBS.labels(result='error')._value.get() == 1
    finally:
        RETRAIN_DURATION.clear()
        RETRAINING_JOBS.clear()


def test_mlops_jobs_allowlist_round_trip():
    assert isinstance(Job.unserialize(RetrainingJob().serialize()), RetrainingJob)
    assert isinstance(Job.unserialize(DriftReportJob().serialize()), DriftReportJob)


def _metric_sample_value(metric, sample_name, labels):
    for family in metric.collect():
        for sample in family.samples:
            if sample.name == sample_name and sample.labels == labels:
                return sample.value
    raise AssertionError(f'missing sample {sample_name} {labels}')
