from __future__ import annotations

from types import SimpleNamespace

from routemq.job import Job  # type: ignore[reportMissingImports]

from app.jobs import drift_report_job, retraining_job
from app.jobs.drift_report_job import DriftReportJob
from app.jobs.retraining_job import RetrainingJob
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
        )

    def from_artifacts(model_dir, model_version=None):
        inference_calls.append((model_dir, model_version))
        return sentinel

    def set_service(service):
        inference_calls.append(service)

    def promote_alias(model_name='PumpAD'):
        mlflow_calls.append(model_name)
        return '9'

    monkeypatch.setattr(retraining_job, 'train_pca_from_skab', train, raising=True)
    monkeypatch.setattr(retraining_job, '_read_champion_metrics', lambda: {'f1': 0.8, 'false_alarm_rate': 0.1})
    monkeypatch.setattr(retraining_job.PcaAnomalyInferenceService, 'from_artifacts', staticmethod(from_artifacts))
    monkeypatch.setattr(retraining_job, 'set_inference_service', set_service, raising=True)
    monkeypatch.setattr(retraining_job, 'redis_manager', fake_redis, raising=True)
    monkeypatch.setattr(retraining_job, '_promote_mlflow_champion_alias', promote_alias, raising=True)

    await RetrainingJob().handle()

    config = train_calls[0]
    assert config.register_model is True
    assert config.log_mlflow is True
    assert config.alias == 'challenger'
    assert config.registered_model_name == 'PumpAD'
    assert inference_calls == [(challenger_dir, None), sentinel]
    assert mlflow_calls == ['PumpAD']
    assert fake_redis.values['pumpad:active:model']['model_dir'] == str(challenger_dir)
    assert fake_redis.values['pumpad:active:model']['metrics'] == {'f1': 0.9, 'false_alarm_rate': 0.1}
    assert fake_redis.values['pumpad:retrain:result']['promoted'] is True
    assert fake_redis.values['pumpad:retrain:result']['metrics'] == {'f1': 0.9, 'false_alarm_rate': 0.1}


async def test_retraining_job_rejects_challenger(monkeypatch, tmp_path):
    fake_redis = FakeRedisManager(enabled=True)
    inference_calls = []
    mlflow_calls = []

    def train(config):
        return SimpleNamespace(
            metrics={'f1': 0.9, 'false_alarm_rate': 0.1},
            output_dir=tmp_path / 'challenger',
        )

    def from_artifacts(model_dir, model_version=None):
        inference_calls.append((model_dir, model_version))
        return object()

    def promote_alias(model_name='PumpAD'):
        mlflow_calls.append(model_name)
        return '9'

    monkeypatch.setattr(retraining_job, 'train_pca_from_skab', train, raising=True)
    monkeypatch.setattr(retraining_job, '_read_champion_metrics', lambda: {'f1': 0.95, 'false_alarm_rate': 0.1})
    monkeypatch.setattr(retraining_job.PcaAnomalyInferenceService, 'from_artifacts', staticmethod(from_artifacts))
    monkeypatch.setattr(retraining_job, 'set_inference_service', lambda service: inference_calls.append(service), raising=True)
    monkeypatch.setattr(retraining_job, 'redis_manager', fake_redis, raising=True)
    monkeypatch.setattr(retraining_job, '_promote_mlflow_champion_alias', promote_alias, raising=True)

    await RetrainingJob().handle()

    assert inference_calls == []
    assert mlflow_calls == []
    assert 'pumpad:active:model' not in fake_redis.values
    assert fake_redis.values['pumpad:retrain:result']['promoted'] is False
    assert 'must exceed' in fake_redis.values['pumpad:retrain:result']['reason']


async def test_drift_report_job_dispatches_retraining_on_drift(monkeypatch):
    fake_redis = FakeRedisManager(enabled=True)
    dispatched = []
    frames = (object(), object())

    def load_frames(job):
        return frames

    def check(reference, current, columns, drift_share=0.5):
        assert (reference, current) == frames
        return DriftResult(dataset_drift=True, drift_share=1.0, n_drifted=8, n_features=8)

    async def dispatch(job):
        dispatched.append(job)

    monkeypatch.setattr(drift_report_job, '_load_drift_frames', load_frames, raising=True)
    monkeypatch.setattr(drift_report_job, 'check_drift', check, raising=True)
    monkeypatch.setattr(drift_report_job, 'dispatch', dispatch, raising=True)
    monkeypatch.setattr(drift_report_job, 'redis_manager', fake_redis, raising=True)

    await DriftReportJob().handle()

    assert len(dispatched) == 1
    assert isinstance(dispatched[0], RetrainingJob)
    assert fake_redis.values['pumpad:drift:result'] == {
        'dataset_drift': True,
        'drift_share': 1.0,
        'n_drifted': 8,
        'n_features': 8,
    }


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


def test_mlops_jobs_allowlist_round_trip():
    assert isinstance(Job.unserialize(RetrainingJob().serialize()), RetrainingJob)
    assert isinstance(Job.unserialize(DriftReportJob().serialize()), DriftReportJob)
