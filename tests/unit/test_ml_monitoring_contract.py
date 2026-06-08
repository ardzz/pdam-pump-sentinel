from __future__ import annotations

import json
import sys
import types
from importlib import import_module
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from ml.datasets.skab_loader import SENSOR_COLUMNS
from ml.monitoring.champion_challenger import should_promote
from ml.monitoring.drift_check import check_drift
from ml.registry import mlflow_client
from ml.training.pca_detector import PcaT2QDetector


def test_check_drift_detects_shifted_numeric_column():
    rng = np.random.default_rng(0)
    columns = tuple(SENSOR_COLUMNS[:4])
    reference = pd.DataFrame(rng.normal(0.0, 1.0, size=(200, len(columns))), columns=pd.Index(columns))
    current = reference.copy()
    current[columns[0]] = current[columns[0]] + 6.0

    result = check_drift(reference, current, columns, drift_share=0.2)

    assert result.dataset_drift is True
    assert result.drift_share > 0.0
    assert result.n_drifted >= 1
    assert result.n_features == len(columns)
    assert result.method == 'evidently'
    assert result.threshold == 0.2
    assert result.report_path is None


def test_check_drift_reports_no_drift_for_identical_frame():
    rng = np.random.default_rng(1)
    columns = tuple(SENSOR_COLUMNS[:4])
    reference = pd.DataFrame(rng.normal(0.0, 1.0, size=(200, len(columns))), columns=pd.Index(columns))

    result = check_drift(reference, reference.copy(), columns, drift_share=0.2)

    assert result.dataset_drift is False
    assert result.drift_share == 0.0
    assert result.n_drifted == 0
    assert result.n_features == len(columns)
    assert result.method == 'evidently'
    assert result.threshold == 0.2


@pytest.mark.parametrize(
    ('champion', 'challenger', 'expected', 'reason_fragment'),
    [
        ({'f1': 0.8, 'false_alarm_rate': 0.1}, {'f1': 0.83, 'false_alarm_rate': 0.1}, True, 'beats f1'),
        ({'f1': 0.8, 'false_alarm_rate': 0.1}, {'f1': 0.81, 'false_alarm_rate': 0.1}, False, 'must exceed'),
        ({'f1': 0.8, 'false_alarm_rate': 0.1}, {'f1': 0.85, 'false_alarm_rate': 0.2}, False, 'exceeds guard'),
        ({'f1': None, 'false_alarm_rate': None}, {'f1': 0.7, 'false_alarm_rate': 0.2}, True, 'no incumbent champion'),
    ],
)
def test_should_promote_contract(champion, challenger, expected, reason_fragment):
    promoted, reason = should_promote(champion, challenger)

    assert promoted is expected
    assert reason_fragment in reason


def test_load_champion_service_resolves_mlflow_alias_offline(monkeypatch, tmp_path):
    sentinel = object()
    downloaded_dir = tmp_path / 'downloaded'
    downloaded_dir.mkdir()
    (downloaded_dir / 'metadata.json').write_text('{"model_family": "pca"}\n', encoding='utf-8')
    calls = _install_fake_mlflow_for_loading(monkeypatch, downloaded_dir)

    def from_artifacts(model_dir, model_version=None):
        calls['artifacts'].append((model_dir, model_version))
        return sentinel

    monkeypatch.setattr(mlflow_client.PcaAnomalyInferenceService, 'from_artifacts', staticmethod(from_artifacts))
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'file:///tmp/mlruns-contract')

    service = mlflow_client.load_champion_service()

    assert service is sentinel
    assert calls['tracking_uris'] == ['file:///tmp/mlruns-contract']
    assert calls['aliases'] == [('PumpAD', 'champion')]
    assert calls['downloads'] == [('run-123', '')]
    assert calls['artifacts'] == [(downloaded_dir, '12')]


def test_load_champion_service_uses_local_fallback_when_mlflow_unavailable(monkeypatch, tmp_path):
    _force_mlflow_unavailable(monkeypatch)
    artifact_dir = _write_local_artifacts(tmp_path / 'model')

    service = cast(Any, mlflow_client.load_champion_service(local_model_dir=str(artifact_dir)))

    assert service is not None
    assert service.sensor_columns == ('a', 'b')
    assert service.window_size == 1
    assert service.model_version == 'PumpADLocal'


def test_load_champion_service_returns_none_without_mlflow_or_local_artifacts(monkeypatch, tmp_path):
    _force_mlflow_unavailable(monkeypatch)

    service = mlflow_client.load_champion_service(local_model_dir=str(tmp_path / 'missing'))

    assert service is None


def _install_fake_mlflow_for_loading(monkeypatch, downloaded_dir):
    calls: dict[str, list[Any]] = {
        'tracking_uris': [],
        'aliases': [],
        'downloads': [],
        'artifacts': [],
    }
    mlflow = types.ModuleType('mlflow')
    mlflow.__path__ = []
    mlflow_any = cast(Any, mlflow)

    def set_tracking_uri(uri):
        calls['tracking_uris'].append(uri)

    mlflow_any.set_tracking_uri = set_tracking_uri

    artifacts = types.ModuleType('mlflow.artifacts')
    artifacts_any = cast(Any, artifacts)

    def download_artifacts(*, run_id, artifact_path):
        calls['downloads'].append((run_id, artifact_path))
        return str(downloaded_dir)

    artifacts_any.download_artifacts = download_artifacts
    tracking = types.ModuleType('mlflow.tracking')
    tracking_any = cast(Any, tracking)
    version = SimpleNamespace(source='runs:/run-123/pca_anomaly_model', version='12', run_id='run-123')

    class FakeMlflowClient:
        def get_model_version_by_alias(self, model_name, alias):
            calls['aliases'].append((model_name, alias))
            return version

    tracking_any.MlflowClient = FakeMlflowClient
    mlflow_any.artifacts = artifacts
    mlflow_any.tracking = tracking
    monkeypatch.setitem(sys.modules, 'mlflow', mlflow)
    monkeypatch.setitem(sys.modules, 'mlflow.artifacts', artifacts)
    monkeypatch.setitem(sys.modules, 'mlflow.tracking', tracking)
    return calls


def _force_mlflow_unavailable(monkeypatch):
    def unavailable_import(name: str):
        if name.startswith('mlflow'):
            raise ImportError(name)
        return import_module(name)

    monkeypatch.setattr(mlflow_client, 'import_module', unavailable_import)


def _write_local_artifacts(artifact_dir):
    artifact_dir.mkdir()
    columns = ['a', 'b']
    rng = np.random.default_rng(2)
    normal = rng.normal(0.0, 1.0, size=(64, len(columns)))
    detector = PcaT2QDetector(n_components=1, threshold_quantile=0.95, scaler='standard').fit(normal)
    joblib = import_module('joblib')
    joblib.dump(detector, artifact_dir / 'pca_detector.joblib')
    metadata = {'sensor_columns': columns, 'params': {'window_size': 1, 'registered_model_name': 'PumpADLocal'}}
    (artifact_dir / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    return artifact_dir
