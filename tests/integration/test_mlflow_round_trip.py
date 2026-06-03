from __future__ import annotations

from pathlib import Path

import pytest

from ml.datasets.skab_loader import SENSOR_COLUMNS
from ml.inference.pca_inference import PcaAnomalyInferenceService
from ml.registry.mlflow_client import load_champion_service
from ml.training.train_pca import PcaTrainingConfig, train_pca_from_skab


def test_mlflow_pca_champion_alias_round_trip_offline(monkeypatch, tmp_path):
    mlflow = pytest.importorskip('mlflow')
    from mlflow.tracking import MlflowClient

    tracking_uri = f'sqlite:///{tmp_path / "mlflow.db"}'
    monkeypatch.setenv('MLFLOW_TRACKING_URI', tracking_uri)
    monkeypatch.chdir(tmp_path)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment('round-trip', artifact_location=(tmp_path / 'mlflow-artifacts').as_uri())
    mlflow.set_experiment('round-trip')

    model_name = 'PumpADRoundTrip'
    fixture_path = Path(__file__).resolve().parents[1] / 'fixtures' / 'skab_tiny.csv'
    train_pca_from_skab(
        PcaTrainingConfig(
            input_path=fixture_path,
            output_dir=tmp_path / 'pca-output',
            window_size=1,
            stride=1,
            n_components=1,
            threshold_quantile=0.95,
            scaler='standard',
            log_mlflow=True,
            register_model=True,
            registered_model_name=model_name,
            alias='champion',
        )
    )

    client = MlflowClient()
    version = client.get_model_version_by_alias(model_name, 'champion')
    service = load_champion_service(model_name=model_name, alias='champion')

    assert str(version.version) == '1'
    assert service is not None
    assert isinstance(service, PcaAnomalyInferenceService)
    assert service.model_version == str(version.version)

    sensor_values = (0.10, 0.20, 1.10, 2.10, 30.10, 31.10, 220.10, 10.10)
    verdict = service.observe('station-a', '2024-01-01T00:00:00Z', dict(zip(SENSOR_COLUMNS, sensor_values, strict=True)))

    assert verdict.window_filled is True
    assert verdict.anomaly in (0, 1)
    assert verdict.score is not None
