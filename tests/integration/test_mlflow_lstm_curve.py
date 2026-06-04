from __future__ import annotations

from pathlib import Path

import pytest

from ml.training.train_lstm_ae import LstmAeTrainingConfig, train_lstm_ae_from_skab


def test_mlflow_lstm_logs_per_epoch_loss_history_offline(monkeypatch, tmp_path):
    mlflow = pytest.importorskip('mlflow')
    from mlflow.tracking import MlflowClient

    tracking_uri = f'sqlite:///{tmp_path / "mlflow.db"}'
    monkeypatch.setenv('MLFLOW_TRACKING_URI', tracking_uri)
    monkeypatch.chdir(tmp_path)

    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.active_run() is not None:
        mlflow.end_run()
    experiment_id = mlflow.create_experiment('lstm-curve', artifact_location=(tmp_path / 'mlflow-artifacts').as_uri())
    mlflow.set_experiment('lstm-curve')

    fixture_path = Path(__file__).resolve().parents[1] / 'fixtures' / 'skab_tiny.csv'
    train_lstm_ae_from_skab(
        LstmAeTrainingConfig(
            input_path=fixture_path,
            output_dir=tmp_path / 'lstm-output',
            validation_input_path=fixture_path,
            window_size=1,
            stride=1,
            threshold_quantile=0.95,
            log_mlflow=True,
            lstm_units=1,
            latent_dim=1,
            epochs=3,
            batch_size=1,
            patience=10,
            seed=123,
        )
    )

    client = MlflowClient()
    runs = client.search_runs([experiment_id])
    assert len(runs) == 1

    run_id = runs[0].info.run_id
    val_loss_history = client.get_metric_history(run_id, 'val_loss')
    loss_history = client.get_metric_history(run_id, 'loss')

    assert len(val_loss_history) == 3
    assert len(loss_history) == 3
    assert runs[0].data.metrics['f1'] >= 0.0
