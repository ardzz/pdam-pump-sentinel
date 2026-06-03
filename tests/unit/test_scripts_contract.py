from importlib import import_module
from types import SimpleNamespace

import pandas as pd
import pytest


def _inject_drift():
    return import_module('scripts.inject_drift')


def _seed_initial_models():
    return import_module('scripts.seed_initial_models')


def _trigger_retrain():
    return import_module('scripts.trigger_retrain')


def test_apply_drift_shifts_target_mean_and_preserves_frame_contract():
    inject_drift = _inject_drift()
    frame = pd.DataFrame(
        {
            'Pressure': [1.0, 2.0, 3.0],
            'Temperature': [30.0, 31.0, 32.0],
            'anomaly': [0, 1, 0],
        }
    )

    shifted = inject_drift.apply_drift(frame, 'Pressure', 2.5)

    assert shifted is not frame
    assert shifted.shape == frame.shape
    assert list(shifted.columns) == list(frame.columns)
    assert shifted['Pressure'].mean() == pytest.approx(frame['Pressure'].mean() + 2.5)
    pd.testing.assert_series_equal(shifted['Temperature'], frame['Temperature'])
    pd.testing.assert_series_equal(shifted['anomaly'], frame['anomaly'])


def test_apply_drift_rejects_missing_column():
    inject_drift = _inject_drift()

    with pytest.raises(ValueError, match='missing'):
        inject_drift.apply_drift(pd.DataFrame({'Pressure': [1.0]}), 'Current', 1.0)


def test_inject_drift_main_round_trips_skab_delimited_csv(tmp_path):
    inject_drift = _inject_drift()
    input_path = tmp_path / 'input.csv'
    output_path = tmp_path / 'output.csv'
    frame = pd.DataFrame(
        {
            'datetime': ['2024-01-01T00:00:00Z', '2024-01-01T00:00:01Z'],
            'Pressure': [2.0, 4.0],
            'Temperature': [30.0, 31.0],
            'anomaly': [0, 1],
            'changepoint': [0, 0],
        }
    )
    frame.to_csv(input_path, sep=inject_drift.SKAB_DELIMITER, index=False)

    result = inject_drift.main([
        '--input',
        str(input_path),
        '--output',
        str(output_path),
        '--column',
        'Pressure',
        '--delta',
        '5',
    ])

    output = pd.read_csv(output_path, sep=inject_drift.SKAB_DELIMITER)
    assert result == output_path
    assert output['Pressure'].tolist() == [7.0, 9.0]
    pd.testing.assert_series_equal(output['Temperature'], frame['Temperature'])
    assert output.shape == frame.shape
    assert list(output.columns) == list(frame.columns)


def test_trigger_retrain_dispatches_retraining_job(monkeypatch):
    trigger_retrain = _trigger_retrain()
    dispatched = []

    async def dispatch(job):
        dispatched.append(job)

    monkeypatch.setattr(trigger_retrain, 'dispatch', dispatch, raising=True)

    trigger_retrain.main([])

    assert len(dispatched) == 1
    assert isinstance(dispatched[0], trigger_retrain.RetrainingJob)


def test_seed_initial_models_builds_champion_registration_config(monkeypatch, tmp_path):
    seed_initial_models = _seed_initial_models()
    calls = []
    input_path = tmp_path / 'train.csv'
    output_dir = tmp_path / 'artifacts'
    input_path.write_text('datetime;Pressure\n2024-01-01T00:00:00Z;1.0\n')

    def train(config):
        calls.append(config)
        return SimpleNamespace(output_dir=output_dir, metrics={'f1': 1.0})

    monkeypatch.setattr(seed_initial_models, 'train_pca_from_skab', train, raising=True)

    result = seed_initial_models.main([
        '--input',
        str(input_path),
        '--output-dir',
        str(output_dir),
        '--window-size',
        '4',
        '--stride',
        '2',
        '--n-components',
        '0.9',
        '--threshold-quantile',
        '0.95',
    ])

    assert result.output_dir == output_dir
    assert len(calls) == 1
    config = calls[0]
    assert config.input_path == input_path
    assert config.output_dir == output_dir
    assert config.window_size == 4
    assert config.stride == 2
    assert config.log_mlflow is True
    assert config.register_model is True
    assert config.alias == 'champion'
    assert config.registered_model_name == 'PumpAD'


@pytest.mark.parametrize(
    'module_name',
    [
        'scripts.inject_drift',
        'scripts.seed_initial_models',
        'scripts.trigger_retrain',
    ],
)
def test_scripts_help_exits_zero(module_name):
    module = import_module(module_name)

    with pytest.raises(SystemExit) as exc:
        module.main(['--help'])

    assert exc.value.code == 0
