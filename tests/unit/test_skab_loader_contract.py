from importlib import import_module
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parents[1] / 'fixtures'
SKAB_FIXTURE = FIXTURES_DIR / 'skab_tiny.csv'

SENSOR_COLUMNS = [
    'Accelerometer1RMS',
    'Accelerometer2RMS',
    'Current',
    'Pressure',
    'Temperature',
    'Thermocouple',
    'Voltage',
    'Volume Flow RateRMS',
]
EXPECTED_COLUMNS = ['datetime', *SENSOR_COLUMNS, 'anomaly', 'changepoint']


def _skab_loader():
    return import_module('ml.datasets.skab_loader')


def test_load_skab_csv_parses_semicolon_fixture_with_expected_columns():
    loader = _skab_loader()

    frame = loader.load_skab_csv(SKAB_FIXTURE)

    assert list(frame.columns) == EXPECTED_COLUMNS
    assert len(frame) == 3
    assert frame.loc[0, 'Accelerometer1RMS'] == pytest.approx(0.10)
    assert frame.loc[1, 'Current'] == pytest.approx(1.20)
    assert frame.loc[2, 'Volume Flow RateRMS'] == pytest.approx(10.30)
    assert frame[['anomaly', 'changepoint']].to_dict('records') == [
        {'anomaly': 0, 'changepoint': 0},
        {'anomaly': 1, 'changepoint': 0},
        {'anomaly': 0, 'changepoint': 1},
    ]


def test_load_skab_csv_defaults_optional_labels_to_zero_when_absent(tmp_path):
    loader = _skab_loader()
    rows = [
        ['datetime', *SENSOR_COLUMNS],
        [
            '2024-01-01T00:00:00Z',
            '0.10',
            '0.20',
            '1.10',
            '2.10',
            '30.10',
            '31.10',
            '220.10',
            '10.10',
        ],
        [
            '2024-01-01T00:00:01Z',
            '0.11',
            '0.21',
            '1.20',
            '2.20',
            '30.20',
            '31.20',
            '220.20',
            '10.20',
        ],
    ]
    no_label_fixture = tmp_path / 'skab_without_labels.csv'
    no_label_fixture.write_text('\n'.join(';'.join(row) for row in rows) + '\n')

    frame = loader.load_skab_csv(no_label_fixture)

    assert list(frame.columns) == EXPECTED_COLUMNS
    assert frame[['anomaly', 'changepoint']].to_dict('records') == [
        {'anomaly': 0, 'changepoint': 0},
        {'anomaly': 0, 'changepoint': 0},
    ]


def test_iter_telemetry_records_emits_station_scoped_sensor_and_label_records():
    loader = _skab_loader()
    frame = loader.load_skab_csv(SKAB_FIXTURE)

    records = list(loader.iter_telemetry_records(frame, station='ipa_01'))

    assert len(records) == 3
    assert records[0] == {
        'station': 'ipa_01',
        'timestamp': '2024-01-01T00:00:00Z',
        'sensors': {
            'Accelerometer1RMS': 0.10,
            'Accelerometer2RMS': 0.20,
            'Current': 1.10,
            'Pressure': 2.10,
            'Temperature': 30.10,
            'Thermocouple': 31.10,
            'Voltage': 220.10,
            'Volume Flow RateRMS': 10.10,
        },
        'labels': {'anomaly': 0, 'changepoint': 0},
    }
    assert records[1]['labels'] == {'anomaly': 1, 'changepoint': 0}
    assert records[2]['labels'] == {'anomaly': 0, 'changepoint': 1}
    assert set(records[0]['sensors']) == set(SENSOR_COLUMNS)
    assert 'anomaly' not in records[0]['sensors']
    assert 'changepoint' not in records[0]['sensors']
