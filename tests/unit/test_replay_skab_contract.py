import json
from importlib import import_module

SENSOR_VALUES = {
    'Accelerometer1RMS': 0.10,
    'Accelerometer2RMS': 0.20,
    'Current': 1.10,
    'Pressure': 2.10,
    'Temperature': 30.10,
    'Thermocouple': 31.10,
    'Voltage': 220.10,
    'Volume Flow RateRMS': 10.10,
}
SAMPLE_RECORD = {
    'station': 'ipa_01',
    'timestamp': '2024-01-01T00:00:00Z',
    'sensors': SENSOR_VALUES,
    'labels': {'anomaly': 1, 'changepoint': 0},
}


def _replay_skab():
    return import_module('scripts.replay_skab')


def test_build_telemetry_topic_uses_station_specific_contract():
    replay_skab = _replay_skab()

    assert replay_skab.build_telemetry_topic('ipa_01') == 'factory/skab/ipa_01/telemetry'


def test_build_telemetry_payload_generates_json_safe_dry_run_payload_with_expected_schema():
    replay_skab = _replay_skab()

    payload = replay_skab.build_telemetry_payload(SAMPLE_RECORD, dry_run=True)

    assert payload == {
        'station': 'ipa_01',
        'timestamp': '2024-01-01T00:00:00Z',
        'sensors': SENSOR_VALUES,
        'labels': {'anomaly': 1, 'changepoint': 0},
        'dry_run': True,
    }
    json.dumps(payload)
