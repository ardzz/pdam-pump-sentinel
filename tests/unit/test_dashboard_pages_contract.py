from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def mock_data(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr('dashboard.data.list_stations', lambda: ['test_station'])
    monkeypatch.setattr('dashboard.data.get_latest_reading', lambda s: {'Current': 1.2, 'Voltage': 220, 'Pressure': 3.5, 'Temperature': 45})
    monkeypatch.setattr('dashboard.data.get_latest_anomaly', lambda s: {'anomaly': 0, 'score': 0.01, 'model_version': 'v1'})
    monkeypatch.setattr('dashboard.data.get_anomaly_history', lambda s, limit=200: [
        {'observed_at': '2026-06-03 10:00:00', 'measurement': 'anomaly_score', 'value_float': 0.01, 'value_int': 0}
    ])
    monkeypatch.setattr('dashboard.data.get_active_model', lambda: {'name': 'PumpAD', 'version': '1', 'activated_at': '2026-01-01'})
    monkeypatch.setattr('dashboard.data.get_model_versions', lambda model_name='PumpAD': [{'version': '1', 'aliases': ['champion'], 'run_id': 'abc'}])
    monkeypatch.setattr('dashboard.data.get_drift_result', lambda: {'timestamp': '2026-06-03', 'metrics': {'drift_detected': False, 'drift_score': 0.1}, 'method': 'evidently'})
    monkeypatch.setattr('dashboard.data.get_retrain_result', lambda: {'finished_at': '2026-06-03', 'success': True, 'version': '2', 'duration_sec': 120})
    return mock

def test_live_sensors_page(mock_data):
    at = AppTest.from_file('dashboard/pages/1_live_sensors.py').run()
    assert not at.exception
    assert at.selectbox[0].value == 'test_station'
    assert any('Current (A)' in m.label for m in at.metric)

def test_anomaly_history_page(mock_data):
    at = AppTest.from_file('dashboard/pages/2_anomaly_history.py').run()
    assert not at.exception
    assert at.title[0].value == 'Anomaly History'
    assert len(at.dataframe) > 0

def test_model_registry_page(mock_data):
    at = AppTest.from_file('dashboard/pages/3_model_registry.py').run()
    assert not at.exception
    assert at.title[0].value == 'Model Registry'
    assert any('PumpAD' in m.value for m in at.metric)

def test_drift_reports_page(mock_data):
    at = AppTest.from_file('dashboard/pages/4_drift_reports.py').run()
    assert not at.exception
    assert at.title[0].value == 'Drift & Retrain Reports'
    assert any('STABLE' in m.value for m in at.metric)

def test_main_app_navigation(mock_data):
    at = AppTest.from_file('dashboard/app.py').run()
    assert not at.exception
