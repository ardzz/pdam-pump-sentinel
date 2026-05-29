from importlib import import_module


def test_build_anomaly_topic_uses_station_specific_contract():
    anomaly_event = import_module('app.models.anomaly_event')

    assert anomaly_event.build_anomaly_topic('ipa_01') == 'factory/skab/ipa_01/anomaly'
