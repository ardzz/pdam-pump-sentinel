import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.controllers.anomaly_controller import Controller
from app.controllers.label_controller import LabelController
from app.observability.metrics import ANOMALY_EVENTS, INFERENCE_EVENTS
from app.services.inference import reset_inference_service, set_inference_service
from ml.inference.pca_inference import PcaAnomalyInferenceService
from ml.training.pca_detector import PcaT2QDetector


class FakePahoClient:
    def __init__(self):
        self.published = []

    def publish(self, topic, payload, qos=0):
        self.published.append({'topic': topic, 'payload': payload, 'qos': qos})
        return SimpleNamespace(rc=0)


def _service(window_size: int = 1, sensor_count: int = 2) -> PcaAnomalyInferenceService:
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 1.0, size=(200, window_size * sensor_count))
    detector = PcaT2QDetector(n_components=2, threshold_quantile=0.95, scaler='standard').fit(normal)
    return PcaAnomalyInferenceService(detector, ['a', 'b'], window_size=window_size, model_version='test')


async def test_controller_publishes_real_inference_payload():
    INFERENCE_EVENTS.clear()
    ANOMALY_EVENTS.clear()
    try:
        set_inference_service(_service(window_size=1))
        client = FakePahoClient()
        payload = {'timestamp': 't0', 'sensors': {'a': 0.0, 'b': 0.0}, 'labels': {'anomaly': 0}}

        result = await Controller.ingest(station='ipa_01', payload=payload, client=client)

        assert result == {'accepted': True, 'station': 'ipa_01'}
        assert len(client.published) == 1
        message = client.published[0]
        assert message['topic'] == 'factory/skab/ipa_01/anomaly'
        assert message['qos'] == 1
        body = json.loads(message['payload'])
        assert body['station'] == 'ipa_01'
        assert body['status'] == 'ok'
        assert body['model_version'] == 'test'
        assert isinstance(body['t2'], float) and isinstance(body['q'], float)
        assert body['anomaly'] in (0, 1)
        assert INFERENCE_EVENTS.labels(station='ipa_01', model_version='test', result='success')._value.get() == 1
    finally:
        reset_inference_service()
        INFERENCE_EVENTS.clear()
        ANOMALY_EVENTS.clear()


async def test_controller_records_inference_error_metric():
    class FailingService:
        def observe(self, station, timestamp, sensors):
            raise RuntimeError('boom')

    INFERENCE_EVENTS.clear()
    try:
        set_inference_service(FailingService())
        client = FakePahoClient()
        payload = {'timestamp': 't0', 'sensors': {'a': 0.0, 'b': 0.0}}

        try:
            await Controller.ingest(station='ipa_01', payload=payload, client=client)
        except RuntimeError:
            pass
        else:
            raise AssertionError('expected inference failure')

        assert INFERENCE_EVENTS.labels(station='ipa_01', model_version='', result='error')._value.get() == 1
    finally:
        reset_inference_service()
        INFERENCE_EVENTS.clear()


async def test_controller_records_anomaly_event_severity():
    class AnomalyService:
        def observe(self, station, timestamp, sensors):
            return SimpleNamespace(
                station=station,
                timestamp=timestamp,
                model_version='v-anomaly',
                window_size=1,
                window_filled=True,
                t2=1.0,
                q=0.2,
                t2_threshold=0.5,
                q_threshold=0.1,
                score=1.2,
                anomaly=1,
                top_contributing_sensor='a',
            )

    INFERENCE_EVENTS.clear()
    ANOMALY_EVENTS.clear()
    try:
        set_inference_service(AnomalyService())
        client = FakePahoClient()
        payload = {'timestamp': 't0', 'sensors': {'a': 10.0, 'b': 0.0}}

        await Controller.ingest(station='ipa_01', payload=payload, client=client)

        assert ANOMALY_EVENTS.labels(station='ipa_01', severity='high', model_version='v-anomaly')._value.get() == 1
    finally:
        reset_inference_service()
        INFERENCE_EVENTS.clear()
        ANOMALY_EVENTS.clear()


async def test_controller_falls_back_to_demo_without_model():
    try:
        set_inference_service(None)
        client = FakePahoClient()
        payload = {'timestamp': 't0', 'sensors': {'a': 0.0, 'b': 0.0}, 'labels': {'anomaly': 1}}

        await Controller.ingest(station='ipa_01', payload=payload, client=client)

        body = json.loads(client.published[0]['payload'])
        assert body == {'station': 'ipa_01', 'anomaly': 1, 'source_timestamp': 't0'}
    finally:
        reset_inference_service()


async def test_label_controller_persists_valid_label_and_returns_deterministic_response(monkeypatch):
    persisted = []

    async def fake_persist(station, label_payload):
        persisted.append({'station': station, 'label_payload': label_payload})

    monkeypatch.setattr('app.controllers.label_controller.persist_operator_label', fake_persist, raising=True)

    result = await LabelController.ingest(
        station='ipa_01',
        payload={
            'source_timestamp': '2024-01-01T00:00:00Z',
            'label': True,
            'operator_id': 'operator-a',
            'reason': 'visual confirmation',
        },
        client=FakePahoClient(),
    )

    assert result == {'accepted': True, 'station': 'ipa_01', 'label': 1}
    assert len(persisted) == 1
    assert persisted[0]['station'] == 'ipa_01'
    label_payload = persisted[0]['label_payload']
    assert label_payload.source_timestamp == '2024-01-01T00:00:00Z'
    assert label_payload.label == 1
    assert label_payload.label_source == 'operator'
    assert label_payload.operator_id == 'operator-a'
    assert label_payload.reason == 'visual confirmation'


@pytest.mark.parametrize(
    'payload',
    [
        {'label': 1},
        {'source_timestamp': 't0'},
        {'source_timestamp': 't0', 'label': 2},
        {'source_timestamp': 't0', 'label': '1'},
        {'source_timestamp': 't0', 'label': 1, 'label_source': ''},
        {'source_timestamp': 't0', 'label': 1, 'operator_id': 7},
    ],
)
async def test_label_controller_rejects_invalid_payload_before_persistence(monkeypatch, payload):
    persisted = []

    async def fake_persist(station, label_payload):
        persisted.append({'station': station, 'label_payload': label_payload})

    monkeypatch.setattr('app.controllers.label_controller.persist_operator_label', fake_persist, raising=True)

    with pytest.raises(ValueError):
        await LabelController.ingest(station='ipa_01', payload=payload, client=FakePahoClient())

    assert persisted == []
