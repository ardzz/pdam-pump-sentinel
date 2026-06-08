import json
from types import SimpleNamespace

import numpy as np

from app.controllers.anomaly_controller import Controller
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
