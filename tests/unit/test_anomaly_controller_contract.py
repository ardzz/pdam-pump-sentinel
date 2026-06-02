import json
from types import SimpleNamespace

import numpy as np

from app.controllers.anomaly_controller import Controller
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
    finally:
        reset_inference_service()


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
