from importlib import import_module
from types import SimpleNamespace

import pytest

TELEMETRY_TOPIC_PATTERN = 'factory/skab/{station}/telemetry'
LABEL_TOPIC_PATTERN = 'factory/skab/{station}/label'


class FakePahoClient:
    def __init__(self):
        self.published = []

    def publish(self, topic, payload, qos=0):
        self.published.append({'topic': topic, 'payload': payload, 'qos': qos})
        return SimpleNamespace(rc=0)


@pytest.mark.asyncio
async def test_routemq_handler_receives_path_params_payload_and_paho_client_for_publish():
    Router = import_module('routemq.router').Router

    class Controller:
        @staticmethod
        async def ingest(station, payload, client):
            client.publish(
                f'factory/skab/{station}/anomaly',
                {'station': station, 'anomaly': 1, 'source_timestamp': payload['timestamp']},
                qos=1,
            )
            return {'accepted': True, 'station': station}

    router = Router()
    router.on(TELEMETRY_TOPIC_PATTERN, Controller.ingest, qos=1)
    route = router.routes[0]
    params = {'station': 'ipa_01'}
    payload = {'timestamp': '2024-01-01T00:00:00Z', 'sensors': {'Pressure': 2.10}}
    client = FakePahoClient()

    result = await route.handler(**params, payload=payload, client=client)

    assert route.topic == TELEMETRY_TOPIC_PATTERN
    assert route.qos == 1
    assert result == {'accepted': True, 'station': 'ipa_01'}
    assert client.published == [
        {
            'topic': 'factory/skab/ipa_01/anomaly',
            'payload': {'station': 'ipa_01', 'anomaly': 1, 'source_timestamp': '2024-01-01T00:00:00Z'},
            'qos': 1,
        }
    ]


def test_repository_router_registers_telemetry_and_label_routes():
    telemetry_router = import_module('app.routers.telemetry').router

    routes = {route.topic: route for route in telemetry_router.routes}

    assert TELEMETRY_TOPIC_PATTERN in routes
    assert LABEL_TOPIC_PATTERN in routes
    assert routes[TELEMETRY_TOPIC_PATTERN].qos == 1
    assert routes[LABEL_TOPIC_PATTERN].qos == 1
    assert routes[TELEMETRY_TOPIC_PATTERN].handler.__name__ == 'ingest'
    assert routes[LABEL_TOPIC_PATTERN].handler.__name__ == 'ingest'
