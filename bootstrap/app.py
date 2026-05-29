from __future__ import annotations

import asyncio
import logging
from typing import Any

from dotenv import load_dotenv
from routemq.logging_config import configure_logging  # type: ignore[reportMissingImports]
from routemq.mqtt_utils import (  # type: ignore[reportMissingImports]
    connect_mqtt_client_with_retries,
    create_mqtt_client,
    get_main_client_id,
    get_mqtt_connection_config,
    parse_mqtt_payload,
)  # type: ignore[reportMissingImports]
from routemq.router import Router  # type: ignore[reportMissingImports]
from routemq.router_registry import create_dynamic_router  # type: ignore[reportMissingImports]
from routemq.worker_manager import WorkerManager  # type: ignore[reportMissingImports]


class Application:
    def __init__(
        self,
        router: Router | None = None,
        env_file: str = '.env',
        show_banner: bool = True,
        log_to_console: bool = True,
    ):
        self.env_file = env_file
        self.show_banner = show_banner
        load_dotenv(env_file)
        configure_logging(log_to_console=log_to_console)

        self.logger = logging.getLogger('PDAM.RouteMQ')
        self.router = router or create_dynamic_router('app.routers')
        self.client: Any | None = None
        self.worker_manager = WorkerManager(self.router, router_directory='app.routers')

    async def _initialize_connections(self) -> None:
        return None

    async def _cleanup_connections(self) -> None:
        return None

    def connect(self) -> None:
        config = get_mqtt_connection_config()
        self.client = create_mqtt_client(
            get_main_client_id(),
            on_connect=self._on_connect,
            on_message=self._on_message,
            username=config.username,
            password=config.password,
        )

        self.worker_manager.start_workers()
        connect_mqtt_client_with_retries(self.client, config.broker, config.port, process='main')

    def run(self) -> None:
        if self.client is None:
            self.connect()
        client = self.client
        if client is None:
            raise RuntimeError('MQTT client was not initialized')

        try:
            client.loop_forever()
        except KeyboardInterrupt:
            self.logger.info('Stopping RouteMQ application')
        finally:
            self.worker_manager.stop_workers()
            if self.client is not None:
                self.client.disconnect()

    def _on_connect(self, client: Any, userdata: Any, flags: Any, rc: int, *extra: Any) -> None:
        self.logger.info('Connected to MQTT broker', extra={'result_code': rc})
        for route in self.router.routes:
            if route.shared:
                continue
            client.subscribe(route.get_subscription_topic(), route.qos)

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        payload = parse_mqtt_payload(msg.payload)
        try:
            asyncio.run(self.router.dispatch(msg.topic, payload, client))
        except Exception:
            self.logger.exception('Error handling MQTT message', extra={'mqtt_topic': msg.topic})


def main() -> None:
    from routemq.cli import main as routemq_main  # type: ignore[reportMissingImports]

    routemq_main()


if __name__ == '__main__':
    main()
