from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from dotenv import load_dotenv
from routemq.job_registry import discover_and_register_jobs  # type: ignore[reportMissingImports]
from routemq.logging_config import configure_logging  # type: ignore[reportMissingImports]
from routemq.mqtt_utils import (  # type: ignore[reportMissingImports]
    connect_mqtt_client_with_retries,
    create_mqtt_client,
    get_main_client_id,
    get_mqtt_connection_config,
    parse_mqtt_payload,
)  # type: ignore[reportMissingImports]
from routemq.redis_manager import redis_manager  # type: ignore[reportMissingImports]
from routemq.router import Router  # type: ignore[reportMissingImports]
from routemq.router_registry import create_dynamic_router  # type: ignore[reportMissingImports]
from routemq.settings import load_telemetry_settings  # type: ignore[reportMissingImports]
from routemq.telemetry import telemetry  # type: ignore[reportMissingImports]
from routemq.tsdb.telemetry_adapters import adapter_from_settings  # type: ignore[reportMissingImports]
from routemq.worker_manager import WorkerManager  # type: ignore[reportMissingImports]

from ml.monitoring.scheduler import RetrainScheduler


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
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.worker_manager = WorkerManager(self.router, router_directory='app.routers')
        self._telemetry_started = False

    async def _initialize_connections(self) -> None:
        if os.getenv('ENABLE_REDIS', 'false').lower() == 'true':
            await redis_manager.initialize()
        discover_and_register_jobs('app.jobs')
        settings = load_telemetry_settings()
        if not settings.enabled:
            return
        adapter = adapter_from_settings(settings.connection, settings.url, async_insert=settings.async_insert)
        self._telemetry_started = await telemetry.start(
            adapter=adapter,
            settings=settings,
        )

    async def _cleanup_connections(self) -> None:
        if self._telemetry_started:
            await telemetry.close()
        self._telemetry_started = False
        self._retrain_scheduler = RetrainScheduler()

    def connect(self) -> None:
        config = get_mqtt_connection_config()
        self.client = create_mqtt_client(
            get_main_client_id(),
            on_connect=self._on_connect,
            on_message=self._on_message,
            username=config.username,
            password=config.password,
        )

        connect_mqtt_client_with_retries(self.client, config.broker, config.port, process='main')

    def run(self) -> None:
        if self.client is None:
            self.connect()
        client = self.client
        if client is None:
            raise RuntimeError('MQTT client was not initialized')

        self.worker_manager.start_workers()
        client.loop_start()
        try:
            self.loop.run_until_complete(self._initialize_connections())
            if os.getenv('ENABLE_RETRAIN_SCHEDULER', 'false').lower() == 'true':
                self._retrain_scheduler.start(self.loop)
            self.logger.info('Application started. Press Ctrl+C to exit.')
            self.loop.run_forever()
        except KeyboardInterrupt:
            self.logger.info('Stopping RouteMQ application')
        finally:
            self.worker_manager.stop_workers()
            self._retrain_scheduler.shutdown()
            self.loop.run_until_complete(self._cleanup_connections())
            client.loop_stop()
            client.disconnect()
            self.loop.close()

    def _on_connect(self, client: Any, userdata: Any, flags: Any, rc: int, *extra: Any) -> None:
        self.logger.info('Connected to MQTT broker', extra={'result_code': rc})
        for route in self.router.routes:
            if route.shared:
                continue
            client.subscribe(route.get_subscription_topic(), route.qos)

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        coro: Any | None = None
        try:
            payload = parse_mqtt_payload(msg.payload)
            coro = self.router.dispatch(msg.topic, payload, client)
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            future.add_done_callback(lambda completed: self._log_dispatch_result(completed, msg.topic))
        except Exception:
            if coro is not None:
                coro.close()
            self.logger.exception('Error handling MQTT message', extra={'mqtt_topic': msg.topic})

    def _log_dispatch_result(self, future: Any, topic: str) -> None:
        try:
            future.result()
        except Exception:
            self.logger.exception('Error handling MQTT message', extra={'mqtt_topic': topic})


def main() -> None:
    from routemq.cli import main as routemq_main  # type: ignore[reportMissingImports]

    routemq_main()


if __name__ == '__main__':
    main()
