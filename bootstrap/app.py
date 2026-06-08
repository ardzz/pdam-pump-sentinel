from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from routemq.health import HealthServer, HealthStatus  # type: ignore[reportMissingImports]
from routemq.job_registry import discover_and_register_jobs  # type: ignore[reportMissingImports]
from routemq.logging_config import configure_logging  # type: ignore[reportMissingImports]
from routemq.metrics.exposition import (  # type: ignore[reportMissingImports]
    negotiate_content_type,
)
from routemq.metrics.exposition import (  # type: ignore[reportMissingImports]
    render as render_metrics,
)
from routemq.metrics.hooks import DefaultHooksHandle, install_default_hooks  # type: ignore[reportMissingImports]
from routemq.metrics.registry import MetricsRegistry  # type: ignore[reportMissingImports]
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
from routemq.settings import (  # type: ignore[reportMissingImports]
    MetricsHttpSettings,
    load_health_http_settings,
    load_metrics_http_settings,
    load_telemetry_settings,
)
from routemq.telemetry import telemetry  # type: ignore[reportMissingImports]
from routemq.tsdb.telemetry_adapters import adapter_from_settings  # type: ignore[reportMissingImports]
from routemq.worker_manager import WorkerManager  # type: ignore[reportMissingImports]

from app.observability.metrics import render_prometheus_client_metrics
from ml.monitoring.scheduler import DriftScheduler, RetrainScheduler

_DUPLICATE_TOTAL_COMMENT_RE = re.compile(rb'(?m)^(# (?:HELP|TYPE) [A-Za-z_:][A-Za-z0-9_:]*)_total_total(?=\s)')
_DUPLICATE_TOTAL_SAMPLE_RE = re.compile(rb'(?m)^([A-Za-z_:][A-Za-z0-9_:]*)_total_total(?=[{\s])')


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
        self._retrain_scheduler = RetrainScheduler()
        self._drift_scheduler = DriftScheduler()
        self.health_status = HealthStatus()
        self.health_server: HealthServer | None = None
        self.metrics_health_server: HealthServer | None = None
        self.metrics_registry: MetricsRegistry | None = None
        self._metrics_hooks_handle: DefaultHooksHandle | None = None

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

    def _start_health_servers(self) -> None:
        if self.health_server is not None or self.metrics_health_server is not None:
            return

        health_settings = load_health_http_settings()
        metrics_settings = load_metrics_http_settings()
        metrics_renderer: Callable[[str | None], tuple[str, bytes]] | None = None

        if metrics_settings.enabled:
            metrics_registry = MetricsRegistry()
            self.metrics_registry = metrics_registry
            self._metrics_hooks_handle = install_default_hooks(
                metrics_registry,
                namespace=metrics_settings.namespace,
                histogram_buckets=metrics_settings.histogram_buckets,
            )
            metrics_renderer = self._build_metrics_renderer(metrics_settings, metrics_registry)

        if metrics_settings.enabled and metrics_settings.separate:
            if health_settings.enabled:
                self.health_server = HealthServer(
                    self.health_status,
                    host=health_settings.host,
                    port=health_settings.port,
                )
                self.health_server.start()
            self.metrics_health_server = HealthServer(
                self.health_status,
                host=metrics_settings.host,
                port=metrics_settings.port,
                metrics_renderer=metrics_renderer,
                metrics_path=metrics_settings.path,
            )
            self.metrics_health_server.start()
            return

        if health_settings.enabled or metrics_renderer is not None:
            self.health_server = HealthServer(
                self.health_status,
                host=health_settings.host,
                port=health_settings.port,
                metrics_renderer=metrics_renderer,
                metrics_path=metrics_settings.path,
            )
            self.health_server.start()

    def _stop_health_servers(self) -> None:
        if self.metrics_health_server is not None:
            self.metrics_health_server.stop()
            self.metrics_health_server = None
        if self.health_server is not None:
            self.health_server.stop()
            self.health_server = None
        if self._metrics_hooks_handle is not None:
            self._metrics_hooks_handle.unregister()
            self._metrics_hooks_handle = None
        self.metrics_registry = None

    def _build_metrics_renderer(
        self,
        settings: MetricsHttpSettings,
        registry: MetricsRegistry,
    ) -> Callable[[str | None], tuple[str, bytes]]:
        default_labels = dict(settings.default_labels)

        def render(accept: str | None) -> tuple[str, bytes]:
            content_type = negotiate_content_type(accept)
            routemq_payload = _normalize_duplicate_total_suffixes(
                render_metrics(registry, content_type=content_type, static_labels=default_labels)
            )
            app_payload = render_prometheus_client_metrics()
            if not app_payload:
                return content_type, routemq_payload
            return content_type, _join_metrics_payloads(routemq_payload, app_payload)

        return render

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
            self.health_status.alive = True
            self.health_status.shutting_down = False
            self._start_health_servers()
            self.loop.run_until_complete(self._initialize_connections())
            if os.getenv('ENABLE_RETRAIN_SCHEDULER', 'false').lower() == 'true':
                self._retrain_scheduler.start(self.loop)
            if os.getenv('ENABLE_DRIFT_SCHEDULER', 'false').lower() == 'true':
                self._drift_scheduler.start(self.loop)
            self.health_status.startup_complete = True
            self.logger.info('Application started. Press Ctrl+C to exit.')
            self.loop.run_forever()
        except KeyboardInterrupt:
            self.logger.info('Stopping RouteMQ application')
        finally:
            self.health_status.shutting_down = True
            self.worker_manager.stop_workers()
            self._retrain_scheduler.shutdown()
            self._drift_scheduler.shutdown()
            self.loop.run_until_complete(self._cleanup_connections())
            self._stop_health_servers()
            client.loop_stop()
            client.disconnect()
            self.health_status.alive = False
            self.loop.close()

    def _on_connect(self, client: Any, userdata: Any, flags: Any, rc: int, *extra: Any) -> None:
        self.logger.info('Connected to MQTT broker', extra={'result_code': rc})
        self.health_status.mqtt_connected = rc == 0
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


def _normalize_duplicate_total_suffixes(payload: bytes) -> bytes:
    payload = _DUPLICATE_TOTAL_COMMENT_RE.sub(rb'\1_total', payload)
    return _DUPLICATE_TOTAL_SAMPLE_RE.sub(rb'\1_total', payload)


def _join_metrics_payloads(*payloads: bytes) -> bytes:
    chunks = []
    has_openmetrics_eof = False
    for payload in payloads:
        if not payload or not payload.strip():
            continue
        chunk = payload.rstrip()
        if chunk.endswith(b'# EOF'):
            has_openmetrics_eof = True
            chunk = chunk[: -len(b'# EOF')].rstrip()
        if chunk:
            chunks.append(chunk)
    if not chunks:
        return b''
    joined = b'\n'.join(chunks) + b'\n'
    if has_openmetrics_eof:
        joined += b'# EOF\n'
    return joined


if __name__ == '__main__':
    main()
