from routemq.job_registry import discover_and_register_jobs  # type: ignore[reportMissingImports]
from routemq.redis_manager import redis_manager  # type: ignore[reportMissingImports]
from routemq.router import Router  # type: ignore[reportMissingImports]

import bootstrap.app as bootstrap_app
from bootstrap.app import Application


def _application() -> Application:
    return Application(router=Router(), show_banner=False, log_to_console=False)


class FakeClient:
    def __init__(self):
        self.loop_started = False
        self.loop_stopped = False
        self.disconnected = False

    def loop_start(self):
        self.loop_started = True

    def loop_stop(self):
        self.loop_stopped = True

    def disconnect(self):
        self.disconnected = True


class FakeLoop:
    def __init__(self):
        self.closed = False
        self.run_forever_calls = 0

    def run_until_complete(self, coro):
        return bootstrap_app.asyncio.run(coro)

    def run_forever(self):
        self.run_forever_calls += 1
        raise KeyboardInterrupt

    def close(self):
        self.closed = True


class FakeLifecycleScheduler:
    def __init__(self):
        self.started_with = []
        self.shutdown_calls = 0

    def start(self, loop):
        self.started_with.append(loop)

    def shutdown(self):
        self.shutdown_calls += 1


async def test_initialize_connections_skips_redis_when_disabled(monkeypatch):
    calls = []

    async def fake_initialize():
        calls.append(True)
        return True

    application = _application()
    monkeypatch.setattr(redis_manager, 'initialize', fake_initialize, raising=True)
    monkeypatch.delenv('ENABLE_REDIS', raising=False)
    monkeypatch.setenv('ENABLE_TELEMETRY', 'false')

    await application._initialize_connections()

    assert calls == []


async def test_initialize_connections_initializes_redis_when_enabled(monkeypatch):
    calls = []

    async def fake_initialize():
        calls.append(True)
        return True

    application = _application()
    monkeypatch.setattr(redis_manager, 'initialize', fake_initialize, raising=True)
    monkeypatch.setenv('ENABLE_REDIS', 'true')
    monkeypatch.setenv('ENABLE_TELEMETRY', 'false')

    await application._initialize_connections()

    assert calls == [True]


def test_discover_and_register_jobs_returns_list():
    result = discover_and_register_jobs('app.jobs')

    assert isinstance(result, list)


def test_metrics_health_server_starts_when_enabled(monkeypatch):
    instances = []

    class FakeHealthServer:
        def __init__(self, status, host='127.0.0.1', port=8080, metrics_renderer=None, metrics_path='/metrics'):
            self.status = status
            self.host = host
            self.port = port
            self.metrics_renderer = metrics_renderer
            self.metrics_path = metrics_path
            self.started = False
            self.stopped = False
            instances.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(bootstrap_app, 'HealthServer', FakeHealthServer)
    monkeypatch.setenv('HEALTH_HTTP_ENABLED', 'true')
    monkeypatch.setenv('HEALTH_HTTP_HOST', '0.0.0.0')
    monkeypatch.setenv('HEALTH_HTTP_PORT', '8080')
    monkeypatch.setenv('METRICS_HTTP_ENABLED', 'true')
    monkeypatch.setenv('METRICS_HTTP_PATH', '/metrics')
    monkeypatch.setenv('METRICS_HTTP_SEPARATE', 'false')
    monkeypatch.setenv('METRICS_NAMESPACE', 'routemq')
    monkeypatch.setenv('METRICS_DEFAULT_LABELS', 'service=pdam-pump-sentinel,env=dev')

    application = _application()
    try:
        application._start_health_servers()

        assert len(instances) == 1
        server = instances[0]
        assert server.status is application.health_status
        assert server.host == '0.0.0.0'
        assert server.port == 8080
        assert server.metrics_path == '/metrics'
        assert callable(server.metrics_renderer)
        assert server.started is True
        assert application.metrics_registry is not None
    finally:
        application._stop_health_servers()
        application.loop.close()


def test_metrics_health_server_does_not_start_when_disabled(monkeypatch):
    instances = []

    class FakeHealthServer:
        def __init__(self, *args, **kwargs):
            instances.append((args, kwargs))

        def start(self):
            raise AssertionError('health server should not start')

    monkeypatch.setattr(bootstrap_app, 'HealthServer', FakeHealthServer)
    monkeypatch.setenv('HEALTH_HTTP_ENABLED', 'false')
    monkeypatch.setenv('METRICS_HTTP_ENABLED', 'false')
    monkeypatch.setenv('METRICS_HTTP_SEPARATE', 'false')

    application = _application()
    try:
        application._start_health_servers()

        assert instances == []
        assert application.health_server is None
        assert application.metrics_health_server is None
        assert application.metrics_registry is None
    finally:
        application._stop_health_servers()
        application.loop.close()


def test_metrics_renderer_normalizes_routemq_counter_names_and_appends_pumpad_metrics(monkeypatch):
    application = _application()

    def fake_render_metrics(registry, content_type, static_labels):
        return b'\n'.join(
            [
                b'# HELP routemq_telemetry_points_accepted_total_total Telemetry points accepted.',
                b'# TYPE routemq_telemetry_points_accepted_total_total counter',
                b'routemq_telemetry_points_accepted_total_total{service="pdam-pump-sentinel"} 14',
                b'',
            ]
        )

    monkeypatch.setattr(bootstrap_app, 'render_metrics', fake_render_metrics)
    monkeypatch.setattr(
        bootstrap_app,
        'render_prometheus_client_metrics',
        lambda: b'pumpad_model_info{name="PumpAD",version="6",alias="champion",model_dir="",run_id=""} 1\n',
    )

    try:
        renderer = application._build_metrics_renderer(bootstrap_app.MetricsHttpSettings(enabled=True), bootstrap_app.MetricsRegistry())
        _content_type, payload = renderer(None)
    finally:
        application.loop.close()

    text = payload.decode('utf-8')
    assert 'routemq_telemetry_points_accepted_total_total' not in text
    assert 'routemq_telemetry_points_accepted_total{service="pdam-pump-sentinel"} 14' in text
    assert 'pumpad_model_info{name="PumpAD",version="6",alias="champion",model_dir="",run_id=""} 1' in text


def test_model_refresh_scheduler_disabled_by_default(monkeypatch):
    application = _application()
    original_loop = application.loop
    fake_client = FakeClient()
    fake_loop = FakeLoop()
    retrain_scheduler = FakeLifecycleScheduler()
    drift_scheduler = FakeLifecycleScheduler()
    model_refresh_scheduler = FakeLifecycleScheduler()
    worker_calls = []
    health_calls = []
    cleanup_calls = []

    async def fake_initialize_connections():
        return None

    async def fake_cleanup_connections():
        cleanup_calls.append(True)

    monkeypatch.delenv('ENABLE_RETRAIN_SCHEDULER', raising=False)
    monkeypatch.delenv('ENABLE_DRIFT_SCHEDULER', raising=False)
    monkeypatch.delenv('ENABLE_MODEL_REFRESH_SCHEDULER', raising=False)
    monkeypatch.setattr(application, 'client', fake_client)
    monkeypatch.setattr(application, 'loop', fake_loop)
    monkeypatch.setattr(application, '_retrain_scheduler', retrain_scheduler)
    monkeypatch.setattr(application, '_drift_scheduler', drift_scheduler)
    monkeypatch.setattr(application, '_model_refresh_scheduler', model_refresh_scheduler)
    monkeypatch.setattr(application.worker_manager, 'start_workers', lambda: worker_calls.append('start'))
    monkeypatch.setattr(application.worker_manager, 'stop_workers', lambda: worker_calls.append('stop'))
    monkeypatch.setattr(application, '_start_health_servers', lambda: health_calls.append('start'))
    monkeypatch.setattr(application, '_stop_health_servers', lambda: health_calls.append('stop'))
    monkeypatch.setattr(application, '_initialize_connections', fake_initialize_connections)
    monkeypatch.setattr(application, '_cleanup_connections', fake_cleanup_connections)

    try:
        application.run()
    finally:
        original_loop.close()

    assert retrain_scheduler.started_with == []
    assert drift_scheduler.started_with == []
    assert model_refresh_scheduler.started_with == []
    assert model_refresh_scheduler.shutdown_calls == 1
    assert worker_calls == ['start', 'stop']
    assert health_calls == ['start', 'stop']
    assert cleanup_calls == [True]
    assert fake_client.loop_started is True
    assert fake_client.loop_stopped is True
    assert fake_client.disconnected is True
    assert fake_loop.run_forever_calls == 1
    assert fake_loop.closed is True


def test_model_refresh_scheduler_enabled_starts_and_shuts_down(monkeypatch):
    application = _application()
    original_loop = application.loop
    fake_client = FakeClient()
    fake_loop = FakeLoop()
    model_refresh_scheduler = FakeLifecycleScheduler()

    async def fake_initialize_connections():
        return None

    async def fake_cleanup_connections():
        return None

    monkeypatch.delenv('ENABLE_RETRAIN_SCHEDULER', raising=False)
    monkeypatch.delenv('ENABLE_DRIFT_SCHEDULER', raising=False)
    monkeypatch.setenv('ENABLE_MODEL_REFRESH_SCHEDULER', 'true')
    monkeypatch.setattr(application, 'client', fake_client)
    monkeypatch.setattr(application, 'loop', fake_loop)
    monkeypatch.setattr(application, '_model_refresh_scheduler', model_refresh_scheduler)
    monkeypatch.setattr(application.worker_manager, 'start_workers', lambda: None)
    monkeypatch.setattr(application.worker_manager, 'stop_workers', lambda: None)
    monkeypatch.setattr(application, '_start_health_servers', lambda: None)
    monkeypatch.setattr(application, '_stop_health_servers', lambda: None)
    monkeypatch.setattr(application, '_initialize_connections', fake_initialize_connections)
    monkeypatch.setattr(application, '_cleanup_connections', fake_cleanup_connections)

    try:
        application.run()
    finally:
        original_loop.close()

    assert model_refresh_scheduler.started_with == [fake_loop]
    assert model_refresh_scheduler.shutdown_calls == 1
