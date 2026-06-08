from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = ROOT / 'infra/grafana/dashboards'


def _load_dashboard(name: str):
    return json.loads((DASHBOARD_DIR / name).read_text(encoding='utf-8'))


def _dashboard_queries(dashboard) -> list[str]:
    return [
        str(target[key])
        for panel in dashboard.get('panels', [])
        for target in panel.get('targets', [])
        for key in ('expr', 'query', 'rawSql')
        if key in target
    ]


def _load_yaml(relative_path: str):
    return yaml.safe_load((ROOT / relative_path).read_text(encoding='utf-8'))


def test_prometheus_scrapes_routemq_app_metrics_endpoint():
    config = _load_yaml('infra/prometheus/prometheus.yml')

    job = next(job for job in config['scrape_configs'] if job['job_name'] == 'routemq-app')

    assert config['global']['scrape_interval'] == '15s'
    assert job['metrics_path'] == '/metrics'
    assert job['static_configs'][0]['targets'] == ['app:8080']


def test_prometheus_loads_local_pumpad_alert_rules():
    config = _load_yaml('infra/prometheus/prometheus.yml')
    rules = _load_yaml('infra/prometheus/rules/pumpad-alerts.yml')
    alerts = {
        rule['alert']
        for group in rules['groups']
        for rule in group['rules']
        if 'alert' in rule
    }

    assert '/etc/prometheus/rules/*.yml' in config['rule_files']
    assert alerts >= {
        'PDAMAppMetricsDown',
        'PDAMTelemetryStale',
        'PDAMInferenceErrors',
        'PDAMPersistenceWriteErrors',
        'PDAMDriftReportStale',
        'PDAMActiveModelStale',
    }


def test_prometheus_scrapes_mosquitto_exporter():
    config = _load_yaml('infra/prometheus/prometheus.yml')

    job = next(job for job in config['scrape_configs'] if job['job_name'] == 'mosquitto-exporter')

    assert job['metrics_path'] == '/metrics'
    assert job['static_configs'][0]['targets'] == ['mosquitto-exporter:9234']


def test_grafana_provisions_clickhouse_datasource():
    config = _load_yaml('infra/grafana/provisioning/datasources/datasource.yml')

    datasource = next(source for source in config['datasources'] if source['name'] == 'ClickHouse')

    assert datasource['uid'] == 'clickhouse'
    assert datasource['type'] == 'grafana-clickhouse-datasource'
    assert datasource['access'] == 'proxy'
    assert datasource['url'] == 'http://clickhouse:8123'
    assert datasource['user'] == 'default'
    assert datasource['isDefault'] is False
    assert datasource['editable'] is False
    assert datasource['jsonData']['defaultDatabase'] == 'default'
    assert datasource['jsonData']['protocol'] == 'http'
    assert datasource['jsonData']['host'] == 'clickhouse'
    assert datasource['jsonData']['server'] == 'clickhouse'
    assert datasource['jsonData']['port'] == 8123
    assert datasource['secureJsonData'] == {}


def test_grafana_dashboard_loads_and_uses_routemq_metrics():
    dashboard = _load_dashboard('pumpad.json')
    expressions = [
        target['expr']
        for panel in dashboard['panels']
        for target in panel.get('targets', [])
    ]

    assert dashboard['uid'] == 'pumpad-observability'
    assert any('routemq_router_dispatch_started_total' in expression for expression in expressions)
    assert any('routemq_router_dispatch_succeeded_total' in expression for expression in expressions)
    assert any('routemq_telemetry_points_accepted_total' in expression for expression in expressions)
    assert any('routemq_telemetry_queue_depth' in expression for expression in expressions)
    assert any('pumpad_inference_events_total' in expression for expression in expressions)
    assert any('pumpad_persistence_writes_total' in expression for expression in expressions)
    assert not any('_total_total' in expression for expression in expressions)


def test_grafana_mlops_dashboard_references_new_metrics():
    dashboard = _load_dashboard('pumpad-mlops.json')
    queries = _dashboard_queries(dashboard)

    assert any('pumpad_model_info' in query for query in queries)
    assert any('pumpad_inference_latency_seconds' in query for query in queries)
    assert any('pumpad_drift_share' in query for query in queries)
    assert any('pumpad_drift_detected' in query for query in queries)
    assert any('pumpad_anomaly_score' in query for query in queries)
    assert any('pumpad_retraining_jobs_total' in query for query in queries)
    assert any('pumpad_inference_events_total' in query for query in queries)
    assert any('pumpad_anomaly_events_total' in query for query in queries)
    assert any('pumpad_drift_report_age_seconds' in query for query in queries)
    assert any('pumpad_retrain_duration_seconds' in query for query in queries)
    assert any('pumpad_active_model_age_seconds' in query for query in queries)


def test_grafana_system_health_dashboard_references_slo_metrics():
    dashboard = _load_dashboard('pumpad-system-health.json')
    queries = _dashboard_queries(dashboard)

    assert dashboard['uid'] == 'pumpad-system-health'
    assert any('up{job="routemq-app"}' in query for query in queries)
    assert any('pumpad_telemetry_freshness_seconds' in query for query in queries)
    assert any('pumpad_persistence_writes_total' in query for query in queries)
    assert any('pumpad_active_model_age_seconds' in query for query in queries)


def test_grafana_dashboards_do_not_use_fake_placeholder_expressions():
    for dashboard_name in ('pumpad.json', 'pumpad-mlops.json', 'pumpad-system-health.json'):
        queries = _dashboard_queries(_load_dashboard(dashboard_name))
        assert not any('0 * sum' in query for query in queries), dashboard_name


def test_grafana_dashboards_have_annotation_queries():
    dashboard_names = [
        'pumpad.json',
        'pumpad-mlops.json',
        'pumpad-system-health.json',
        'pumpad-mqtt-broker.json',
    ]

    for dashboard_name in dashboard_names:
        dashboard = _load_dashboard(dashboard_name)
        assert dashboard.get('annotations', {}).get('list'), dashboard_name


def test_grafana_mqtt_broker_dashboard_exists_and_queries_mosquitto():
    dashboard_path = DASHBOARD_DIR / 'pumpad-mqtt-broker.json'

    assert dashboard_path.exists()

    queries = _dashboard_queries(_load_dashboard('pumpad-mqtt-broker.json'))
    assert any(
        'broker_clients_connected' in query or 'broker_messages_received' in query
        for query in queries
    )


def test_compose_includes_mosquitto_exporter_service():
    compose = _load_yaml('infra/docker-compose.dev.yml')
    service = compose['services']['mosquitto-exporter']

    assert service['image'] == 'sapcc/mosquitto-exporter:latest'
    assert service['environment']['BROKER_ENDPOINT'] == 'tcp://mosquitto:1883'
    assert service['environment']['BIND_ADDRESS'] == '0.0.0.0:9234'
    assert service['ports'] == ['${MQTT_EXPORTER_PORT:-19234}:9234']
    assert service['depends_on']['mosquitto']['condition'] == 'service_healthy'


def test_compose_defines_prometheus_and_grafana_services():
    compose = _load_yaml('infra/docker-compose.dev.yml')
    services = compose['services']

    assert services['prometheus']['image'].startswith('prom/prometheus')
    assert './prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro' in services['prometheus']['volumes']
    assert './prometheus/rules:/etc/prometheus/rules:ro' in services['prometheus']['volumes']
    assert services['grafana']['image'].startswith('grafana/grafana')
    assert services['grafana']['environment']['GF_INSTALL_PLUGINS'] == 'grafana-clickhouse-datasource'
    assert './grafana/provisioning:/etc/grafana/provisioning:ro' in services['grafana']['volumes']
    assert 'grafana-data' in compose['volumes']
