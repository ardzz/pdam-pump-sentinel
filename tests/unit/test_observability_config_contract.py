from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(relative_path: str):
    return yaml.safe_load((ROOT / relative_path).read_text(encoding='utf-8'))


def test_prometheus_scrapes_routemq_app_metrics_endpoint():
    config = _load_yaml('infra/prometheus/prometheus.yml')

    job = next(job for job in config['scrape_configs'] if job['job_name'] == 'routemq-app')

    assert config['global']['scrape_interval'] == '15s'
    assert job['metrics_path'] == '/metrics'
    assert job['static_configs'][0]['targets'] == ['app:8080']


def test_grafana_dashboard_loads_and_uses_routemq_metrics():
    dashboard = json.loads((ROOT / 'infra/grafana/dashboards/pumpad.json').read_text(encoding='utf-8'))
    expressions = [target['expr'] for panel in dashboard['panels'] for target in panel['targets']]

    assert dashboard['uid'] == 'pumpad-observability'
    assert any('routemq_mqtt_messages_received_total_total' in expression for expression in expressions)
    assert any('routemq_telemetry_points_accepted_total_total' in expression for expression in expressions)
    assert any('routemq_telemetry_queue_depth' in expression for expression in expressions)


def test_compose_defines_prometheus_and_grafana_services():
    compose = _load_yaml('infra/docker-compose.dev.yml')
    services = compose['services']

    assert services['prometheus']['image'].startswith('prom/prometheus')
    assert './prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro' in services['prometheus']['volumes']
    assert services['grafana']['image'].startswith('grafana/grafana')
    assert './grafana/provisioning:/etc/grafana/provisioning:ro' in services['grafana']['volumes']
    assert 'grafana-data' in compose['volumes']
