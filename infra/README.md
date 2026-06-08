# Infrastructure notes

- Prometheus scrapes the RouteMQ application at `app:8080/metrics` and the MQTT exporter at `mosquitto-exporter:9234/metrics`.
- Prometheus loads local alert rules from `infra/prometheus/rules/pumpad-alerts.yml` for app scrape health, telemetry freshness, inference errors, persistence write errors, drift report age, and active model age.
- The app metrics endpoint is wired by `bootstrap/app.py` and exposes both RouteMQ framework metrics and PDAM-specific `pumpad_*` metric families.
- Grafana dashboards under `infra/grafana/dashboards/` consume those metrics for pipeline observability, MLOps SLIs, system/dependency health, and MQTT broker health.
- This is a local Docker Compose portfolio stack. Kubernetes, cloud alert routing, PagerDuty-style escalation, tracing, and centralized log correlation remain future hardening work.
