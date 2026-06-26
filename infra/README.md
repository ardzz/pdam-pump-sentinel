# Infrastructure notes

- Prometheus scrapes the RouteMQ application at `app:8080/metrics` and the MQTT exporter at `mosquitto-exporter:9234/metrics`.
- Prometheus loads local alert rules from `infra/prometheus/rules/pumpad-alerts.yml` for app scrape health, telemetry freshness, inference errors, persistence write errors, drift report age, active model age, and the high-severity anomaly alert rule `PDAMHighSeverityAnomalyEvents`.
- The app metrics endpoint is wired by `bootstrap/app.py` and exposes both RouteMQ framework metrics and PDAM-specific `pumpad_*` metric families, including current operator action state via `pumpad_operator_action_state{station,action_type}`.
- ClickHouse stores append-only operator action history in `operator_actions`; Grafana `pumpad-mlops` shows the row `Anomaly alert and operator action evidence` with high-severity anomaly rate, current operator action state, and recent operator actions.
- Grafana dashboards under `infra/grafana/dashboards/` consume those metrics for pipeline observability, MLOps SLIs, system/dependency health, MQTT broker health, and local anomaly/operator action evidence.
- This is a local Docker Compose portfolio stack. Kubernetes, cloud alert routing, Alertmanager routing, PagerDuty-style escalation, OpenTelemetry tracing, and Loki-style centralized logs remain future hardening work.
