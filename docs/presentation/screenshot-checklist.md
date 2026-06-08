# Demo Screenshot Checklist

Capture sequence aligned with `design.md §13.1` storyboard and the portfolio observability spec in `docs/plans/2026-06-08-portfolio-observability-upgrades.md`. Use `make screenshots TAG=<phase>` between demo phases to batch all canonical screenshots (`TAG` aliases `SCREENSHOT_TAG`), or pass `SCREENSHOT_TARGETS="<labels>"` for a subset.

## Prerequisites

- Dev stack up via `make dev` (mosquitto, redis, clickhouse, mlflow, prometheus, grafana, app).
- Dashboard up via `make dashboard` (Streamlit on `:8501`); set `TELEMETRY_URL`, `MLFLOW_TRACKING_URI`, `MQTT_HOST`, and `MQTT_PORT` if you are not using the default local ports.
- Stack has been seeded via `scripts/seed_initial_models.py` so `PumpAD@champion` resolves.
- Grafana anonymous viewer enabled (default after the dev compose update; override via `GF_AUTH_ANONYMOUS_ENABLED=false` if you need login-only).

## Canonical URLs

| Label | URL | Caption hint |
|---|---|---|
| `mlflow-home` | http://localhost:5000/ | MLflow landing — show experiments list |
| `mlflow-experiments` | http://localhost:5000/#/experiments | Experiments table including `pump_sentinel_model_comparison` |
| `mlflow-models` | http://localhost:5000/#/models | Registered models with PumpAD versions + champion alias |
| `mlflow-pumpad` | http://localhost:5000/#/models/PumpAD | PumpAD detail — version history + aliases panel |
| `grafana-routemq` | http://localhost:13000/d/pumpad-observability/pdam-pump-sentinel-routemq-observability?from=now-1h&to=now&timezone=browser&refresh=10s&kiosk=tv | RouteMQ observability dashboard in kiosk mode |
| `grafana-pipeline-observability` | http://localhost:13000/d/pumpad-observability/pdam-pump-sentinel-routemq-observability?from=now-1h&to=now&timezone=browser&refresh=10s&kiosk=tv | Pipeline row: dispatch → inference → persistence |
| `grafana-mlops` | http://localhost:13000/d/pumpad-mlops/mlops-loop?from=now-1h&to=now&timezone=browser&refresh=10s&kiosk=tv | Grafana MLOps loop dashboard |
| `grafana-mlops-observability` | http://localhost:13000/d/pumpad-mlops/mlops-loop?from=now-1h&to=now&timezone=browser&refresh=10s&kiosk=tv | MLOps SLIs: inference events, anomaly severity, drift age, retrain duration, model age |
| `grafana-system-health` | http://localhost:13000/d/pumpad-system-health/system-health?from=now-1h&to=now&timezone=browser&refresh=10s&kiosk=tv | Grafana system health dashboard |
| `grafana-slo-health` | http://localhost:13000/d/pumpad-system-health/system-health?from=now-1h&to=now&timezone=browser&refresh=10s&kiosk=tv | SLO/dependency health: persistence errors and active model freshness |
| `grafana-mqtt-broker` | http://localhost:13000/d/pumpad-mqtt-broker/mqtt-broker?from=now-1h&to=now&timezone=browser&refresh=10s&kiosk=tv | Grafana MQTT broker clients, throughput, and uptime |
| `streamlit-home` | http://localhost:8501/ | Streamlit landing |
| `streamlit-overview` | http://localhost:8501/overview | Streamlit Overview landing |
| `streamlit-observability-snapshot` | http://localhost:8501/overview | Observability Snapshot cards |
| `streamlit-live-sensors` | http://localhost:8501/live_sensors | Live anomaly status (red/green) + score |
| `streamlit-anomaly-history` | http://localhost:8501/anomaly_history | Anomaly score timeline + raw telemetry |
| `streamlit-model-registry` | http://localhost:8501/model_registry | name / version / activated_at |
| `streamlit-drift-reports` | http://localhost:8501/drift_reports | Drift & Training summary |
| `streamlit-system-health` | http://localhost:8501/system_health | App Metric Freshness + Service Checks |
| `streamlit-runbook` | http://localhost:8501/runbook | Streamlit Runbook page |
| `streamlit-runbook-observability` | http://localhost:8501/runbook | Metric-driven observability triage expander |
| `mlflow-compare-experiments` | http://localhost:5050/#/experiments | MLflow report server cross-family comparison experiment |

Run `make screenshots-list` to print this catalog at any time.

## Per-phase capture sequence

Storyboard timing matches `scripts/run_e2e_demo.py` phases. T+9 is optional and appears when the demo is run with `--observability-evidence` or `DEMO_OBSERVABILITY_EVIDENCE=true`.

| When | Tag | Targets | Reason |
|---|---|---|---|
| Before T+0 (cold demo) | `baseline` | `mlflow-pumpad streamlit-model-registry streamlit-live-sensors grafana-routemq` | Show champion v1 in place, dashboard hijau, baseline traffic level |
| After T+1 normal replay | `t1-normal` | `streamlit-live-sensors streamlit-anomaly-history` | Anomaly score low/intermittent, dashboard green |
| After T+2 anomalous replay | `t2-anomaly` | `streamlit-live-sensors streamlit-anomaly-history` | "ANOMALY DETECTED" red banner + score spike |
| After T+3 drift inject | `t3-drift` | `streamlit-anomaly-history grafana-routemq` | Score elevated under drift, observability rates change |
| After T+4 detect | `t4-detect` | `streamlit-drift-reports` | (UI capture optional; orchestrator already logs `DriftResult`) |
| After T+5 retrain | `t5-retrain` | `mlflow-experiments mlflow-pumpad` | New experiment run, new PumpAD version row |
| After T+7 promote | `t7-promote` | `mlflow-pumpad streamlit-model-registry` | Champion alias points to challenger version |
| After T+8 recover | `t8-recover` | `streamlit-live-sensors streamlit-anomaly-history grafana-routemq` | Dashboard green again, anomaly history shows recovery, observability normalized |
| After T+9 evidence | `t9-observability` | `grafana-pipeline-observability grafana-mlops-observability grafana-slo-health streamlit-observability-snapshot streamlit-runbook-observability` | Portfolio evidence for the upgraded observability layer |

## Workflow

```bash
# Once: bring up everything
make dev
make dashboard       # in a second terminal — Streamlit

# Optional: seed MLflow + Redis champion (idempotent)
MLFLOW_TRACKING_URI=http://localhost:5000 REDIS_HOST=localhost REDIS_PORT=6379 ENABLE_REDIS=true \
  uv run python scripts/seed_initial_models.py \
    --input tests/fixtures/skab_tiny.csv --output-dir /tmp/pdam-seed \
    --window-size 1 --stride 1 --n-components 2

# Baseline shot before kicking the orchestrator
make screenshots TAG=baseline

# Drive the verified portfolio demo, including optional T+9 evidence in the same clean run
uv run python scripts/run_e2e_demo.py --clean --observability-evidence

# For narrated walkthrough only, after the verified run above has passed:
# uv run python scripts/run_e2e_demo.py --no-assert

# Between phases as needed:
make screenshots TAG=t2-anomaly SCREENSHOT_TARGETS="streamlit-live-sensors streamlit-anomaly-history"
make screenshots TAG=t7-promote SCREENSHOT_TARGETS="mlflow-pumpad streamlit-model-registry"
make screenshots TAG=t8-recover
make screenshots TAG=t9-observability SCREENSHOT_TARGETS="grafana-pipeline-observability grafana-mlops-observability grafana-slo-health streamlit-observability-snapshot streamlit-runbook-observability"

# Or capture everything at once after the demo:
make screenshots TAG=postrun
```

Output lands under `docs/presentation/screenshots/<tag>-<label>-<UTC timestamp>.png`. Use `--no-timestamp` to disable the timestamp suffix when overwriting is desired.

## Notes

- Chrome binary is auto-discovered from the Playwright cache under `~/.cache/ms-playwright/chromium-*`. Override via `PDAM_CHROME_BIN=/path/to/chrome` if you have a different setup.
- Grafana panels need ~5–10s after the dashboard opens for queries to settle; the script already waits via `--virtual-time-budget`.
- Streamlit page routing uses lower_snake page names (`/live_sensors`, `/anomaly_history`, etc.) derived from the file names under `dashboard/pages/`.
- For visual continuity in slides, capture all "before" shots in one batch and all "after" shots in another so colors, widths, and viewport sizes match.
- If a capture comes back blank or login-page-sized (Grafana < 10 KB), confirm `GF_AUTH_ANONYMOUS_ENABLED=true` in the running container's env (`docker exec pdam-pump-sentinel-dev-grafana-1 env | grep GF_AUTH`).
