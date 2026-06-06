# Demo Screenshot Checklist

Capture sequence aligned with `design.md §13.1` storyboard. Use `make screenshots TAG=<phase>` between demo phases to batch all canonical screenshots, or pass `SCREENSHOT_TARGETS="<labels>"` for a subset.

## Prerequisites

- Dev stack up via `make dev` (mosquitto, redis, clickhouse, mlflow, prometheus, grafana, app).
- Host RouteMQ app running via `make run` (so Streamlit data is live).
- Dashboard up via `make dashboard` (Streamlit on `:8501`).
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
| `streamlit-home` | http://localhost:8501/ | Streamlit landing |
| `streamlit-live-sensors` | http://localhost:8501/live_sensors | Live anomaly status (red/green) + score |
| `streamlit-anomaly-history` | http://localhost:8501/anomaly_history | Anomaly score timeline + raw telemetry |
| `streamlit-model-registry` | http://localhost:8501/model_registry | name / version / activated_at |
| `streamlit-drift-reports` | http://localhost:8501/drift_reports | Drift & Training summary |

Run `make screenshots-list` to print this catalog at any time.

## Per-phase capture sequence

Storyboard timing matches `scripts/run_e2e_demo.py` phases.

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

## Workflow

```bash
# Once: bring up everything
make dev
make run             # in a second terminal — host app
make dashboard       # in a third terminal — Streamlit

# Optional: seed MLflow + Redis champion (idempotent)
MLFLOW_TRACKING_URI=http://localhost:5000 REDIS_HOST=localhost REDIS_PORT=6379 ENABLE_REDIS=true \
  uv run python scripts/seed_initial_models.py \
    --input tests/fixtures/skab_tiny.csv --output-dir /tmp/pdam-seed \
    --window-size 1 --stride 1 --n-components 2

# Baseline shot before kicking the orchestrator
make screenshots TAG=baseline

# Drive the demo (pause / no-assert recommended for live walkthrough)
uv run python scripts/run_e2e_demo.py --clean --no-assert

# Between phases as needed:
make screenshots TAG=t2-anomaly SCREENSHOT_TARGETS="streamlit-live-sensors streamlit-anomaly-history"
make screenshots TAG=t7-promote SCREENSHOT_TARGETS="mlflow-pumpad streamlit-model-registry"
make screenshots TAG=t8-recover

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
