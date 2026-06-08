# Portfolio Observability Upgrade Plan & Spec

**Project:** PDAM Pump Sentinel  
**Date:** 2026-06-08  
**Status:** Draft implementation spec  
**Goal:** Upgrade the current demo/dev observability layer into a portfolio-grade, industrial-style observability story for an ML anomaly detection system.

---

## 1. Executive Summary

PDAM Pump Sentinel already has a strong observability foundation: Prometheus metrics, Grafana dashboards, Streamlit operator pages, MLflow tracking, Evidently drift checks, Redis latest state, and ClickHouse anomaly history. The upgrade targets production-style monitoring behavior for the local portfolio stack: local alert rules, explicit SLIs/SLOs, telemetry freshness/data-quality signals, historical drift/retraining evidence, operator-facing runbooks, and clear future boundaries for tracing/log correlation.

This plan turns the project into a LinkedIn/portfolio-ready system with a clear story:

> “I built an industrial-style ML observability layer for pump anomaly detection, covering infrastructure health, telemetry freshness, model latency, drift, retraining, data quality, alerting, and operator runbooks.”

The plan intentionally avoids Kubernetes, cloud migration, PagerDuty, or enterprise incident tooling. The target is a **local Docker Compose portfolio stack** that is realistic, demonstrable, testable, and screenshot-friendly.

---

## 2. Current Baseline

### 2.1 Existing Strengths

| Capability | Current State | Key Files |
|---|---|---|
| Prometheus scrape | App and MQTT exporter are scraped | `infra/prometheus/prometheus.yml` |
| App metrics | PumpAD model, latency, score, drift, freshness, retrain metrics | `app/observability/metrics.py` |
| Metrics endpoint | RouteMQ metrics and PumpAD metrics exposed together | `bootstrap/app.py` |
| Grafana dashboards | RouteMQ, MLOps, System Health, MQTT Broker | `infra/grafana/dashboards/*.json` |
| Streamlit operator UI | Overview, live sensors, anomaly history, registry, drift, health, runbook | `dashboard/pages/*.py` |
| MLflow | Model tracking, registry alias, metrics, artifacts, tags | `ml/registry/mlflow_client.py` |
| Drift checks | Evidently drift result and retraining trigger | `ml/monitoring/drift_check.py`, `app/jobs/drift_report_job.py` |
| Persistence | Redis latest state and ClickHouse telemetry/anomaly rows | `app/services/persistence.py`, `infra/clickhouse/init.sql` |
| Tests | Metrics/config/dashboard/job/persistence contracts | `tests/unit/*observability*`, `tests/unit/test_mlops_jobs_contract.py` |

### 2.2 Current Gaps

| Gap | Impact |
|---|---|
| No external notification routing | Local alert rules exist, but Alertmanager/Grafana notification routing remains future work. |
| No explicit SLI/SLO definitions | Reliability targets are implied, not measurable as portfolio evidence. |
| Live data-quality monitoring is weak | Bad sensor data can look like a real anomaly. |
| Drift/retrain state is mostly latest-only Redis payload | Hard to show lifecycle history over time. |
| Retrain metadata is incomplete | Dashboard cannot fully explain when/why/how retraining happened. |
| No distributed tracing | Cannot trace MQTT telemetry through validation, inference, Redis, ClickHouse, and anomaly publish. |
| No centralized structured logs | Debugging relies on scattered process logs. |
| Human feedback is only ack/mute/note | No durable true/false anomaly labels or model-improvement loop yet. |

---

## 3. Scope

### 3.1 In Scope

- Local Prometheus/Grafana alerting and SLO-style dashboards.
- New or enriched `pumpad_*` Prometheus metrics.
- Live telemetry freshness/data-quality signals for pump telemetry.
- Timestamped drift/retrain history persisted to ClickHouse.
- Streamlit operator observability upgrades.
- Runbook improvements tied to actual alerts and metrics.
- Demo/screenshot evidence for portfolio and LinkedIn.

### 3.2 Deferred / Future Work

- OpenTelemetry tracing with Tempo.
- Centralized logs with Loki + Vector.
- Durable human feedback loop with labels and retraining integration.
- Runtime MLflow alias polling.
- Kubernetes/cloud deployment.
- Enterprise incident management tooling.

These are implementable, but they are not required for the first portfolio milestone.

---

## 4. MVP Definition

The MVP is complete after **Phases 1–3**.

### MVP Deliverables

1. Explicit SLI/SLO definitions in code/docs.
2. New or enriched observability metric contracts.
3. Backend instrumentation for inference, persistence, drift, and retraining evidence.
4. Grafana panels showing SLO status, alerts, data freshness, model health, drift, retraining, and pipeline health.

### MVP Acceptance Criteria

| Check | Acceptance Criteria |
|---|---|
| Metrics contract | `tests/unit/test_observability_metrics_contract.py` asserts all new metric families and label names. |
| App metrics endpoint | `tests/unit/test_bootstrap_worker_contract.py` still proves RouteMQ metrics and `pumpad_*` metrics are exposed together. |
| Grafana contract | `tests/unit/test_observability_config_contract.py` asserts dashboards reference real metrics and no fake placeholder expressions. |
| Offline tests | New unit tests do not require Docker, Streamlit, Grafana, Prometheus, Redis, ClickHouse, or MLflow to be running. |
| Portfolio story | Grafana can show service health, model health, drift, retrain outcome, and telemetry freshness in one coherent flow. |

---

## 5. Phase Plan

## Phase 1 — Observability Contract & SLI/Metric Taxonomy

### Objective

Define the portfolio-grade observability vocabulary before changing runtime behavior.

### Proposed SLIs

| SLI | Meaning | Data Source |
|---|---|---|
| App scrape availability | Can Prometheus scrape the app metrics endpoint? | `up{job="routemq-app"}` |
| Telemetry freshness | How stale is the latest station reading? | `pumpad_telemetry_freshness_seconds` |
| Inference latency | How long model scoring takes | `pumpad_inference_latency_seconds` |
| Inference success/error volume | Whether inference is processing reliably | new `pumpad_inference_events_total` |
| Persistence write health | Whether Redis/ClickHouse writes succeed | new `pumpad_persistence_writes_total` |
| Drift state | Whether production data has drifted | `pumpad_drift_detected`, `pumpad_drift_share` |
| Retraining health | Whether retraining completes/promotes/rejects/errors | `pumpad_retraining_jobs_total`, new duration metric |
| Active model freshness | Whether champion model metadata is present and fresh | new `pumpad_active_model_age_seconds` |

### Proposed New Metrics

| Metric | Type | Labels | Purpose |
|---|---|---|---|
| `pumpad_inference_events_total` | Counter | `station`, `model_version`, `result` | Count successful/error inference events. |
| `pumpad_persistence_writes_total` | Counter | `target`, `result` | Count Redis/ClickHouse write outcomes. |
| `pumpad_anomaly_events_total` | Counter | `station`, `severity`, `model_version` | Count anomaly events by severity. |
| `pumpad_drift_report_age_seconds` | Gauge | none | Age of latest drift report. |
| `pumpad_retrain_duration_seconds` | Histogram | `result` | Retraining duration distribution. |
| `pumpad_active_model_age_seconds` | Gauge | `name`, `version`, `alias` | Age of active champion model metadata. |
| `pumpad_observability_build_info` | Gauge | `schema_version` | Dashboard/schema compatibility marker. |

### Files

| File | Change |
|---|---|
| `tests/unit/test_observability_metrics_contract.py` | Add expected metrics and labels. |
| `app/observability/metrics.py` | Add metric definitions and helper setters. |
| `app/observability/_prometheus_fallback.py` | Update only if fallback compatibility requires it. |

### Acceptance Criteria

- Metric tests assert every new metric family name.
- Metric tests assert stable label names.
- Rendered Prometheus text contains new metric families.
- Existing metric tests still pass.

---

## Phase 2 — Backend Pipeline Instrumentation

### Objective

Emit the Phase 1 signals from actual runtime paths.

### Files

| File | Change |
|---|---|
| `app/controllers/anomaly_controller.py` | Record inference success/error, latency, anomaly severity. |
| `app/services/persistence.py` | Record Redis and ClickHouse write success/error. |
| `app/jobs/drift_report_job.py` | Record drift report timestamp, age, and status. |
| `app/jobs/retraining_job.py` | Record retrain duration, result, active model age. |
| `tests/unit/test_anomaly_controller_contract.py` | Assert inference metrics with fake service. |
| `tests/unit/test_persistence_contract.py` | Assert Redis/ClickHouse counters with fake adapters. |
| `tests/unit/test_mlops_jobs_contract.py` | Assert drift/retrain metrics and payload shape. |

### Acceptance Criteria

- Valid fake telemetry increments `pumpad_inference_events_total{result="success"}`.
- Fake inference failure increments `pumpad_inference_events_total{result="error"}` without hiding the existing error behavior.
- Redis and ClickHouse fake writes increment `pumpad_persistence_writes_total` with `target="redis"` and `target="clickhouse"`.
- Drift job sets drift report age to a bounded non-negative value.
- Retraining job observes duration exactly once per completed fake run.

---

## Phase 3 — Grafana MVP Dashboards

### Objective

Make the new observability signals visible and screenshot-ready in Grafana.

### Dashboard Upgrades

| Dashboard | Add / Update |
|---|---|
| `pumpad-mlops.json` | Inference event rate, p95 latency, anomaly event rate, drift age, retrain duration, active model age. |
| `pumpad-system-health.json` | Persistence write errors, telemetry freshness, app scrape state, active model state, Redis/ClickHouse/MLflow/MQTT status text. |
| `pumpad.json` | Pipeline row linking RouteMQ dispatch to inference and persistence outcomes. |

### Files

| File | Change |
|---|---|
| `infra/grafana/dashboards/pumpad-mlops.json` | Add MLOps observability row. |
| `infra/grafana/dashboards/pumpad-system-health.json` | Add system health/SLO row. |
| `infra/grafana/dashboards/pumpad.json` | Add pipeline observability row. |
| `tests/unit/test_observability_config_contract.py` | Assert dashboard queries reference real metrics. |

### Acceptance Criteria

- Dashboard contract tests find all new metric names in JSON.
- No upgraded panel uses fake `0 * sum(...)`-style placeholder expressions.
- Existing dashboard UIDs remain stable:
  - `pumpad-observability`
  - `pumpad-mlops`
  - `pumpad-system-health`
  - `pumpad-mqtt-broker`

---

## Phase 4 — Streamlit Operator Observability

### Objective

Make the operator console explain the same observability state in plain language.

### Files

| File | Change |
|---|---|
| `dashboard/data.py` | Add observability summary helpers and freshness readers. |
| `dashboard/widgets.py` | Extend global status banner with severity and actionable failed-check details. |
| `dashboard/pages/0_overview.py` | Add `Observability Snapshot` cards. |
| `dashboard/pages/4_drift_reports.py` | Show drift age, stale warning, and retrain linkage. |
| `dashboard/pages/5_system_health.py` | Show app metric freshness and service checks. |
| `dashboard/pages/6_runbook.py` | Add metric-driven triage steps. |
| `tests/unit/test_dashboard_data_contract.py` | Test new helpers with fakes. |
| `tests/unit/test_dashboard_widgets_contract.py` | Test GREEN/DEGRADED/RED logic. |
| `tests/unit/test_dashboard_pages_contract.py` | Assert new section labels exist. |

### Acceptance Criteria

- Overview page includes `Observability Snapshot`.
- Fake stale telemetry produces `RED` or `DEGRADED` status depending on severity.
- Fake service failures produce actionable text.
- Dashboard tests remain offline and do not require live services.

---

## Phase 5 — MLflow & Evidently Evidence Enrichment

### Objective

Turn drift and retraining outputs into stronger MLOps evidence.

### Files

| File | Change |
|---|---|
| `ml/monitoring/drift_check.py` | Add optional drift summary fields if safely available. |
| `app/jobs/drift_report_job.py` | Add `timestamp`, `method`, `threshold`, `n_features`, `n_drifted`, optional `report_path`. |
| `app/jobs/retraining_job.py` | Add `started_at`, `finished_at`, `duration_seconds`, `success`, `version`, `run_id`, `error`. |
| `ml/registry/mlflow_client.py` | Add stable observability traceability tags. |
| `tests/unit/test_ml_monitoring_contract.py` | Assert drift summary shape. |
| `tests/unit/test_mlops_jobs_contract.py` | Assert retrain/drift payload metadata. |
| `tests/unit/test_mlflow_registry_contract.py` | Assert traceability tags with fake client. |

### Acceptance Criteria

- Drift Redis payload includes `timestamp`, `method="evidently"`, `drift_share`, `n_drifted`, `n_features`.
- Retrain Redis payload includes `started_at`, `finished_at`, `duration_seconds`, `success`, `promoted`, `reason`.
- MLflow fake client receives an observability schema/version tag.
- Existing MLflow dataset/signature/system metric tests continue to pass.

---

## Phase 6 — Demo Evidence & Regression Tests

### Objective

Make the observability story repeatable for screenshots, report evidence, and LinkedIn.

### Files

| File | Change |
|---|---|
| `scripts/run_e2e_demo.py` | Add optional observability evidence checks after existing demo phases. |
| `scripts/capture_screenshots.py` | Add target labels for upgraded Grafana/Streamlit panels. |
| `docs/presentation/screenshot-checklist.md` | Add observability capture sequence. |
| `tests/unit/test_scripts_contract.py` | Assert new screenshot targets and demo labels. |

### Acceptance Criteria

- Screenshot target list includes upgraded Grafana and Streamlit observability panels.
- Unit tests verify labels/targets offline.
- Optional live QA can run only when services are intentionally started.

---

## Phase 7 — Docs, Runbook & Portfolio Packaging

### Objective

Turn the technical work into a portfolio narrative.

### Files

| File | Change |
|---|---|
| `README.md` | Add concise observability evidence section and link to this spec. |
| `infra/README.md` | Remove stale note contradicting current metrics server wiring. |
| `docs/presentation/screenshot-checklist.md` | Link upgraded evidence sequence. |
| `dashboard/pages/6_runbook.py` | Keep operator runbook text aligned with final signals. |
| `docs/laporan/laporan-akhir.md` | Optionally update observability limitations/future work language. |

### Acceptance Criteria

- README clearly says what observability is implemented.
- Docs distinguish implemented local observability from future hardening.
- Portfolio narrative is honest: no cloud/Kubernetes/PagerDuty claims.

---

## 6. Dependency Graph

```text
Phase 1: Metric taxonomy
  ↓
Phase 2: Backend instrumentation
  ↓              ↘
Phase 5: ML evidence  → Phase 3: Grafana dashboards
  ↓                    ↓
Phase 4: Streamlit operator observability
  ↓
Phase 6: Demo evidence
  ↓
Phase 7: Portfolio packaging
```

Recommended execution order:

1. Phase 1
2. Phase 2 and Phase 5
3. Phase 3 and Phase 4
4. Phase 6
5. Phase 7

---

## 7. Testing Strategy

| Stage | Command |
|---|---|
| Metric contract | `uv run pytest tests/unit/test_observability_metrics_contract.py` |
| Backend signals | `uv run pytest tests/unit/test_anomaly_controller_contract.py tests/unit/test_persistence_contract.py tests/unit/test_mlops_jobs_contract.py` |
| Dashboard contracts | `uv run pytest tests/unit/test_observability_config_contract.py tests/unit/test_dashboard_data_contract.py tests/unit/test_dashboard_widgets_contract.py tests/unit/test_dashboard_pages_contract.py` |
| ML evidence | `uv run pytest tests/unit/test_ml_monitoring_contract.py tests/unit/test_mlflow_registry_contract.py` |
| Scripts | `uv run pytest tests/unit/test_scripts_contract.py` |
| Full unit suite | `uv run pytest tests/unit` |
| Lint | `uv run ruff check .` |
| Optional live demo | `make demo-fast` |
| Optional screenshots | `make screenshots TAG=observability-upgrade` (`TAG` aliases `SCREENSHOT_TAG`) |

All unit tests must remain offline-friendly.

---

## 8. Risk Register

| Risk | Phase | Mitigation |
|---|---|---|
| Metric label cardinality grows too high | 1–2 | Keep labels bounded: station, model_version, result, target, severity. Do not label by event ID or timestamp. |
| Grafana JSON becomes fragile | 3 | Add strict config contract tests and keep UIDs stable. |
| Evidently output shape differs by version | 5 | Treat per-feature details as optional; keep aggregate drift fields required. |
| Streamlit probes depend on local host ports | 4 | Use configurable URLs/ports and fake them in tests. |
| Retraining job metadata changes break dashboard | 5 | Add payload contract tests before dashboard changes. |
| Demo checks become service-dependent | 6 | Keep unit tests offline; live checks are optional. |

---

## 9. Commit Strategy

Recommended atomic commits:

| Commit | Contents |
|---|---|
| `docs: add portfolio observability upgrade spec` | This plan/spec only. |
| `test: define observability metric contracts` | Metric/dashboard/data tests first. |
| `feat: extend pumpad observability metrics` | `app/observability/*` and green metric tests. |
| `feat: instrument inference and persistence signals` | Controller/persistence changes and focused tests. |
| `feat: enrich drift and retrain evidence` | Drift/retrain/MLflow payload changes and tests. |
| `feat: upgrade grafana observability dashboards` | Grafana JSON and config contract tests. |
| `feat: upgrade streamlit operator observability` | Dashboard data/widgets/pages and tests. |
| `test: add observability demo evidence checks` | Script contracts and optional live evidence labels. |
| `docs: update observability portfolio narrative` | README, infra note, screenshot checklist, runbook wording. |

Do not commit generated screenshots together with code changes. Keep screenshot evidence as a separate final artifact update if needed.

---

## 10. Final Success Criteria

The portfolio observability upgrade is complete when:

1. `tests/unit/test_observability_metrics_contract.py` passes with the new metric taxonomy.
2. Backend tests prove inference, persistence, drift, and retraining signals are emitted.
3. Grafana dashboards reference the new real metrics.
4. Streamlit shows a human-readable observability snapshot and runbook guidance.
5. Drift and retrain payloads include timestamp/duration/result evidence.
6. Screenshot checklist includes upgraded observability targets.
7. README can honestly claim local industrial-style observability without overclaiming production deployment.

Recommended portfolio headline:

> Industrial-style observability for an ML anomaly detection system: Prometheus metrics, local alert rules, Grafana dashboards, SLO-style health, telemetry freshness/data-quality monitoring, drift/retrain evidence, MLflow registry, and operator runbooks.
