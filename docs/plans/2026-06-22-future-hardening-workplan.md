# Future Hardening Workplan

**Project:** PDAM Pump Sentinel  
**Date:** 2026-06-22  
**Status:** Draft implementation workplan  
**Goal:** Turn the remaining honest future-work gaps into sequenced, testable implementation slices without changing the current MVP/demo framing.

---

## 1. Executive Summary

PDAM Pump Sentinel already has the core MVP/demo flow: MQTT telemetry intake, synchronous anomaly inference, Redis/ClickHouse persistence, MLflow registry integration, drift/retraining jobs, Prometheus metrics, Grafana dashboards, and Streamlit operator pages. The remaining work in this plan is **hardening and roadmap work**, not a blocker for the current demo.

The current deployment-preferred SKAB baseline remains `02_pca_spectral` with held-out `test_f1 = 0.5583306581059391`. Supervised candidates such as XGBoost and LightGBM must stay framed as label-supervised future candidates until durable two-class operator labels and held-out validation evidence exist.

The safest execution path is:

1. Add a synchronous MLflow alias refresh core before adding polling.
2. Add durable operator labels before label-supervised retraining.
3. Add live ClickHouse rolling-window retraining with PCA first.
4. Add LSTM-AE and supervised candidates only after data/label contracts are stable.
5. Harden observability with opt-in local alert routing, tracing, logs, and E2E evidence.
6. Update docs after behavior exists.

---

## 2. Current Baseline

| Capability | Current State | Evidence / File Area |
|---|---|---|
| Inference singleton | Process-local service loaded at cold start | `app/services/inference.py` |
| Runtime hot-swap | Retraining path can call `set_inference_service` | `app/jobs/retraining_job.py` |
| MLflow champion loading | Cold-start champion alias/fallback exists | `ml/registry/mlflow_client.py` |
| Alias polling | Not implemented for long-running app processes | `bootstrap/app.py`, `ml/monitoring/scheduler.py` |
| Operator labels | Ack/mute/note exists, durable true/false labels do not | controller/persistence/dashboard areas |
| Retraining source | PCA/SKAB-path oriented | `app/jobs/retraining_job.py`, `ml/training/train_pca.py` |
| Supervised training | Trainer exists, not deployment-safe without labels | `ml/training/train_supervised.py` |
| Observability | Local Docker Compose stack exists | `infra/`, `app/observability/metrics.py` |
| Hardening gaps | External routing, tracing, centralized logs, E2E hardening | `infra/prometheus/`, `infra/grafana/`, `docs/plans/2026-06-08-portfolio-observability-upgrades.md` |

---

## 3. Scope

### 3.1 In Scope

- MLflow alias metadata reading and safe runtime inference refresh.
- Optional scheduler-driven alias polling after the refresh core is proven.
- Durable operator label intake and ClickHouse label storage.
- Live ClickHouse rolling-window dataset extraction with chronological splits.
- PCA-first live retraining path.
- LSTM-AE live retraining only after rolling-window contracts are stable.
- Supervised XGBoost/LightGBM only after durable two-class label evidence exists.
- Persistent MLOps evidence for model refresh, retraining, and labels where needed.
- Opt-in local alert routing, tracing, structured logs, and Grafana/Prometheus evidence panels.
- Offline-first unit tests plus optional live-stack E2E checks.
- Honest README/report/plan updates after implementation.

### 3.2 Out of Scope

- Cloud/Kubernetes production migration.
- Enterprise incident management claims.
- PagerDuty or external SaaS alerting unless explicitly configured and verified.
- Treating supervised models as deployment-safe without durable two-class labels.
- Replacing the current PCA spectral champion without same-protocol evidence.
- Speculative backward compatibility for unreleased internal shapes.

---

## 4. Execution Model

| Wave | Work | Dependency |
|---|---|---|
| Wave 1A | Alias refresh core | None, but needs an alias/manual promotion path for E2E proof |
| Wave 1B | Alias polling scheduler and bootstrap env gate | Wave 1A |
| Wave 2 | Durable operator labels | None; can start after Wave 1A if teams split work |
| Wave 3A | Rolling-window extraction and PCA live retraining | Wave 2 not required for PCA, but label schema should be known |
| Wave 3B | LSTM-AE live retraining | Wave 3A |
| Wave 3C | Supervised retraining candidates | Wave 2 plus durable two-class labels |
| Wave 4 | Observability hardening | Stable metrics/evidence contracts from Waves 1-3 |
| Wave 5 | E2E hardening and docs | Waves 1-4 |

---

## 5. Work Packages

## WP1A - MLflow Alias Refresh Core

### Objective

Add a synchronous refresh path that can detect a changed MLflow `champion` alias, fully load the new inference service, and atomically publish it while keeping the old service active on failure.

### Files

| File | Work |
|---|---|
| `ml/registry/mlflow_client.py` | Add alias metadata helper returning model name, alias, version, run id, and artifact/provenance fields without downloading artifacts. |
| `app/services/inference.py` | Add refresh helper around current singleton and `set_inference_service`. |
| `tests/unit/test_mlflow_registry_contract.py` | Fake MLflow alias metadata contract tests. |
| `tests/unit/test_inference_loader_contract.py` | Changed/unchanged/failure/concurrent read tests for refresh behavior. |

### Acceptance Criteria

| Check | Acceptance Criteria |
|---|---|
| Metadata read | Alias metadata can be read with a fake MLflow client and no artifact download. |
| Unchanged alias | Refresh is a no-op when alias version/run id has not changed. |
| Changed alias | New service is fully loaded before singleton replacement. |
| Failure retention | Any metadata/load failure keeps the incumbent service active and logs/counts the failure. |
| Atomic publish | In-flight reads never observe a partially constructed service. |
| Provenance | Active model info preserves available local metadata and records MLflow version/run id when available. |

### Verification

```bash
uv run pytest tests/unit/test_mlflow_registry_contract.py tests/unit/test_inference_loader_contract.py
uv run ruff check ml/registry/mlflow_client.py app/services/inference.py tests/unit/test_mlflow_registry_contract.py tests/unit/test_inference_loader_contract.py
```

### Notes

- This is the recommended first implementation slice.
- Do not add polling in this slice.
- Manual promotion of an MLflow alias is acceptable for E2E proof if automatic promotion is not wired yet.

## WP1B - Alias Polling Scheduler

### Objective

Run the proven refresh core periodically in long-running app processes behind an environment flag.

### Files

| File | Work |
|---|---|
| `ml/monitoring/scheduler.py` | Add `ModelRefreshScheduler` using interval trigger, `coalesce=True`, and `max_instances=1`. |
| `bootstrap/app.py` | Start/stop scheduler behind an env flag. |
| `app/observability/metrics.py` | Add bounded refresh success/error/staleness metrics if needed. |
| `tests/unit/test_retrain_scheduler_contract.py` | Scheduler dispatch and shutdown contract tests. |
| `tests/unit/test_bootstrap_worker_contract.py` | Env-gated startup/shutdown tests. |

### Acceptance Criteria

| Check | Acceptance Criteria |
|---|---|
| Env gate | Polling is disabled by default and enabled explicitly. |
| Safe cadence | Polling does not overlap refresh executions. |
| Failure behavior | Registry errors do not kill the app. |
| Visibility | Last success/failure and stale refresh state are observable in logs or bounded metrics. |

### Verification

```bash
uv run pytest tests/unit/test_retrain_scheduler_contract.py tests/unit/test_bootstrap_worker_contract.py
uv run ruff check ml/monitoring/scheduler.py bootstrap/app.py app/observability/metrics.py
```

## WP2 - Durable Operator Labels

### Objective

Persist operator true/false anomaly feedback in ClickHouse so future retraining can join labels to telemetry windows.

### Files

| File | Work |
|---|---|
| `infra/clickhouse/init.sql` | Add `operator_labels` table. |
| `app/services/persistence.py` | Add label persistence helper. |
| `app/controllers/anomaly_controller.py` | Add or delegate label intake handling. |
| `app/routers/telemetry.py` | Add label topic route if MQTT is the chosen intake path. |
| `tests/unit/test_persistence_contract.py` | Fake ClickHouse label insert tests. |
| `tests/unit/test_anomaly_controller_contract.py` | Valid/invalid label intake tests. |

### Acceptance Criteria

| Check | Acceptance Criteria |
|---|---|
| Schema | Label rows store station, source timestamp, label, source/operator metadata, created time, and optional reason. |
| Joinability | Labels can be joined to telemetry/anomaly windows by station and time. |
| Idempotency | Repeated label id or natural key policy is deterministic. |
| Bounded metrics | No free-text reason, operator id, event id, or raw timestamp becomes a Prometheus label. |
| Offline tests | Unit tests use fakes and do not require ClickHouse running. |

### Verification

```bash
uv run pytest tests/unit/test_persistence_contract.py tests/unit/test_anomaly_controller_contract.py
uv run ruff check app/services/persistence.py app/controllers/anomaly_controller.py app/routers/telemetry.py
```

### Manual Check

1. Start local stack.
2. Confirm `operator_labels` exists in ClickHouse.
3. Send one valid label event through the chosen intake path.
4. Query the row back from ClickHouse.

## WP3A - Rolling Windows and PCA Live Retraining

### Objective

Move retraining from SKAB-path-only toward live ClickHouse windows while preserving PCA as the first production-wired family.

### Files

| File | Work |
|---|---|
| `ml/datasets/live_windows.py` | New module for ClickHouse query, ordering, window construction, and chronological splits. |
| `app/jobs/retraining_job.py` | Add data-source selection and live-window retraining mode. |
| `ml/training/train_pca.py` | Accept prebuilt/live windows while preserving existing SKAB contract. |
| `ml/registry/mlflow_client.py` | Log dataset source, window bounds, split metadata, and provenance tags. |
| `tests/unit/test_mlops_jobs_contract.py` | Retraining payload/source/promotion tests. |
| `tests/unit/test_train_pca_contract.py` | PCA live-window contract tests. |

### Acceptance Criteria

| Check | Acceptance Criteria |
|---|---|
| Rolling extraction | Live ClickHouse rows become ordered fixed-size windows. |
| Chronological split | Train/validation/test boundaries do not leak future rows backward. |
| Existing path | SKAB-path retraining still works. |
| PCA first | PCA live-window retraining is the first supported live path. |
| Evidence | Retrain payload and MLflow tags record source, time range, split, and model family. |

### Verification

```bash
uv run pytest tests/unit/test_mlops_jobs_contract.py tests/unit/test_train_pca_contract.py
uv run ruff check app/jobs/retraining_job.py ml/training/train_pca.py ml/registry/mlflow_client.py ml/datasets/live_windows.py
```

## WP3B - LSTM-AE Live Retraining

### Objective

Add LSTM-AE live-window support after the shared rolling-window contract is stable.

### Files

| File | Work |
|---|---|
| `ml/training/train_lstm_ae.py` | Add compatible live-window input path. |
| `app/jobs/retraining_job.py` | Add guarded LSTM-AE family selection. |
| `tests/unit/test_lstm_ae_training_contract.py` | Live-window input and result contract tests. |

### Acceptance Criteria

| Check | Acceptance Criteria |
|---|---|
| Contract reuse | LSTM-AE consumes the same rolling-window data contract as PCA. |
| Honest promotion | LSTM-AE is a challenger unless same-protocol evidence beats the champion gate. |
| Default safety | LSTM-AE does not become the default live retraining path merely by existing. |

### Verification

```bash
uv run pytest tests/unit/test_lstm_ae_training_contract.py tests/unit/test_mlops_jobs_contract.py
uv run ruff check ml/training/train_lstm_ae.py app/jobs/retraining_job.py
```

## WP3C - Label-Supervised Retraining Candidates

### Objective

Enable supervised models only when durable labels provide valid two-class evidence.

### Files

| File | Work |
|---|---|
| `ml/training/train_supervised.py` | Enforce durable label and two-class evidence gates. |
| `app/jobs/retraining_job.py` | Add supervised family selection and honest skip payloads. |
| `ml/registry/mlflow_client.py` | Log label counts, class balance, split metadata, and validation evidence. |
| `tests/unit/test_supervised_training_contract.py` | Normal-only skip, two-class run, and promotion-gate tests. |

### Acceptance Criteria

| Check | Acceptance Criteria |
|---|---|
| Normal-only labels | XGBoost/LightGBM skip honestly when only class `[0]` exists. |
| Two-class labels | Supervised trainer runs only with durable two-class labels. |
| Promotion gate | Supervised candidate is not deployment-safe without held-out label-supervised evidence. |
| Report payload | Results state source, label counts, class balance, and skip reason. |

### Verification

```bash
uv run pytest tests/unit/test_supervised_training_contract.py tests/unit/test_mlops_jobs_contract.py tests/unit/test_mlflow_registry_contract.py
uv run ruff check ml/training/train_supervised.py app/jobs/retraining_job.py ml/registry/mlflow_client.py
```

## WP4 - Observability Hardening

### Objective

Add opt-in local production-style observability around refresh, label intake, retraining, and pipeline correlation.

### Files

| File | Work |
|---|---|
| `infra/docker-compose.dev.yml` | Add opt-in local services/config for alert routing, tracing, and logs if chosen. |
| `infra/prometheus/prometheus.yml` | Add local routing/scrape configuration as needed. |
| `infra/prometheus/rules/pumpad-alerts.yml` | Add alerts for refresh failures, stale label intake, retrain errors, and evidence freshness. |
| `infra/grafana/dashboards/` | Add panels for refresh, labels, retraining evidence, alerts, traces/log references. |
| `bootstrap/app.py` | Initialize tracing/log correlation behind env flags. |
| `app/controllers/anomaly_controller.py` | Add bounded correlation context around ingest, inference, publish, persistence. |
| `app/middleware/correlation.py` | Extend correlation logging context where useful. |
| `app/jobs/retraining_job.py` | Add retraining trace/log evidence. |
| `tests/unit/test_observability_config_contract.py` | Dashboard/config/alert contract tests. |
| `tests/unit/test_bootstrap_worker_contract.py` | Env-gated tracing/logging init tests. |

### Acceptance Criteria

| Check | Acceptance Criteria |
|---|---|
| Local-only framing | Implementation and docs call this local opt-in hardening, not cloud production. |
| Alert routing | A local receiver-compatible route exists and is testable when enabled. |
| Trace/log correlation | One telemetry message can be correlated through ingest, inference, and persistence where enabled. |
| Dashboard truth | Grafana panels reference real metrics, not placeholders. |
| Cardinality safety | No event id, operator id, free-text reason, or raw timestamp is used as a metric label. |

### Verification

```bash
uv run pytest tests/unit/test_observability_config_contract.py tests/unit/test_bootstrap_worker_contract.py tests/unit/test_anomaly_controller_contract.py
uv run ruff check bootstrap/app.py app/controllers/anomaly_controller.py app/middleware/correlation.py app/jobs/retraining_job.py
```

### Manual Check

1. Start the opt-in local stack.
2. Trigger one telemetry message and one known alert condition.
3. Confirm alert routing target receives the alert.
4. Confirm correlation id appears across logs/traces where enabled.
5. Confirm Grafana panels populate after demo traffic.

## WP5 - E2E and Production-Style Hardening

### Objective

Prove the integrated hardening path with offline tests first and optional live-stack checks second.

### Files

| File | Work |
|---|---|
| `scripts/run_e2e_demo.py` | Add optional phases for label intake, model refresh, rolling retrain, and observability evidence. |
| `tests/unit/test_scripts_contract.py` | Offline script flag/phase tests. |
| `tests/integration/test_mlflow_round_trip.py` | Extend alias refresh integration only if stable. |
| `infra/docker-compose.dev.yml` | Tighten health checks/profiles/resource-safe defaults if needed. |

### Acceptance Criteria

| Check | Acceptance Criteria |
|---|---|
| Offline first | Unit tests do not require Docker, Redis, ClickHouse, MLflow, Prometheus, or Grafana. |
| Live opt-in | Stack E2E checks are skipped unless explicitly enabled. |
| Clear failures | Missing services fail with actionable precondition messages. |
| Limited cleanup | Demo cleanup is explicit and limited to demo keys/tables. |

### Verification

```bash
uv run pytest tests/unit
uv run ruff check .
```

Optional live check:

```bash
make demo-fast DEMO_EXTRA_ARGS="--observability-evidence"
```

## WP6 - Documentation and Report Framing

### Objective

Update project documentation after implementation lands so the narrative remains honest and evidence-based.

### Files

| File | Work |
|---|---|
| `README.md` | Update implemented vs future limitations. |
| `infra/README.md` | Explain optional hardening services and local-stack boundaries. |
| `docs/plans/2026-06-08-portfolio-observability-upgrades.md` | Update status of hardening items. |
| `docs/plans/2026-06-09-skab-extra-anomaly-experiments.md` | Preserve PCA/supervised caution if referenced. |
| `docs/laporan/laporan-akhir.md` | Update report language only after behavior exists. |
| `artifacts/skab-model-experiments/summary.md` | Preserve supervised-model caution if regenerated. |

### Acceptance Criteria

| Check | Acceptance Criteria |
|---|---|
| Honest status | Current MVP/demo remains described as mostly implemented. |
| Hardening status | New work is described as hardening completion, not original blocker repair. |
| Supervised caution | XGBoost/LightGBM are not deployment-safe without durable label-supervised evidence. |
| Local boundary | Alert routing/tracing/logging claims match actual local implementation. |

### Verification

```bash
rg -n "PagerDuty|Kubernetes|cloud production|deployment-safe|production-ready" README.md docs infra artifacts
```

Review each hit and keep only claims that are supported by implemented behavior.

---

## 6. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Alias exists only as cold-start fallback | Polling is built around a static target | Make manual/existing alias promotion a WP1 precondition and test fixture. |
| Singleton swap races with inference reads | Intermittent production failures | WP1A must test atomic publish and in-flight read behavior. |
| MLflow helper swallows failures | Refresh appears healthy when stale | Log/count failures while keeping old service active. |
| Incumbent metrics come from local metadata | Incorrect champion comparison after alias refresh | Preserve local metadata fallback and record MLflow version/run id when present. |
| Sparse labels | Supervised training overclaims | Require durable two-class evidence and honest skip payloads. |
| Data leakage in live retraining | Inflated evaluation metrics | Chronological splits and leakage tests before model work. |
| Observability tool sprawl | Demo becomes brittle | Keep tracing/logs/routing opt-in and local. |
| High-cardinality metrics | Prometheus instability | Never put event ids, operator ids, free text, or timestamps in metric labels. |
| Docs overclaim hardening | Portfolio credibility risk | Update docs only after behavior lands and keep local-stack wording. |

---

## 7. First Implementation Slice

Start with **WP1A - MLflow Alias Refresh Core**.

### Why This Slice First

| Reason | Detail |
|---|---|
| Smallest safe unit | It isolates alias change detection and atomic swap before timing/scheduler complexity. |
| High value | Long-running app processes can refresh without restart once scheduler arrives. |
| Offline testable | Fake MLflow client and fake service loader cover the important behavior. |
| Foundation | Future retraining promotions become useful for already-running processes. |

### Implementation Steps

1. Add alias metadata contract tests using fake MLflow responses.
2. Add inference refresh tests for changed alias, unchanged alias, load failure, and concurrent read during swap.
3. Implement metadata helper in `ml/registry/mlflow_client.py`.
4. Implement synchronous refresh helper in `app/services/inference.py`.
5. Run focused tests and ruff for touched files.
6. Only then proceed to WP1B polling scheduler.

---

## 8. Definition of Done

| Area | Done When |
|---|---|
| Runtime refresh | Running app can update inference service when MLflow `champion` alias changes. |
| Labels | Operator labels are stored durably in ClickHouse and queryable. |
| Rolling retrain | Retraining can use live ClickHouse windows with chronological splits. |
| Supervised models | Supervised candidates run/promote only with durable two-class labels and held-out validation evidence. |
| Observability | Refresh, labels, retraining, alerts, traces/logs, and evidence are visible through opt-in local tooling. |
| E2E | Offline tests pass and optional live demo proves the integrated hardening path. |
| Docs | README/report/plans clearly separate implemented MVP/demo from hardening and future limits. |
