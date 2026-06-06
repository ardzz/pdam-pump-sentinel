# Sprint — Remaining Tasks

Status snapshot grounded in `git log` (HEAD `1322fb3`, ahead of `origin/main` by 3 commits pending push) and the roadmap in `design.md §11`. This is a living checklist; update as items land.

## Done (verified)

- **Week 1–2** — Foundation, infra stack, SKAB ingestion pipeline, dashboard v0.
- **Week 3 (ML baseline, core)** — Window features (raw/spectral/enriched), PCA T²/Q champion, LSTM-AE challenger, event-based eval metrics, MLflow tracking integrated.
- **Week 3 tail — live MLflow seeded** — `scripts/seed_initial_models.py` ran against the server at `:5000`; registered model `PumpAD` with `@champion → v1`; `load_champion_service` round-trip confirmed against the server (`7ced737` for the underlying alias-resolution fix; verified live this sprint).
- **Week 4 (MLOps loop)** — Registered-model loader + hot-swap, Evidently drift, scheduled retraining (APScheduler), champion-challenger promotion gate (`should_promote`, `f1_margin=0.02`, `far_guard=1.05`), drift injector.
- **Week 5 (partial)** — Prometheus metrics from RouteMQ hooks + Grafana dashboards (system + ML). Commits `f4981fd`, `832303e`, `03a7156`.
- **Extra model families** — Isolation Forest (unsupervised) + supervised XGBoost/LightGBM, family-aware loader dispatch.
- **MLflow tracking maximized for the report** — Cross-family comparison experiment `pump_sentinel_model_comparison` with consistent tags (`model_family`, `feature_mode`, `split_strategy`, `dataset`, `report_run`); per-iteration live curves (LSTM-AE per-epoch `loss`/`val_loss`; XGB `val_xgb_logloss_round`; LGBM `val_lgbm_binary_logloss_round`); SKAB dataset cards via `mlflow.data` with stable digest per manifest; system metrics autolog (CPU/memory/disk); run-traceability tags (`git.commit.sha`, `git.branch`, `git.is_dirty`, `python.version`, `host.name`, `package.{mlflow,scikit-learn,tensorflow,xgboost,lightgbm}`); model signatures + input examples on each `*_model/` artifact. Commits `e9fd362`, `f9ade52`, `16a1ba8`, `47126ba`, `894e7db`, `2ac6f9a`.
- **MLflow bugfixes** — Champion alias resolution under MLflow 3.12 (`registered_model_version` is `None`); reload from run artifacts not sklearn-flavor dir (`7ced737`). XGB `MlflowXGBCallback` lifted to module-level and made picklable (`894e7db`). System metrics enabled before the comparison script opens its first run (`2ac6f9a`).
- **Permanent port remap** — `infra/.env` (gitignored) with remapped host ports; Makefile `dev`/`dev-down`/`ps`/`logs` use `--env-file infra/.env`; unused `mysql` service dropped from dev compose (`427989f`, `95664c6`).
- **`.playwright-mcp/` gitignored** (`6a661e2`); **stray `mlflow.db` gitignored** (`9e52b78`).
- **Cross-family comparison report serving** — `make mlflow-report` target serves comparison runs from `data/mlflow_live.db` on port 5050 in parallel with docker MLflow on `:5000` (`d514ad8`).
- **Presentation supporting material** — `docs/presentation/labeling-strategy-notes.md` 121-line speaker notes with industry citations (6 vendors), peer-reviewed academic citations (4 papers), 13 repo proof-points, 5 honest gaps, and Indonesian speaker script (`e714009`).
- **Demo-blocker P0 fixes** (surfaced from this sprint's RouteMQ ↔ ML integration audit):
  - `46bc0c3` — `get_inference_service()` falls back to MLflow `load_champion_service('PumpAD','champion')` when `PUMPAD_MODEL_DIR` is empty/invalid; `MLFLOW_TRACKING_URI=http://mlflow:5000` set on app service in dev compose. Partial mitigation for the §5 "alias polling" gap — cold start is now wired; runtime polling still TODO.
  - `8788e83` — Persistence writes anomaly observation as a single ClickHouse row `measurement='anomaly_score'`, `value_float=score`, `value_int=flag`. Dashboard live-sensors + anomaly-history charts now actually render.
  - `dd38b06` — Retraining job + seed script write `name` / `version` / `activated_at` (additive) on `pumpad:active:model` Redis key. Dashboard Model Registry page no longer shows N/A.
- **Smoke L1 verification — end-to-end pipeline confirmed live** (host bootstrap.app, fresh docker stack, replay → controller → PCA inference → MQTT publish → Redis + ClickHouse). Evidence captured: `pumpad:latest:anomaly:ipa_01` carrying real PCA T²/Q with `model_version="1"`; 3× `anomaly_score` rows in `telemetry_observations`; `pumpad:active:model` populated with dashboard-aligned fields. P0-A `load_champion_service` fallback exercised live — app downloaded artifacts on first message. Fixes landed during smoke:
  - `f4d994c` — Aligned docker MLflow image to `v3.12.0` (matching host 3.12 client) and enabled `--serve-artifacts` + `--artifacts-destination` + `--default-artifact-root mlflow-artifacts:/` so external clients can upload artifacts via HTTP proxy instead of the container-local filesystem path that triggered `PermissionError: '/mlflow'`.
  - `fce3fad` + `a836a0c` — `scripts/seed_initial_models.py` now calls `redis_manager.initialize()` + `disconnect()` around its standalone Redis write, gated on the raw `enabled` attribute rather than `is_enabled()` (which is `enabled AND _redis_client is not None`, so it short-circuited the whole init chain pre-init on the first iteration of the fix). Re-run smoke after `a836a0c` confirmed `pumpad:active:model` populates with `name` / `version` / `activated_at` directly from the seed; `FakeRedisManager` stub updated to track the lifecycle calls.
  - `6f4123a` — `make replay` no longer crashes for missing `--input`; new `REPLAY_INPUT` / `REPLAY_STATION` / `REPLAY_HOST` / `REPLAY_PORT` / `REPLAY_LIMIT` / `REPLAY_EXTRA_ARGS` Make variables, default fixture `tests/fixtures/skab_tiny.csv`.
  - `bfa5ecf` — `infra/.env.example` documents the host port remap variables (`MQTT_PORT`, `CLICKHOUSE_*`, `APP_METRICS_PORT`, `PROMETHEUS_PORT`, `GRAFANA_PORT`, `MLFLOW_PORT`) so `infra/.env` overrides are reproducible across workstations.

## Remaining

### Week 5 — Observability + Polish (remainder)

- [ ] Slide deck → `docs/presentation/` (follow §13 demo storyboard, 13–14 min). Include MLflow Compare-Runs screenshots from experiment `pump_sentinel_model_comparison` and the per-iteration curves (LSTM-AE epoch loss, XGB/LGBM logloss-per-round). Pull from `labeling-strategy-notes.md` for industry/academic backing.
- [ ] ADR docs → `docs/adr/` (key decisions: honest-eval split strategy, no point-adjustment, champion-challenger gate, MLflow alias pattern, MLflow Datasets + traceability tags choices).
- [x] **Demo script automation** — `scripts/run_e2e_demo.py` + `make demo` / `make demo-fast` / `make demo-skip-retrain` targets (`0ace5a7` + `1322fb3`). Covers all §13.1 phases T+0..T+8 with per-phase banners, asserts, and PASS/FAIL exit code: T+0 baseline precondition (active model + alias + ClickHouse table); T+1 replay normal; T+2 replay anomalous; T+3 inject drift via `scripts/inject_drift.py`; T+4 Evidently `DriftResult` (drift_share + dataset_drift); T+5 inline `RetrainingJob.handle` (deterministic, no queue worker required); T+6 MLflow run + challenger version verification; T+7 dynamic alias promotion to challenger version; T+8 `set_inference_service(load_champion_service(...))` in-process hot-swap + replay → asserts `Redis pumpad:latest:anomaly:{station}.model_version == challenger`. Verified live against running stack: all 9 phases PASS. Pytest variant under `tests/e2e/` not implemented (architectural follow-up).
- [ ] Proposal final pass (`docs/proposal.md`).
- [ ] README polish — surface `make mlflow-report`, MQTT topics `factory/skab/{station}/telemetry` and `factory/skab/{station}/anomaly`, and known limitations (no auto-refresh, synchronous inference path).

### Week 6 — Buffer

- [ ] Bug fixing.
- [ ] Final documentation.
- [ ] Presentation practice (timed dry-run).
- [ ] Backup demo video recording.

### Architectural follow-ups (post-MVP, honest gaps)

Documented in `docs/presentation/labeling-strategy-notes.md §5` and surfaced from this sprint's RouteMQ ↔ ML audit. Frame these as "future work" in the deck — they are roadmap, not blockers.

From `labeling-strategy-notes.md §5`:

- [ ] **Operator label intake** — `app/controllers/label_controller.py` (NEW) + `infra/clickhouse/init.sql` extend (labels table) + dashboard triage UI on anomaly-history page so operators can confirm/reject anomalies.
- [ ] **Supervised promotion in retraining loop** — `app/jobs/retraining_job.py:76-90` extension to dispatch supervised training when labeled data accumulates.
- [ ] **Supervised training alias setter** — `ml/training/train_supervised.py:816-850` accepts `alias` arg but does not actually set the alias in the MLflow registry. Wire it.
- [ ] **MLflow alias polling in live app** — `app/services/inference.py` add a reload-polling thread so an externally-promoted MLflow `@champion` is picked up without a hot-swap call. (`46bc0c3` added cold-start fallback; runtime polling still missing.)
- [ ] **DriftReportJob APScheduler hook** — `ml/monitoring/scheduler.py` add a drift hook so the drift → retrain chain runs automatically. Today the job exists but is not scheduled, so the chain is manual.

From RouteMQ ↔ ML integration audit (this session):

- [ ] **Streamlit dashboard auto-refresh** — `dashboard/app.py` add `streamlit-autorefresh` or equivalent polling so the operator sees live state without manual reload.
- [ ] **Alert mechanism beyond MQTT publish** — current implementation only publishes to `factory/skab/{station}/anomaly`; add at minimum a structured log alert at controller or persistence layer, optionally Slack/email/webhook for production.
- [ ] **Reconcile Queue → Worker claim** — `README.md` and `design.md` claim a Queue → Worker pattern for anomaly inference; reality is synchronous in the controller (`app/controllers/anomaly_controller.py:18-23`). Either implement async queue dispatch OR update docs to reflect the synchronous flow.

### Parked technical follow-ups

- [ ] **Reproducible supervised in-distribution split** — commit a manifest or `make` target for the chrono 80/20 (Kaggle-comparable) split and add it as an opt-in family in `scripts/train_all_for_comparison.py` so the ~0.90 in-distribution number lands in the same experiment for the report. (The cross-group supervised split is already wired.)
- [ ] **Deferred: live spectral/enriched serving** — extend the inference/observe service to compute spectral/enriched features online so a spectral/enriched/supervised model can be SERVED live. Live champion currently serves raw-feature PCA only; spectral/enriched wins are offline-eval only.
- [ ] Optional: CatBoost + stacked ensemble (4th/5th Kaggle models); prod `docker-compose.yml`.

## Honest evaluation spectrum (report reference)

Report BOTH, each clearly labeled — the F1 number depends entirely on split honesty:

| Setting | Split | F1 | Note |
|---|---|---|---|
| Unsupervised PCA-spectral (champion) | train=normal-only, test=valve2+other | 0.58 | Deployable, novel-fault generalization |
| Supervised cross-group | labeled anomalies in train | 0.60 (AUC 0.937) | Novel-fault generalization |
| Supervised file-level stratified | — | 0.70 | — |
| Supervised in-distribution (chrono 80/20, Kaggle-comparable) | all fault types in train AND test | XGB 0.909 / LGBM 0.905 | Requires labeled examples of every fault type |
| Supervised random-window (max leakage) | — | 0.985 | Not a valid generalization claim |

Do NOT present `~0.90` as novel-fault detection. Never use point-adjustment (inflates; literature-confirmed). SKAB is a surrogate dataset — no real PDAM/benchmark claims.
