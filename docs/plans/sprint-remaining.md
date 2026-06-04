# Sprint — Remaining Tasks

Status snapshot grounded in `git log` (HEAD `2ac6f9a`, ahead of `origin/main`) and the roadmap in `design.md §11`. This is a living checklist; update as items land.

## Done (verified)

- **Week 1–2** — Foundation, infra stack, SKAB ingestion pipeline, dashboard v0.
- **Week 3 (ML baseline, core)** — Window features (raw/spectral/enriched), PCA T²/Q champion, LSTM-AE challenger, event-based eval metrics, MLflow tracking integrated.
- **Week 3 tail — live MLflow seeded** — `scripts/seed_initial_models.py` ran against the server at `:5000`; registered model `PumpAD` with `@champion → v1`; `load_champion_service` round-trip confirmed against the server (`7ced737` for the underlying alias-resolution fix; verified live this sprint).
- **Week 4 (MLOps loop)** — Registered-model loader + hot-swap, Evidently drift, scheduled retraining, champion-challenger promotion gate (`should_promote`, `f1_margin=0.02`, `far_guard=1.05`), drift injector.
- **Week 5 (partial)** — Prometheus metrics from RouteMQ hooks + Grafana dashboards (system + ML). Commits `f4981fd`, `832303e`, `03a7156`.
- **Extra model families** — Isolation Forest (unsupervised) + supervised XGBoost/LightGBM, family-aware loader dispatch.
- **MLflow tracking maximized for the report** — Cross-family comparison experiment `pump_sentinel_model_comparison` with consistent tags (`model_family`, `feature_mode`, `split_strategy`, `dataset`, `report_run`); per-iteration live curves (LSTM-AE per-epoch `loss`/`val_loss`; XGB `val_xgb_logloss_round`; LGBM `val_lgbm_binary_logloss_round`); SKAB dataset cards via `mlflow.data` with stable digest per manifest; system metrics autolog (CPU/memory/disk); run-traceability tags (`git.commit.sha`, `git.branch`, `git.is_dirty`, `python.version`, `host.name`, `package.{mlflow,scikit-learn,tensorflow,xgboost,lightgbm}`); model signatures + input examples on each `*_model/` artifact. Commits `e9fd362`, `f9ade52`, `16a1ba8`, `47126ba`, `894e7db`, `2ac6f9a`.
- **MLflow bugfixes** — Champion alias resolution under MLflow 3.12 (`registered_model_version` is `None`); reload from run artifacts not sklearn-flavor dir (`7ced737`). XGB `MlflowXGBCallback` lifted to module-level and made picklable (`894e7db`). System metrics enabled before the comparison script opens its first run (`2ac6f9a`).
- **Permanent port remap** — `infra/.env` (gitignored) with remapped host ports; Makefile `dev`/`dev-down`/`ps`/`logs` use `--env-file infra/.env`; unused `mysql` service dropped from dev compose (`427989f`, `95664c6`).
- **`.playwright-mcp/` gitignored** (`6a661e2`).

## Remaining

### Week 5 — Observability + Polish (remainder)

- [ ] Slide deck → `docs/presentation/` (follow §13 demo storyboard, 13–14 min). Include MLflow Compare-Runs screenshots from experiment `pump_sentinel_model_comparison` and the per-iteration curves (LSTM-AE epoch loss, XGB/LGBM logloss-per-round).
- [ ] ADR docs → `docs/adr/` (key decisions: honest-eval split strategy, no point-adjustment, champion-challenger gate, MLflow alias pattern, MLflow Datasets + traceability tags choices).
- [ ] Demo script automation (one-command live-demo runner per §13.1 T+0…T+8).
- [ ] Proposal final pass (`docs/proposal.md`).
- [ ] README polish.

### Week 6 — Buffer

- [ ] Bug fixing.
- [ ] Final documentation.
- [ ] Presentation practice (timed dry-run).
- [ ] Backup demo video recording.

## Parked technical follow-ups

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
