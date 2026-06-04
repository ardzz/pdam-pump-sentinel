# Sprint — Remaining Tasks

Status snapshot grounded in `git log` (HEAD `8a3ccd2`, in sync with `origin/main`) and the roadmap in `design.md §11`. This is a living checklist; update as items land.

## Done (verified)

- **Week 1–2** — Foundation, infra stack, SKAB ingestion pipeline, dashboard v0.
- **Week 3 (ML baseline, core)** — Window features (raw/spectral/enriched), PCA T²/Q champion, LSTM-AE challenger, event-based eval metrics, MLflow tracking integrated.
- **Week 4 (MLOps loop)** — Registered-model loader + hot-swap, Evidently drift, scheduled retraining, champion-challenger promotion gate (`should_promote`, `f1_margin=0.02`, `far_guard=1.05`), drift injector.
- **Week 5 (partial)** — Prometheus metrics from RouteMQ hooks + Grafana dashboards (system + ML). Commits `f4981fd`, `832303e`, `03a7156`.
- **Extra model families** — Isolation Forest (unsupervised) + supervised XGBoost/LightGBM, family-aware loader dispatch.
- **MLflow bugfixes** — Champion alias resolution under MLflow 3.12 (`registered_model_version` is `None`); reload from run artifacts not sklearn-flavor dir. Verified round-trip live on local sqlite (`7ced737`).

## Remaining

### Week 3 tail — seed a LIVE MLflow

- [ ] Run `scripts/seed_initial_models.py` against the running MLflow at `:5000` (it already sets `log_mlflow=True`, `register_model=True`, `alias='champion'`).
- [ ] Confirm in the MLflow UI: experiments visible + registered model with `@champion` alias.
- [ ] Verify `load_champion_service` round-trip against the **server** (not just local sqlite).

### Week 5 — Observability + Polish (remainder)

- [ ] Slide deck → `docs/presentation/` (follow §13 demo storyboard, 13–14 min).
- [ ] ADR docs → `docs/adr/` (key decisions: honest-eval split strategy, no point-adjustment, champion-challenger gate, MLflow alias pattern).
- [ ] Demo script automation (one-command live-demo runner per §13.1 T+0…T+8).
- [ ] Proposal final pass (`docs/proposal.md`).
- [ ] README polish.

### Week 6 — Buffer

- [ ] Bug fixing.
- [ ] Final documentation.
- [ ] Presentation practice (timed dry-run).
- [ ] Backup demo video recording.

## Parked technical follow-ups

- [ ] **Permanent port remap** — add `infra/.env` (gitignored) with remapped host ports; switch Makefile `dev`/`dev-down`/`ps`/`logs` to `--env-file infra/.env`; drop the unused `mysql` service (`ENABLE_MYSQL=false`). Current remap is per-`up` only and plain `make dev` re-conflicts on `8123`.
- [ ] Add `.playwright-mcp/` to `.gitignore` (stray browser-session artifacts).
- [ ] **Reproducible supervised split** — commit a manifest or `make` target for the in-distribution datetime-sorted chrono 80/20 split; log supervised XGBoost/LightGBM runs to MLflow for the report.
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
