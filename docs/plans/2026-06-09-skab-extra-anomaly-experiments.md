# SKAB Extra Anomaly Experiments: Plan & Subagent-Driven Execution Spec

**Project:** PDAM Pump Sentinel  
**Date:** 2026-06-09  
**Status:** Implementation plan with Wave 5 reporting updates  
**Goal:** Extend the honest SKAB model-comparison harness with additional unsupervised and calibration baselines while preserving the current no-leakage evaluation protocol.

---

## 1. Executive Summary

The current strongest honest detector is **PCA spectral** under the manifest protocol: train on normal-only data, calibrate thresholds on validation data, and report final metrics on held-out test data. This plan adds extra anomaly-detection experiments to answer a professor-facing question:

> “Besides PCA spectral, what other unsupervised or calibration approaches are reasonable for pump-sensor anomaly detection, and do they improve the false-alarm/recall trade-off?”

The implementation should be staged and subagent-driven. Start with the lowest-risk work: wire the existing Isolation Forest trainer into the SKAB experiment harness. Then add One-Class SVM, conformal thresholding, and a simple forecasting-residual baseline. Matrix Profile/distance profile is treated as a risk-gated spike because it may require new dependencies and heavier runtime.

The plan intentionally avoids supervised fault-label training. XGBoost/LightGBM remain diagnostic or upper-bound experiments only when labeled anomaly examples are available in train; they must not be presented as deployment-safe novel-fault detectors under the normal-only manifest.

---

## 2. Current Baseline

### 2.1 Existing Harness

Main integration file:

| File | Role |
|---|---|
| `scripts/run_skab_model_experiments.py` | Runs PCA raw/spectral, PCA ensemble, LSTM-AE, LSTM-AE DBSCAN threshold, XGBoost, LightGBM, and writes summary artifacts. |
| `tests/unit/test_skab_experiment_harness_contract.py` | Current harness tests for threshold variants and result ranking. |
| `ml/evaluation/metrics.py` | Shared split metrics and threshold-free metrics. |
| `ml/datasets/skab_manifest.py` | Manifest split contract and duplicate-split checks. |
| `ml/features/windowing.py` | Raw sensor windowing and label/changepoint propagation. |
| `ml/features/spectral.py` | Spectral window features used by PCA spectral. |
| `ml/registry/mlflow_client.py` | Existing MLflow logging/loading helpers for PCA, LSTM-AE, and Isolation Forest. |
| `scripts/train_all_for_comparison.py` | Existing MLflow comparison-run script; useful reference for tracking extra experiment rows. |

Existing experiment rows:

| Key | Family | Status |
|---|---|---|
| `01_pca_raw` | PCA raw windows | Implemented |
| `02_pca_spectral` | PCA spectral features | Implemented; current preferred candidate |
| `03_pca_feature_bagging_ensemble` | PCA ensemble | Implemented |
| `04_lstm_ae` | LSTM autoencoder | Implemented |
| `05_lstm_ae_dbscan_threshold` | LSTM score-threshold variant | Implemented |
| `06_xgboost` | Supervised upper-bound | Skipped under normal-only train |
| `07_lightgbm` | Supervised upper-bound | Skipped under normal-only train |
| `08_isolation_forest_spectral` | Isolation Forest spectral | Implemented as normal-only comparative baseline |
| `09_oneclass_svm_spectral` | One-Class SVM spectral | Implemented as normal-only comparative baseline |
| `10_isolation_forest_conformal` | Isolation Forest conformal threshold | Implemented as calibration wrapper, not a separately trained model |
| `11_forecasting_residual_naive` | Naive lag-one forecasting residual | Implemented as normal-only comparative baseline |
| `12_distance_profile_nearest_normal` | Nearest-normal distance profile | Implemented as bounded optional comparative baseline |

### 2.2 Reusable Local Code

Isolation Forest already exists and should be reused first:

| File | Reuse |
|---|---|
| `ml/training/train_isoforest.py` | Existing train/score/threshold/artifact flow. |
| `ml/inference/isoforest_inference.py` | Existing live inference service. |
| `tests/unit/test_isoforest_training_contract.py` | Existing training contract. |
| `tests/unit/test_isoforest_inference_contract.py` | Existing inference contract. |
| `tests/unit/test_isoforest_loader_contract.py` | Loader dispatch contract. |
| `scripts/train_all_for_comparison.py` | Already includes Isolation Forest comparison wiring. |

Missing and planned:

| Method | Local Status |
|---|---|
| One-Class SVM | Implemented as `09_oneclass_svm_spectral`. |
| Conformal thresholding | Implemented first for Isolation Forest as `10_isolation_forest_conformal`. |
| Forecasting residual detector | Implemented as `11_forecasting_residual_naive`. |
| Matrix Profile / distance profile | Implemented without new dependencies as `12_distance_profile_nearest_normal`. |

### 2.3 MLflow Baseline

There are two experiment paths with different MLflow behavior:

| Path | Current MLflow Behavior |
|---|---|
| `scripts/train_all_for_comparison.py` | Starts MLflow runs and logs comparison metrics/artifacts. |
| `scripts/run_skab_model_experiments.py` | Uses local artifacts by default; current trainer configs pass `log_mlflow=False`. |

Existing model-level MLflow hooks:

| Family | Current Support |
|---|---|
| PCA | `log_pca_training_run(...)` exists. |
| LSTM-AE | `log_lstm_ae_training_run(...)` exists. |
| Isolation Forest | `log_isoforest_training_run(...)` exists. |
| XGBoost/LightGBM | Comparison-script logging exists; supervised rows are skipped under normal-only train. |
| One-Class SVM / forecasting residual / conformal / distance profile | Supported through opt-in harness-level row logging. |

The new SKAB harness rows should eventually be visible in MLflow Compare Runs, but MLflow logging must remain opt-in so local tests and offline artifact generation stay cheap and deterministic.

---

## 3. Non-Negotiable Evaluation Protocol

All new experiments must follow the same honest SKAB protocol.

| Rule | Requirement |
|---|---|
| Training data | Fit detectors on manifest train split, normal-only. |
| Calibration data | Use validation scores for threshold calibration. Prefer validation-normal scores for deployment-safe thresholds. |
| Test data | Held-out test labels are used only for final metrics. |
| No point adjustment | Do not inflate metrics with post-hoc point adjustment. |
| No random-window leakage | Do not split overlapping windows randomly across train/test. |
| Score orientation | Higher anomaly score must mean more abnormal across all models. |
| Reporting | Report continuous score metrics where available plus thresholded F1/precision/recall/FAR. |
| Metadata | Every row must record feature mode, threshold method, calibration split, and whether train labels were used. |

Success is not only higher F1. A method can be useful if it improves one interpretable trade-off, such as lower false-alarm rate at acceptable recall, or if it provides a strong future-work story.

---

## 4. Experiment Stages

## Stage 1 — Harness Protocol Guardrails

### Objective

Lock the no-leakage protocol before adding more models.

### Work

- Extend `tests/unit/test_skab_experiment_harness_contract.py` with protocol assertions.
- Assert new model rows expose stable keys and metadata.
- Assert threshold metadata identifies validation calibration, not test tuning.

### Acceptance Criteria

- Tests protect train-normal-only fitting assumptions.
- Tests protect validation-only thresholding assumptions.
- Tests fail if a model row omits threshold metadata or score artifacts.

### Subagent Assignment

| Task | Category | Skills | Notes |
|---|---|---|---|
| Add harness protocol tests | `deep` | `[]` | Repo-specific test design. |

---

## Stage 2 — Isolation Forest Harness Wiring

### Objective

Wire the existing Isolation Forest trainer into `scripts/run_skab_model_experiments.py` as the first additional baseline.

### Work

- Import existing Isolation Forest training config/function from `ml/training/train_isoforest.py`.
- Add a harness helper mirroring `_run_pca_experiment(...)` or other training-result wrappers.
- Add experiment row after LightGBM:

```text
08_isolation_forest_spectral
```

- Prefer spectral features first so comparison is fair against PCA spectral.
- Keep existing PCA/LSTM/supervised behavior unchanged.

### Acceptance Criteria

- Smoke run includes an Isolation Forest row.
- Row records validation-calibrated threshold metadata.
- Existing Isolation Forest training tests still pass.
- Existing SKAB harness tests still pass.

### Subagent Assignment

| Task | Category | Skills | Notes |
|---|---|---|---|
| Wire IF into harness | `quick` | `[]` | Existing implementation; integration only. |

---

## Stage 3 — One-Class SVM Baseline

### Objective

Add One-Class SVM as a true unsupervised novelty-detection baseline.

### Design

Create a new training module:

```text
ml/training/train_oneclass_svm.py
```

Use `ml/training/train_isoforest.py` as the template.

Score orientation:

```text
anomaly_score = -model.decision_function(X)
```

Preprocessing:

- Fit scaler only on train-normal windows.
- Prefer `StandardScaler` for first implementation.
- Consider `RobustScaler` if validation behavior is unstable.

Initial conservative config:

| Parameter | Initial Values |
|---|---|
| `kernel` | `rbf` |
| `nu` | `0.005`, `0.01`, `0.02`, `0.05`, `0.1` |
| `gamma` | `scale`, `auto` |
| `feature_mode` | `spectral` first |

Planned row:

```text
09_oneclass_svm_spectral
```

### Acceptance Criteria

- Fits only on normal train windows.
- Calibrates threshold on validation scores.
- Does not tune on test labels.
- Has runtime guard or smoke-mode limits because kernel OCSVM can be slow.

### Subagent Assignment

| Task | Category | Skills | Notes |
|---|---|---|---|
| Add OCSVM training module and tests | `deep` | `[]` | Needs scaling/runtime care. |

---

## Stage 4 — Conformal Threshold Wrapper

### Objective

Add a reusable threshold-calibration wrapper for existing anomaly scores.

### Design

Add harness-level utilities near the existing DBSCAN threshold rule:

```text
ConformalThresholdRule
_derive_conformal_threshold(...)
_run_conformal_threshold_experiment(...)
```

Calibration formula:

```text
threshold = corrected upper quantile of normal calibration scores
```

For calibration set size `n` and target alpha:

```text
rank = ceil((n + 1) * (1 - alpha))
```

Then threshold test scores without using test labels.

Implemented row:

```text
10_isolation_forest_conformal
```

The first implementation uses Isolation Forest scores. It is a threshold calibration wrapper, not a separately trained model family. PCA spectral remains the deployment-preferred baseline unless a same-protocol rerun gives stronger evidence for another normal-only candidate.

### Acceptance Criteria

- Synthetic score tests verify corrected quantile behavior.
- Metadata records `threshold_method=conformal`, `alpha`, `calibration_split`, and source model key.
- Test labels are unavailable during threshold derivation.

### Subagent Assignment

| Task | Category | Skills | Notes |
|---|---|---|---|
| Add conformal threshold wrapper | `deep` | `[]` | Main risk is leakage semantics. |

---

## Stage 5 — Forecasting Residual Baseline

### Objective

Add a simple prediction-error detector to support the professor-facing “actual differs from expected” explanation.

### Design

Create:

```text
ml/training/train_forecasting_residual.py
```

Keep the first baseline simple and dependency-light:

| Candidate | Recommendation |
|---|---|
| Previous-value predictor | Best first baseline; simple, fast, explainable. |
| Rolling linear/autoregressive model | Good second option if time allows. |
| ARIMA / ExponentialSmoothing | Optional; only if dependencies/runtime are acceptable. |

Score:

```text
score_t = aggregate_channels(abs(y_t - yhat_t))
```

Use mean or max absolute residual across sensors. Calibrate the final threshold on validation-normal residual scores.

Implemented row:

```text
11_forecasting_residual_naive
```

### Acceptance Criteria

- Residuals are aligned with the correct target timestamp.
- No future samples from validation/test are used to train the predictor.
- Threshold calibration uses validation scores only.
- Synthetic injected-anomaly fixture shows score increases around anomaly.

### Subagent Assignment

| Task | Category | Skills | Notes |
|---|---|---|---|
| Add forecasting residual trainer | `deep` | `[]` | Needs time alignment and leakage checks. |

---

## Stage 6 — Matrix Profile / Distance Profile Spike

### Objective

Decide whether Matrix Profile/distance profile is feasible without derailing runtime or dependency scope.

### Recommended Approach

Use a risk-gated spike, not a mandatory benchmark stage.

Option A — dependency-backed:

- Use `stumpy` if dependency addition is approved.
- Use `stumpy.stump` for univariate sensor discord scoring.
- Use `stumpy.mstump(..., discords=True)` for multivariate exploration if runtime allows.

Option B — dependency-light:

- Implement a bounded NumPy distance-profile baseline over fixed windows.
- Score each test window by nearest distance to normal reference windows.
- Use aggressive sampling/runtime limits.

Option C — defer:

- If runtime or dependency risk is too high, document as future work with literature support.

Planned row if implemented:

```text
12_distance_profile_nearest_normal
```

Wave 4 implemented the dependency-light option as a nearest-normal distance profile with runtime caps. It should be reported as a bounded optional comparative baseline, not as proof that matrix-profile methods beat PCA spectral.

### Acceptance Criteria

- Either a bounded, tested baseline exists, or a documented skip path exists.
- No heavyweight dependency is added without explicit approval.
- Runtime guard is present for full SKAB runs.

### Subagent Assignment

| Task | Category | Skills | Notes |
|---|---|---|---|
| Matrix Profile feasibility spike | `ultrabrain` | `[]` | Decide implement-vs-defer with evidence. |

---

## Stage 7 — Reporting & Summary Artifacts

### Objective

Make new experiment outputs presentation-ready without exaggerating claims.

### Work

- Ensure `summary.csv` and `summary.md` include all completed methods.
- Ensure reporting text names the implemented rows through `12_distance_profile_nearest_normal`.
- Document whether harness MLflow logging is disabled by default or enabled by `--log-mlflow`; default local summaries must not require MLflow.
- Add clear metadata fields:
  - `model_family`
  - `feature_mode`
  - `threshold_method`
  - `calibration_split`
  - `train_label_policy`
  - `dependency_status` if relevant
- In docs/report language, distinguish:
  - deployable normal-only detectors,
  - threshold calibration variants,
  - optional/future-work methods,
  - supervised upper-bound methods.
- Keep PCA spectral as the deployment-preferred baseline unless same-protocol held-out metrics from a rerun prove another normal-only candidate should be reviewed.

### Acceptance Criteria

- The table can be shown to a professor without implying test leakage.
- Optional distance-profile status is explicit.
- PCA spectral remains comparable under the same protocol.

### Subagent Assignment

| Task | Category | Skills | Notes |
|---|---|---|---|
| Reporting update | `writing` | `[]` | Only after experiments produce results. |

---

## Stage 8 — MLflow Logging for Extra SKAB Experiments

### Objective

Make completed extra experiment rows visible in MLflow without turning every local harness run into a mandatory tracking-server dependency.

### Design

Add opt-in MLflow behavior to the SKAB harness after the extra model rows are stable.

Recommended CLI/API shape:

```text
uv run python scripts/run_skab_model_experiments.py --log-mlflow --mlflow-experiment-name pump_sentinel_skab_extra_experiments
```

The exact flag names can follow existing project conventions, but the behavior must be:

- Default remains local-only: no MLflow server needed for tests or normal artifact generation.
- When enabled, each completed experiment row gets a clear MLflow run name matching the row key.
- Runs log params, metrics, summary artifacts, metadata, and traceability tags.
- Skipped rows may log status metadata, but must not pretend to have `test_f1`.
- Model artifact logging should reuse existing model-level hooks where available:
  - PCA via `log_pca_training_run(...)` or existing artifact logging.
  - LSTM-AE via `log_lstm_ae_training_run(...)` when appropriate.
  - Isolation Forest via `log_isoforest_training_run(...)`.
  - One-Class SVM / forecasting residual only after their trainer contracts exist.
- Conformal threshold rows are score-threshold wrappers, not independently trained models; log them as calibration runs/artifact rows, not as new registered model families unless a concrete deployable artifact exists.

### Acceptance Criteria

- MLflow logging is opt-in.
- Unit tests can verify logging calls with fakes/mocks without a live MLflow server.
- Logged params include `model_family`, `feature_mode`, `split_protocol`, `threshold_method`, `calibration_split`, and `train_label_policy` where available.
- Logged metrics include held-out `test_` metrics when present.
- No default test or smoke run requires MLflow.

### Subagent Assignment

| Task | Category | Skills | Notes |
|---|---|---|---|
| Add opt-in MLflow logging for extra SKAB harness rows | `deep` | `[]` | Use fake MLflow tests; avoid live server dependency. |

---

## Stage 9 — Final Verification Gate

### Objective

Prove the staged implementation is correct before claiming completion.

### Verification Commands

```bash
uv run pytest tests/unit/test_skab_experiment_harness_contract.py -q
uv run pytest tests/unit/test_isoforest_training_contract.py -q
uv run pytest -q
uv run ruff check .
```

Experiment runs:

```bash
uv run python scripts/run_skab_model_experiments.py --smoke
uv run python scripts/run_skab_model_experiments.py
```

### Acceptance Criteria

- Relevant unit tests pass.
- Full test suite passes or unrelated pre-existing failures are documented.
- Ruff passes.
- Smoke experiment produces expected new rows.
- Full experiment produces summary artifacts.
- No threshold or hyperparameter selection uses held-out test labels.

### Subagent Assignment

| Task | Category | Skills | Notes |
|---|---|---|---|
| Verification and review | `deep` | `review-work` | Use after significant implementation. |

---

## 5. Subagent-Driven Execution Model

Implementation should run in waves, with each work unit delegated to a specialist subagent and verified before the next dependency wave.

| Wave | Parallel Work | Dependency |
|---|---|---|
| 1 | Harness protocol tests | None |
| 2 | Isolation Forest harness wiring | Wave 1 |
| 3 | One-Class SVM, conformal wrapper, forecasting residual, Matrix Profile spike | Wave 2 |
| 4 | MLflow logging for completed extra rows | Completed Wave 3 items |
| 5 | Reporting updates | Completed Wave 3 and Wave 4 items |
| 6 | Final verification/review | All implemented items |

### Delegation Template

Every implementation delegation must include:

```text
1. TASK: Atomic goal.
2. EXPECTED OUTCOME: Concrete deliverables and success criteria.
3. REQUIRED TOOLS: Allowed tool class.
4. MUST DO: Exhaustive requirements.
5. MUST NOT DO: Forbidden actions and leakage risks.
6. CONTEXT: Files, patterns, and constraints.
```

### Stage-Gate Rule

Do not start a dependent wave until the previous wave has evidence:

- changed-file diagnostics are clean,
- relevant tests pass,
- subagent output is inspected,
- no protocol violation is introduced.

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| One-Class SVM slow on full windows | Medium | Medium | Use smoke limits, conservative grids, and runtime skip metadata. |
| Threshold leakage through test labels | Medium | High | Add tests and metadata for calibration source. |
| Matrix Profile dependency/runtime bloat | High | Medium | Treat as spike; require approval before new dependency. |
| Forecast residual future leakage | Medium | High | Add residual alignment and no-future tests. |
| New baselines underperform PCA spectral | High | Low | Present as honest comparison; underperformance is still valid evidence. |
| False alarms remain high | Medium | Medium | Use conformal thresholding and report FAR/recall trade-off. |

---

## 7. Future Commit Boundaries

Do not commit during planning. If implementation is later committed, keep boundaries atomic:

1. Harness protocol tests.
2. Isolation Forest harness wiring.
3. One-Class SVM baseline.
4. Conformal threshold wrapper.
5. Forecasting residual baseline.
6. Matrix Profile spike or documented deferral.
7. Opt-in MLflow logging for completed extra rows.
8. Reporting and final verification cleanup.

---

## 8. Definition of Done

This plan is fully executed when:

- `08_isolation_forest_spectral` runs in the SKAB harness.
- One-Class SVM is available as an unsupervised spectral baseline.
- Conformal thresholding can wrap at least one compatible score source.
- Forecasting residual baseline is implemented or explicitly deferred with reason.
- Matrix Profile/distance profile is implemented with runtime guard or explicitly deferred.
- Completed extra rows can be logged to MLflow through an opt-in path.
- `summary.csv` and `summary.md` honestly report all completed methods.
- Full verification passes.
- The report/presentation can explain every method without claiming test-label tuning or supervised novel-fault detection.
