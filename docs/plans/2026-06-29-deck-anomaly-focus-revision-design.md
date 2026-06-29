# Deck Revision Design — Anomaly-Detection Focus

**Date:** 2026-06-29
**Scope:** Restructure `docs/presentation/pdam-pump-sentinel-deck.pptx` from a broad 30-slide
DevOps+MLOps+observability tour into a focused ~23-slide deck centered on **anomaly detection**.
Deck text stays English; white theme; template title -> subtitle (CYAN) -> content.

## Goal

The current deck broadened away from the core topic. This revision keeps anomaly detection as the
spine (problem -> anomaly types -> EDA -> candidate models -> results -> champion -> how PCA works)
and demotes everything else (MLOps, RouteMQ, Streamlit, Observability) to honest overviews placed
after the modeling story, each following a "definition first, then our case" pattern.

## Honest framing (unchanged invariants)

- SKAB is a surrogate water-circulation testbed, not real PDAM field data.
- Observability/Streamlit evidence is local Docker Compose, not production/cloud/SRE.
- Use the honest-eval spectrum; never sell an in-distribution number as novel-fault readiness.
- No API keys/secrets in any slide.

## Real-data corrections folded into this revision

Verified against the actual repo + regenerated EDA over `data/skab_split_manifest.json`:

1. **`RH` is NOT a SKAB sensor.** The 8 sensors are `Accelerometer1RMS`, `Accelerometer2RMS`,
   `Current`, `Pressure`, `Temperature`, `Thermocouple`, `Voltage`, `Volume Flow RateRMS`
   (`ml/datasets/skab_loader.py:6-15`, `docs/plans/design.md:360-366`). The Anomaly-4
   (multivariate) slide + chart must drop "Temp/RH" and use a real pair.
2. **Multivariate example = `Accelerometer1RMS` <-> `Accelerometer2RMS`.** Real Pearson:
   **-0.45 (normal/train) -> +0.997 (fault/test)** — two vibration channels lock together under
   fault while each amplitude can still look in-range. (train/test `correlation_matrix.csv`.)
3. **Stride = 1, not 30.** Retained experiments used `window_size=60, stride=1` (overlap 59/60)
   (`artifacts/skab-model-experiments/metadata.json:22-24`). Design doc planned stride 30
   (`design.md:262-263`) but that is not what ran. Deck states **window 60 / stride 1** everywhere;
   drop "stride 30 / 50% overlap".
4. **Split = 1 / 16 / 18 files (manifest), not "file 0-6 / 7-9".** Real manifest:
   train = `anomaly-free/anomaly-free.csv` (1 file, 9405 rows, all normal); validation = 16
   `valve1` files; test = 18 `valve2`+`other` files. This *strengthens* the story: train is
   normal-only (so supervised XGB/LGBM are literally SKIPPED — train has class `[0]` only,
   `summary.md` rows 32-33), and test faults differ from validation faults (true novel-fault
   generalization). Use these real numbers.

## Real EDA numbers (for slides 8-9; cite, do not invent)

Source: regenerated via `scripts/generate_skab_eda.py --split-manifest data/skab_split_manifest.json`.

- **Sensors (8):** vibration x2 (`Accelerometer1RMS`, `Accelerometer2RMS`), motor current
  (`Current`), hydraulic pressure (`Pressure`), thermal x2 (`Temperature`, `Thermocouple`),
  supply voltage (`Voltage`), volume flow (`Volume Flow RateRMS`). Sampling ~1 Hz. Missing rate
  **0%** across all sensors (strict `allow_missing=False`).
- **Windows (window 60 / stride 1):** train 9346 (all normal); validation 17216
  (9963 normal / 7253 anomaly); test 18179 (10527 normal / 7652 anomaly ~= 58/42).
- **Per-sensor mean +/- std (normal baseline):** Current 2.40+/-0.49 · Temperature 89.5+/-0.67 ·
  Thermocouple 28.5+/-0.73 · Voltage 228.6+/-11.0 · Volume Flow 125.2+/-1.6 · Pressure 0.11+/-0.25 ·
  Accelerometer1RMS 0.213+/-0.005 · Accelerometer2RMS 0.268+/-0.004.
- **Correlation normal -> fault (highlights):** Accel1<->Accel2 -0.45 -> +0.997 ·
  Temperature<->Thermocouple -0.89 -> +0.38 · Temperature<->Volume Flow -0.76 -> +0.78 ·
  Thermocouple<->Volume Flow +0.83 · Current<->Voltage +0.47.

## Modeling results (slide 12; honest spectrum)

- PCA-spectral (`02_pca_spectral`, champion) F1 **0.558** (precision 0.549, recall 0.568) — best
  completed normal-only row (`artifacts/skab-model-experiments/summary.md`).
- Honest-eval spectrum: PCA-spectral **0.58** (deployable, novel-fault) | supervised cross-group
  **0.60** (AUC 0.937) | in-distribution XGB **0.909** / LGBM **0.905** (upper bound, needs labels) |
  random-window **0.985** (leakage, NOT a claim).
- Supervised XGB/LGBM **SKIPPED** in the honest split (train normal-only) — the real reason
  unsupervised is the Day-1 champion.
- Promotion gate: promote iff `F1_challenger > F1_champion + 0.02` AND
  `FAR_challenger <= FAR_champion * 1.05`.

## Final spine (23 slides)

### Core — Anomaly Detection
1. **Title** — PDAM Pump Sentinel: multivariate anomaly detection for water pumps. (`kind=title`,
   image `slide-overview-pipeline.png`)
2. **Why This Problem Matters** — pump faults are multivariate, gradual, costly. (image
   `slide-problem-threshold.png`)
3. **Anomaly 1: Point** — single spike/drop. (`kind=figure`, `problem-anomaly-point.png`)
4. **Anomaly 2: Context / Change Point** — new regime, never crosses static limit. (`figure`,
   `problem-anomaly-context.png`)
5. **Anomaly 3: Pattern / Subsequence** — stuck/flatline shape. (`figure`,
   `problem-anomaly-pattern.png`)
6. **Anomaly 4: Multivariate Relationship** — each channel fine, relationship breaks.
   **FIX: Accelerometer1 <-> Accelerometer2, corr -0.45 (normal) -> +0.997 (fault).** (`figure`,
   regen `problem-anomaly-multivariate.png`)
7. **Why Min/Max Alone Is Not Enough** — static limit catches spike, misses the other 3 families.
   (`figure`, `problem-minmax-limit.png`)
8. **EDA Definitive** — 8 sensors + roles, ~1 Hz, missing 0%, manifest train 1 / val 16 / test 18,
   window 60 / stride 1. (`kind=detail`, image `slide-data-sensors.png`)
9. **EDA Numerical + Correlation Heatmap** — per-sensor range table + **real correlation heatmap**;
   punchline: sensors are strongly coupled in normal -> that coupling is what PCA learns and what
   breaks under fault. (`kind=detail`, NEW image `eda-correlation-heatmap.png`)
10. **Candidate Models: Supervised vs Unsupervised** — explicit rosters: unsupervised (PCA T2/Q
    spectral champ / raw / bagging, Isolation Forest +conformal, One-Class SVM, LSTM-AE +DBSCAN,
    forecasting-residual, distance-profile) vs supervised (XGBoost, LightGBM only); identical
    `AnomalyVerdict`; supervised SKIPPED on normal-only train. (`kind=detail`, image
    `model-candidates-training-split.png`)
11. **Model Inputs: Rows -> Windows -> Features** — hero figure: ninerouter-generated supervised-vs-
    unsupervised data-usage visualization (`supervised-vs-unsupervised-data.png`); shared window 60 /
    stride 1, raw + spectral features; supervised trains on labeled rows, unsupervised (PCA champ,
    RobustScaler) on normal-only. (`kind=figure`)
12. **Modeling Results: Honest F1 Spectrum** — ranked F1 table (10-row retained ladder, champion
    02_pca_spectral 0.558 highlighted) + honest caption (0.60 / AUC 0.937 / 0.909-0.905 / 0.985 =
    context, not deployed). (`kind=table`)
13. **Why PCA-Spectral as Day-1 Champion** — deployable before labeled faults exist; supervised
    can't even train in the honest split; fast, decomposable, operator-explainable. (image
    `slide-pca-scatter.png`)
14. **PCA Mechanics** — scale -> learn directions -> project -> reconstruct -> residual. (`detail`,
    `modeling-pca-t2-q-road.png`)
15. **T2 vs Q/SPE** — on-pattern-extreme vs off-pattern-residual; OR decision;
    score = max(T2/thr, Q/thr); top contributing sensor. (`detail`, `pca-reconstruct-residual.png`)
16. **Spectral FFT** — why spectral beats raw: rhythm shift before level moves. (`detail`,
    `pca-eigen-scree.png`)

### Support — overviews (definition first, then our case)
17. **What Is MLOps + Why It Helps Us** — lifecycle ops (train -> deploy -> monitor -> retrain);
    our case (cold-start champion, drift -> retrain, downtime-free promotion, operator evidence);
    tools (MLflow, Evidently, Prometheus + Grafana). (`kind=detail`)
18. **MLflow: Registry + Champion-Challenger Gate** — (1) Model Registry: versions + `@champion`
    alias + hot-swap; (2) promotion gate formula. (`kind=detail`, reuse `mlops-evidence-loop.png`)
19. **RouteMQ Overview** — MQTT app framework: Router -> Middleware -> Controller;
    `factory/skab/{station}/telemetry` -> ingest -> synchronous inference -> publish anomaly +
    persist (Redis fast / ClickHouse history). Big picture only. (`kind=detail`, reuse
    `pipeline-observability-chain.png`)
20. **Operator Dashboard (Streamlit)** — 7-page grid: Overview (health pill, KPI, station picker) ·
    Live Sensors (5s) · Anomaly History (timeline/severity/drilldown) · Model Registry · Drift
    Reports · System Health (10s) · Runbook. Operator actions ack/mute/note -> Redis keys (NOT yet
    training labels). + 1 real Streamlit screenshot. (`kind=screenshot`,
    `t9-observability-streamlit-observability-snapshot-20260628T0535Z.png`)
21. **What Is Observability** — general: metrics, logs, traces (3 pillars); ask arbitrary questions
    about system state from outside. (`kind=detail`)
22. **Observability in Our AIoT Case** — Prometheus `pumpad_*` + RouteMQ metrics; Grafana
    (pipeline / MLOps / system health / broker); local alert rules; Streamlit snapshot. Honest:
    local Docker Compose evidence, not production SRE. + 1 real Grafana screenshot.
    (`kind=screenshot`, `t9-observability-grafana-pipeline-observability-20260609T130723Z.png`)
23. **Closing — Limitations + Roadmap** — sync inference, env-flag schedulers, PCA-only retrain,
    process-local hot-swap -> operator labels, supervised promotion gates, incident routing.
    (`kind=closing`, `slide-roadmap.png`)

## Assets

**New (3):**
- `problem-anomaly-multivariate.png` — REGEN matplotlib chart: Accel1 vs Accel2, normal (-0.45)
  vs fault (+0.997) relationship break, English labels.
- `eda-correlation-heatmap.png` — NEW from real data (already produced under
  `/tmp/opencode/skab_eda_out/train/correlation_heatmap.png`); produce a deck-styled version
  (palette INK/TEAL/AMBER, white bg). Optionally normal-vs-fault pair.
- `model-candidates-training-split.png` — NEW diagram: a train file's rows (normal vs anomalous),
  unsupervised sees normal-only, supervised sees all + labels.

**Reused:** all current `problem-anomaly-*`, `problem-minmax-limit`, `pca-*`, `modeling-*`,
`slide-*`, `mlops-evidence-loop`, `pipeline-observability-chain`, plus 2 real screenshots
(`t9-observability-streamlit-*`, `t9-observability-grafana-*`).

**Cut from old deck:** demo storyboard T+0-T+9; "What's Implemented Today"; standalone three-layer
architecture; "Scope and Honest Data Framing" (folded into EDA/title); 2 of 4 screenshot-evidence
slides; deep "data contract" + "feature modes" split (folded into slide 10); MLOps internals slides
(Redis/ClickHouse, Prometheus catalog, Grafana 4-dashboard atlas, local alerts, RouteMQ internals,
drift/retrain) collapsed into slides 17-19, 22.

## Implementation steps

1. Rewrite the `SLIDES` tuple in `scripts/generate_presentation_deck.py`: cut, reorder, add EDA (2)
   + Candidate Models (1) + restructured support sections; apply the 4 corrections (Accel pair,
   stride 60/1, split 1/16/18, real EDA numbers).
2. Generate the 3 new assets (parent-only image generation; charts via matplotlib like
   `/tmp/opencode/make_charts.py`; heatmap from real data; training-split diagram). Color-key
   decorative diagrams transparent if needed; charts/heatmap keep white bg.
3. Rewrite `docs/presentation/deck-outline.md` to match the new 23-slide spine.
4. Update `tests/unit/test_presentation_deck_contract.py`: `len(SLIDES) == 23` (2 places); adjust
   the `## Slide ` outline count assertion to the new section count; keep forbidden-fragment scan;
   keep no-secrets guarantee.
5. Regenerate the deck, then QA: `ruff` clean, tests pass, `soffice` render -> `pdfinfo` Pages == 23,
   per-slide `pdftoppm` checks for slides 6/9/11 (the fixed/new ones), Indonesian-token + secrets
   scan clean.

## Verification pass (5 parallel sub-agents) — corrections, enrichments, guardrails

All findings are cited to repo file:line. Apply on top of the slide specs above. This is the
authoritative correction layer.

### Slides 3-9 (anomaly types + EDA)

Corrections:
- Accel1<->Accel2 correlation is **train normal baseline -0.4455 -> test split +0.9966**; the test
  value spans the WHOLE held-out test split (valve2+other files, normal AND anomaly rows), so do
  NOT call it "fault-only" — say "normal baseline (train) vs fault-file test split".
- The 4-type taxonomy (point/context/pattern/multivariate) is **general ML taxonomy, not
  repo-defined**. Only outlier-vs-changepoint/collective separation is grounded
  (`design.md:42-48`, `ml/features/windowing.py:54-59`). Frame slides 3-5 as teaching taxonomy;
  keep SKAB grounding only where it exists.
- "no constant columns" -> "no constant **sensor** columns" (train `anomaly`/`changepoint` are
  constant-true). (`train/summary.json:5-17`)
- Do NOT depict val/test as one clean continuous 1 Hz stream: cadence mode is 1.0s but the
  concatenated val/test are not monotonic; test has 718 duplicate timestamps + a 1,887,107-second
  max gap (file concatenation). Use heatmap + balance bar, not a continuous timeline.
  (`val/test timestamp_quality.json`)
- No physical units exist in the repo — use sensor roles, not units. (`skab_loader.py:6-15`)

Enrichments (real, add to slides 8-9):
- Row-level: train 9405 rows (0 anomaly / 0 changepoint); val 18160 (6309 anomaly / 63
  changepoint); test 19241 (6758 anomaly / 66 changepoint). (`{train,validation,test}/report.md:5-8`)
- Window class balance: train 9346 all-normal; val 9963N/7253A of 17216; test 10527N/7652A of 18179.
  (`artifacts/skab-model-experiments/02_pca_spectral/metrics.json`)
- Label ranges: train 0; val 16 anomaly ranges (valve1) + 63 changepoint; test 18 anomaly ranges
  (4 valve2 + 14 other) + 66 changepoint. (`{...}/label_ranges.csv`)
- Distribution drift normal -> test (strong visual): Volume Flow p01 121.0 -> 14.99; Accel1 max
  0.227 -> 0.723; Accel2 max 0.280 -> 0.800; Temperature max 91.7 -> 95.0; Thermocouple max
  29.5 -> 33.4. (`{train,test}/sensor_distributions.csv`)
- Correlation per split: train Accel1<->Accel2 -0.45 / Temp<->Thermo -0.89; validation +0.485 /
  +0.867; test +0.997 / +0.38. (`{...}/correlation_matrix.csv`)

### Slides 10-12 (model inputs + candidates + results)

Corrections:
- **RobustScaler-on-normal is PCA-specific, not all candidates.** Supervised uses StandardScaler +
  fits all rows (`train_supervised.py:42-50,272-278`). Slide 10 must say "PCA champion: RobustScaler
  on normal-only windows".
- `windowing.py` has **no stride default**; 60/1 is the harness default (stride 30 only under
  `--smoke`). (`windowing.py:20-27`, `run_skab_model_experiments.py:278-283`,
  `metadata.json:18-24`)
- Champion F1 exact = **0.5583** (round to 0.56/0.58 on slide); precision 0.5488, recall 0.5682,
  PR-AUC 0.6381, ROC-AUC 0.6843; thresholds T2 1.0776 / Q 6139.43.
  (`02_pca_spectral/metrics.json:23-55`)
- **0.60 / 0.909 / 0.905 / 0.985 are NOT rows in the retained summary** — only the unsupervised rows
  are. 0.60 = supervised cross-group via `data/skab_supervised_manifest.json` (train
  valve1+anomaly-free, val other, test valve2); 0.909/0.905/0.985 = ADR 0001 / sprint-remaining
  context. Slide 12 must footnote these as split-taxonomy context, with the retained champion 0.558
  as the deployed-model artifact.

Enrichments:
- Exact completed F1 ladder (`summary.md:22-31`): PCA-spectral 0.558 > IsoForest-conformal 0.519 >
  PCA-bagging 0.519 > IsoForest-spectral 0.493 > PCA-raw 0.490 > distance-profile 0.467 >
  forecasting-naive 0.443 > LSTM-AE 0.260 > LSTM-DBSCAN 0.225 > OneClassSVM 0.000.
- `AnomalyVerdict` == `SupervisedAnomalyVerdict` field-for-field (`pca_inference.py:22-35`,
  `supervised_inference.py:29-42`); supervised sets t2=q=None and duplicates the probability
  threshold into t2/q_threshold (`supervised_inference.py:183-200`).
- Spectral feature count = 8 x (4+4) = 64; enriched = sensor_count x (n_bands+7)+6 (offline only).
- Hyperparams (`metadata.json:1-24`): seed 42; LSTM epochs 30 / batch 32 / patience 5; supervised &
  IsolationForest 200 estimators; OneClassSVM nu 0.05; PCA ensemble 8 / subset 0.7.

### Slides 13-16 (why champion + PCA mechanics)

Corrections:
- Code default scaler = **RobustScaler** (worksheet teaches z-score; do not claim z-score is the
  default). (`pca_detector.py:30-35,126-135`)
- **Dominant frequency = intuition only**, not a live feature column. (`spectral.py:67-82`)
- Spectral centroid in code is **energy-weighted** (|rfft|^2), not the worksheet's magnitude-weighted
  formula; if a formula is shown, use energy-weighted. (`spectral.py:77-80`)
- Band energies = `np.array_split` over FFT energy bins (mean-centered per sensor first), **not named
  Hz bands**. (`spectral.py:74-80`)
- "Top contributing sensor" = **largest residual contributor**, not causal root cause; it is
  residual-derived even when the alarm was T2-driven. (`pca_inference.py:250-267`)
- "Supervised can't train" applies **only to the normal-only honest split**; cross-group supervised
  exists. Keep the claim scoped.
- Drop "us-scale latency" (speaker-note only, unmeasured); "fast" is fine.

Confirmed math to cite: T2 = sum(score^2 / explained_variance); Q = squared reconstruction residual
via `inverse_transform`; n_components=0.9 (svd_solver full); thresholds = empirical normal quantiles
(p95) with floor; OR rule strict `>`; score = max(T2/T2_thr, Q/Q_thr). (`pca_detector.py:55-114`)

### Slides 17-18 (MLOps + MLflow) — FRAMING CORRECTION

- **The champion-challenger gate is NOT an MLflow feature** — it is custom project logic
  (`ml/monitoring/champion_challenger.py:8-32`) that decides, after which MLflow only moves the
  alias. Slide 18 should read: "(1) MLflow Model Registry — versioned models + `@champion` alias;
  (2) a **custom** champion-challenger promotion gate that drives the alias move." Do not call the
  gate an MLflow feature.
- Gate exact: `f1_chal > f1_champ + 0.02 AND far_chal <= far_champ * 1.05`; rationale: 0.01 F1 =
  noise, 0.02 buffer, FAR <= +5%. (`ADR 0003:78-87`, `champion_challenger.py:24-32`)
- Hot-swap is **process-local** (module-global + thread lock; `set_inference_service` in-process).
  "Downtime-free" = no app restart in the demo path, not multi-process/production.
- Schedulers **off by default**: ENABLE_RETRAIN_SCHEDULER (30m), ENABLE_DRIFT_SCHEDULER (15m),
  ENABLE_MODEL_REFRESH_SCHEDULER (5m); compose sets none. (`bootstrap/app.py:199-204`)
- "No runtime alias polling" is outdated -> "optional alias-refresh scheduler exists, off by default".
- Cold-start loads `PUMPAD_MODEL_DIR` first, then MLflow `@champion` (not MLflow-first).
- Scheduled retrain is **PCA by default** but code also supports lstm_ae/xgboost/lightgbm/
  live_clickhouse.
- Drift: Evidently `DataDriftPreset(drift_share)`; dataset_drift from drifted-column share;
  DRIFT_SHARE_THRESHOLD default 0.5; detected drift -> RetrainingJob.

### Slides 19-22 (RouteMQ + Streamlit + Observability)

Corrections (use real titles):
- Streamlit page titles: Overview renders "PDAM Pump Sentinel"; "Live Sensors" -> "Live Sensor
  Monitoring"; "Drift Reports" -> "Drift & Retrain Reports". Others match (Anomaly History, Model
  Registry, System Health, Operator Runbook). Show page-name + real title.
- "MLOps health pill" is a broader **Operator Console Health** banner (MLflow/Redis/ClickHouse/MQTT/
  active-model/telemetry probes).
- ack/mute/note write **operator_actions** (Redis + ClickHouse), explicitly NOT training labels; a
  separate label path exists (`factory/skab/{station}/label` -> label_controller -> `operator_labels`
  table). Keep "not yet training labels" and may note the separate label route.

Confirmed / enrich:
- Middleware order: Validate (JSON + non-empty sensors) -> RateLimit (50 msg/1s/station) ->
  Correlation; inference synchronous; anomaly topic `factory/skab/{station}/anomaly` QoS 1; persist
  Redis-first then ClickHouse. (`telemetry.py:12-21`, `anomaly_controller.py:26-56`, `persistence.py`)
- ack/note TTL 30 days, mute 15m. (`widgets.py:16,71-92`)
- pumpad_* catalog = 15 metrics (`metrics.py:33-99`); RouteMQ framework metrics `routemq_*`; 2 scrape
  jobs: routemq-app (app:8080) + mosquitto-exporter (9234). (`prometheus.yml:7-18`)
- Alert group `pumpad-local-observability`, 7 rules: AppMetricsDown, TelemetryStale, InferenceErrors,
  PersistenceWriteErrors, DriftReportStale, ActiveModelStale, HighSeverityAnomalyEvents.
  (`pumpad-alerts.yml:1-65`)
- Grafana: pumpad-observability "RouteMQ Observability", pumpad-mlops "MLOps Loop",
  pumpad-system-health "System Health", pumpad-mqtt-broker "MQTT Broker"; all refresh 10s.

Honesty guardrails:
- Slide 21 defines metrics/logs/traces; Slide 22 must say the IMPLEMENTATION proves **metrics +
  dashboards + local alert rules** only — centralized logs/traces (Loki/OTel), Alertmanager/PagerDuty
  routing, and k8s/cloud are future, not evidence. (`infra/README.md:8`)
- Only Live Sensors + System Health autorefresh; other pages are manual/TTL.
- Anomaly inference is synchronous; queue/worker is MLOps-only.

## Open / deferred

- Optional: recapture fresh Streamlit/Grafana screenshots via `make replay` before final.
- Commit happens only on explicit user request (generator + test + deck.pptx + new PNGs + this
  design doc + outline).

## Addendum: Slides 13-16 computational-cost + continuous-learning revision (2026-06-30)

Decision (brainstorming): weave cost-computation rationale + mathematical low-cost proof + continuous-learning tie across the existing four PCA slides — no deck-length change, still 23 (option A). Visual: a new matplotlib cost-contrast chart on slide 14 (option V2).

Per-slide changes:
- Slide 13 (Why PCA-Spectral): reframed to lead with cost + continuous learning. PCA is the Day-1 champion because the CT loop retrains on a cadence (scheduled ~30 min + drift-triggered, PCA path) and a single full-SVD + closed-form-threshold fit is cheap, deterministic, and reproducible for the champion-challenger gate. `kind=detail`, image `slide-pca-scatter.png`.
- Slide 14 (PCA Mechanics): now hosts the low-cost PROOF. Train = one full SVD on normal windows (n=9346, d=64), O(n*d^2 + d^3); thresholds = p95 empirical quantiles (one O(n log n) sort); inference per window O(d*k); FFT O(d log d). Contrast: LSTM-AE ~100 epochs backprop per retrain vs PCA single SVD. Image swapped to NEW `pca-vs-lstm-retrain-cost.png` (matplotlib, log-scale bars + Big-O, structural not measured).
- Slide 15 (T2 vs Q/SPE): added one cost line — both statistics reuse one projection, scoring O(d*k), negligible at ~1 Hz.
- Slide 16 (Spectral FFT): added one cost line — per-window FFT O(d log d) per channel, no change to low-cost profile.

Grounding (honest): `PcaT2QDetector.fit` = scaler.fit_transform + PCA(svd_solver='full', n_components=0.9).fit + p95 quantile thresholds (`pca_detector.py:42-64`). `LstmAeTrainingConfig.epochs=100` (`train_lstm_ae.py:38`). Retrain/drift cadence defaults ~30/15 min, off-by-default, PCA-path (`ml/monitoring/scheduler.py`). Proof is complexity-theoretic (Big-O) + real config; FLOPs labeled order-of-magnitude; NO measured production latency claimed.

New asset: `docs/presentation/assets/generated/pca-vs-lstm-retrain-cost.png`. `modeling-pca-t2-q-road.png` is now unused on slide 14 (left in place).

## Addendum: RouteMQ per-layer split (2026-06-30)

Decision (brainstorming): A + P1 + V3. Rework the single "RouteMQ Overview" slide into a five-slide block (map + four per-layer slides); deck grows 23 -> 27. Visual: a ninerouter-generated "connect it all" architecture map on the map slide; the four layer slides are text-only (`kind=detail`, two-column bullets).

New structure (old slide 19 expands to 19-23; old 20-23 renumber to 24-27):
- 19 RouteMQ: Connect It All Together — hero map `routemq-connect-map.png` (`kind=figure`, no bullets). Sync path MQTT -> Router -> Middleware -> Controller -> anomaly out + persist (Redis/ClickHouse); async Queue/Worker -> Drift/Retrain -> MLflow registry; champion loaded back to Controller; Dashboard + Observability read storage.
- 20 Layer 1 Router — subscribe telemetry topic at QoS 1, declarative dispatch to anomaly_controller.ingest.
- 21 Layer 2 Middleware — Validate -> Rate-limit (50/s/station) -> Correlation, ordered chain.
- 22 Layer 3 Controller + Inference + Persistence — sync core: inference service loads champion, scores window, publishes anomaly, persists Redis -> ClickHouse.
- 23 Layer 4 Queue/Worker — DriftReportJob + RetrainingJob async; read ClickHouse/SKAB, write MLflow + Redis; scheduler off-by-default; ties to slides 13-14 cheap retraining.
- 24-27: Operator Dashboard (Streamlit), What Is Observability, Observability in Our AIoT Case, Closing (renumbered, content unchanged).

Grounding: Router (`app/routers/telemetry.py`); Middleware (`ValidateTelemetryMiddleware`, `InMemoryRateLimitMiddleware`, `CorrelationLoggingMiddleware`); Controller (`anomaly_controller.py`) + services (`inference.py`, `persistence.py`); Jobs (`DriftReportJob`, `RetrainingJob`).

Contract test updated: slide count 23 -> 27 (three assertions). New asset: `routemq-connect-map.png`. `pipeline-observability-chain.png` is now unused on the RouteMQ slide (left in place).
