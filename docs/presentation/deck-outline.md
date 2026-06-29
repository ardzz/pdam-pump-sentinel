# PDAM Pump Sentinel Presentation Deck Outline

This outline follows `docs/plans/2026-06-29-deck-anomaly-focus-revision-design.md`, specifically the final 23-slide spine plus the verification pass corrections.

Honest framing: SKAB dipakai sebagai surrogate water-circulation testbed, not real PDAM operational data. Observability proof is local Docker Compose evidence, not production cloud or SRE evidence.

## Slide 1 — PDAM Pump Sentinel

**Title:** PDAM Pump Sentinel: Multivariate Anomaly Detection for Water Pumps

**Key message:** The project is an AIoT slice for pump anomaly detection, from MQTT telemetry to inference, model lifecycle evidence, operator UI, and local observability.

**Slide bullets:**

- Position the work as multivariate anomaly detection for water-circulation pump behavior.
- State the honest data frame: SKAB is a surrogate testbed for methodology and demo proof.
- Show the pipeline image `slide-overview-pipeline.png` as the audience map for the deck.

**Speaker cue:** Start with the pump-maintenance problem, then say the deck focuses on the anomaly-detection spine before support systems.

## Slide 2 — Why This Problem Matters

**Title:** Why This Problem Matters

**Key message:** Pump faults can be gradual, coupled across sensors, and expensive when noticed too late.

**Slide bullets:**

- A single threshold may catch a spike, but it can miss regime shifts, flatline patterns, and broken sensor relationships.
- Water-distribution pumps are critical assets, so late detection affects service continuity and repair cost.
- The system needs to read sensor behavior together, not as eight isolated min/max checks.

**Speaker cue:** Frame the problem as a pattern-recognition problem, not a dashboard-alarm problem.

## Slide 3 — Anomaly 1: Point

**Title:** Anomaly 1: Point

**Key message:** A point anomaly is a single sample or short spike that stands out from its local behavior.

**Slide bullets:**

- Use this as teaching taxonomy, not as a repo-defined SKAB category.
- Example shape: one sudden jump or drop in one channel.
- Static min/max limits can catch this family when the spike crosses the limit.

**Speaker cue:** Keep this quick. It is the familiar case that makes the next three cases easier to contrast.

## Slide 4 — Anomaly 2: Context / Change Point

**Title:** Anomaly 2: Context / Change Point

**Key message:** A contextual anomaly can look acceptable at one instant but abnormal in its operating regime.

**Slide bullets:**

- The signal moves into a new regime without necessarily crossing a static limit.
- The deck can connect this to the outlier-vs-changepoint separation noted in the design docs.
- The reviewer should see why a time-window model is more useful than checking each row alone.

**Speaker cue:** Say the issue is not always the value, it is the value in context.

## Slide 5 — Anomaly 3: Pattern / Subsequence

**Title:** Anomaly 3: Pattern / Subsequence

**Key message:** A pattern anomaly is a suspicious shape across a sequence, such as stuck, flat, or rhythm-broken behavior.

**Slide bullets:**

- Use this as teaching taxonomy for collective or subsequence behavior.
- The shape can be wrong even when individual samples are inside range.
- This motivates row-to-window modeling before the deck introduces feature extraction.

**Speaker cue:** Make the point that a single row can look harmless while the short story around it is abnormal.

## Slide 6 — Anomaly 4: Multivariate Relationship

**Title:** Anomaly 4: Multivariate Relationship

**Key message:** Some faults break the relationship between channels while each channel still looks plausible alone.

**Slide bullets:**

- Use the verified SKAB pair: Accelerometer1<->Accelerometer2, shown as `Accelerometer1RMS` and `Accelerometer2RMS` in code and EDA.
- Verified Pearson relationship: r -0.45 normal -> +0.997 over the test split.
- The +0.997 value is for the whole held-out test split with valve2 plus other files, so don't describe it as only anomalous rows.

**Speaker cue:** This is the anchor example for why multivariate learning matters. The model learns coupling, not just limits.

## Slide 7 — Why Min/Max Alone Is Not Enough

**Title:** Why Min/Max Alone Is Not Enough

**Key message:** Static limits catch obvious spikes but miss context, pattern, and multivariate failures.

**Slide bullets:**

- Min/max is useful as a guardrail, but it is not a complete anomaly detector.
- A value can stay in range while its regime, shape, or relationship changes.
- This slide closes the problem setup and opens the EDA section.

**Speaker cue:** Avoid over-attacking thresholds. Say thresholds are baseline safety checks, not the whole strategy.

## Slide 8 — EDA Definitive

**Title:** EDA Definitive

**Key message:** The dataset contract is fixed: eight sensors, clear roles, no missing sensor values, and a windowed split that matches retained experiments.

**Slide bullets:**

- Sensors and roles: vibration x2, motor current, hydraulic pressure, thermal x2, supply voltage, and volume flow.
- Sampling cadence mode is about 1 Hz, but concatenated validation/test files aren't one continuous timeline.
- Missing rate is 0% across sensors with strict missing policy.
- Manifest files: train 1, validation 16, test 18.
- Row counts: train 9405 rows with 0 anomaly and 0 changepoint, validation 18160 rows with 6309 anomaly and 63 changepoint, test 19241 rows with 6758 anomaly and 66 changepoint.
- Windowing: window 60 / stride 1. Windows: train 9346 all normal, validation 17216 with 9963 normal and 7253 anomaly, test 18179 with 10527 normal and 7652 anomaly.

**Speaker cue:** Stress that these are real regenerated numbers, not design placeholders. Don't add physical units because the repo doesn't define them.

## Slide 9 — EDA Numerical + Correlation Heatmap

**Title:** EDA Numerical + Correlation Heatmap

**Key message:** Normal sensors are coupled, and that coupling is exactly what PCA can learn and later test.

**Slide bullets:**

- Normal baseline mean +/- std: Current 2.40 +/- 0.49, Temperature 89.5 +/- 0.67, Thermocouple 28.5 +/- 0.73, Voltage 228.6 +/- 11.0.
- Normal baseline mean +/- std continued: Volume Flow 125.2 +/- 1.6, Pressure 0.11 +/- 0.25, Accelerometer1RMS 0.213 +/- 0.005, Accelerometer2RMS 0.268 +/- 0.004.
- Correlation highlights: Accel1/Accel2 train -0.45, validation +0.485, test +0.997. Temperature/Thermocouple train -0.89, validation +0.867, test +0.38.
- Distribution drift examples from train normal to test: Volume Flow p01 121.0 -> 14.99, Accel1 max 0.227 -> 0.723, Accel2 max 0.280 -> 0.800, Temperature max 91.7 -> 95.0, Thermocouple max 29.5 -> 33.4.
- Use the real correlation heatmap image `eda-correlation-heatmap.png`.

**Speaker cue:** The punchline is simple: PCA is justified because normal operation has structure, not because any single channel has a magic limit.

## Slide 10 — Candidate Models: Supervised vs Unsupervised

**Title:** Candidate Models: Supervised vs Unsupervised

**Key message:** The same SKAB problem is evaluated through a common verdict contract, but the training labels each family may see are different.

**Slide bullets (two explicit rosters):**

- Unsupervised candidates (train on normal-only windows): PCA T2/Q — spectral champion, raw, feature-bagging; Isolation Forest (+ conformal threshold); One-Class SVM; LSTM autoencoder (+ DBSCAN threshold); plus forecasting-residual and distance-profile baselines.
- Supervised candidates (need labeled normal + fault rows): XGBoost and LightGBM only.
- Honest Day-1 split has a normal-only train file, so XGBoost/LightGBM are SKIPPED (single class [0]); unsupervised is the only family that can train.
- Both families emit an identical AnomalyVerdict, so downstream consumers receive the same output shape.

**Speaker cue:** This slide explains why unsupervised is not a weak fallback. It is the only family that can train under the Day-1 split.

## Slide 11 — Model Inputs: Rows -> Windows -> Features

**Title:** Model Inputs: Rows -> Windows -> Features

**Key message:** Every candidate is compared through the same row-to-window contract, while preprocessing differs by model family where the code says it does.

**Presentation: ninerouter-generated supervised-vs-unsupervised data-usage visualization (`supervised-vs-unsupervised-data.png`) as the hero figure, with a short input-contract caption.**

- Shared input: 60-sample windows (stride 1), raw or spectral FFT features, identical for every candidate.
- What differs is data usage: supervised trains on labeled normal + fault rows; unsupervised (PCA champion, RobustScaler) trains on normal-only windows.
- Left panel: labeled two-class decision boundary. Right panel: normal-only cluster envelope with flagged outliers.

**Speaker cue:** Use the visualization to land the labels point — same windows in, different label regimes out.

## Slide 12 — Modeling Results: Honest F1 Spectrum

**Title:** Modeling Results: Honest F1 Spectrum

**Key message:** The retained deployed artifact is PCA-spectral F1 0.558, while the higher numbers are context for different split assumptions.

**Presentation — ranked F1 table (columns: Model, F1, Family / note):**

- Highlighted champion row: `02_pca_spectral`, F1 0.558, precision 0.549, recall 0.568, thresholds T2 1.0776 and Q 6139.43 (PR-AUC 0.6381, ROC-AUC 0.6843 in the note).
- Completed retained ladder as table rows, descending F1: PCA-spectral 0.558, IsoForest-conformal 0.519, PCA-bagging 0.519, IsoForest-spectral 0.493, PCA-raw 0.490, distance-profile 0.467, forecasting-naive 0.443, LSTM-AE 0.260, LSTM-DBSCAN 0.225, OneClassSVM 0.000.
- Honest caption under the table: 0.60 cross-group supervised with AUC 0.937, 0.909 XGB and 0.905 LGBM in-distribution, and 0.985 random-window leakage are context only, not the deployed model.
- The 0.60, 0.909, 0.905, and 0.985 numbers are not retained summary rows and must not be sold as the deployed model.

**Speaker cue:** Be honest: 0.558 is the artifact we keep, and the ranked table explains why that is still the defensible Day-1 choice.

## Slide 13 — Why PCA-Spectral as Day-1 Champion

**Title:** Why PCA-Spectral as Day-1 Champion

**Key message:** PCA-spectral is the Day-1 champion because cheap, deterministic retraining fits an industrial continuous-learning loop, not only because of its F1.

**Slide bullets (cost + continuous-learning rationale):**

- Continuous learning retrains on a cadence (scheduled ~30 min + drift-triggered, PCA path), so the detector must be cheap and deterministic.
- PCA retrains in one full-SVD pass with closed-form thresholds: no epochs, no gradient descent, no hyperparameter search, reproducible for the champion-challenger gate.
- It also deploys at a normal-only cold start, where supervised XGBoost/LightGBM cannot even train.
- Fast, decomposable, operator-explainable; the mathematical low-cost proof is on the next slide.

**Speaker cue:** Lead with the operational reason: the champion is the model cheap enough to retrain on cadence and reproducible enough to promote safely.

## Slide 14 — PCA Mechanics

**Title:** PCA Mechanics

**Key message:** One deterministic full SVD trains the detector and everything after is closed-form, which is exactly what makes retraining cheap.

**Slide bullets (low-cost proof + cost-contrast chart `pca-vs-lstm-retrain-cost.png`):**

- Train once: fit RobustScaler, then one full SVD on normal windows (n=9346 rows, d=64 spectral features, n_components=0.9 keeps 90% variance).
- Training cost is that SVD: O(n*d^2 + d^3); thresholds are p95 empirical quantiles (one O(n log n) sort). No epochs, no backprop, no search.
- Inference per window = project then reconstruct = O(d*k), k = kept components; deterministic and CPU-only.
- Contrast chart: LSTM-AE trains ~100 epochs of backprop per retrain (config epochs=100) vs PCA's single SVD; structural pass count, not measured wall-clock.

**Speaker cue:** This is the math that proves low cost: one SVD plus closed-form scoring, orders of magnitude cheaper to retrain than a 100-epoch autoencoder.

## Slide 15 — T2 vs Q/SPE

**Title:** T2 vs Q/SPE

**Key message:** T2 catches extreme-but-on-pattern windows, while Q/SPE catches off-pattern residuals.

**Slide bullets:**

- T2 = sum(score^2 / explained_variance): extreme but still on a pattern PCA already knows.
- Q/SPE = squared reconstruction residual: the relationship itself changed.
- Decision is OR with strict `>`: anomaly if T2 > threshold OR Q > threshold; thresholds are p95 quantiles of normal.
- Reported score = max(T2/T2_thr, Q/Q_thr); top sensor is the largest residual contributor, not a proven root cause.
- Both statistics reuse one projection, so scoring a window is O(d*k): live inference adds negligible cost at the ~1 Hz cadence.

**Speaker cue:** Use a careful phrase: top contributing sensor helps triage, it doesn't prove physical causality.

## Slide 16 — Spectral FFT

**Title:** Spectral FFT

**Key message:** FFT features give the model rhythm information, but the dominant frequency is intuition only in this deck.

**Slide bullets:**

- Each window is mean-centered per sensor, then summarized by mean, std, range, FFT band energies, and spectral centroid.
- Bands are equal splits of FFT energy bins (np.array_split, not named Hz bands); feature count = sensors x (n_bands + 4) = 64.
- Dominant frequency is taught as intuition only; it is not a live feature column.
- A rhythm shift can move band energy before the raw level moves (illustrative, not a measured lead time).
- Feature build is cheap: per-window FFT is O(d log d) per channel, so spectral awareness does not change the low-cost profile.

**Speaker cue:** Don't over-formalize the FFT slide. It is here to make spectral features understandable.

## Slide 17 — What Is MLOps + Why It Helps Us

**Title:** What Is MLOps + Why It Helps Us

**Key message:** MLOps is the lifecycle around a model: train, deploy, monitor, compare, promote, and retrain.

**Slide bullets:**

- General definition: model lifecycle operations across training, deployment, monitoring, retraining, and promotion.
- Our case: cold-start champion, drift reports, retrain jobs, champion-challenger comparison, and operator evidence.
- Demo path can move a model without an app restart, but hot-swap is process-local.
- Tools in this project: MLflow, Evidently, Prometheus, and Grafana.

**Speaker cue:** This is a definition-first support slide. Keep it tied to the anomaly story, not tool tourism.

## Slide 18 — MLflow: Registry + Champion-Challenger Gate

**Title:** MLflow: Registry + Champion-Challenger Gate

**Key message:** MLflow stores model versions and the `@champion` alias, while custom project logic decides whether the alias should move.

**Slide bullets:**

- MLflow feature used here: Model Registry with versioned models and a `@champion` alias.
- Custom project feature: champion-challenger gate that drives the alias move.
- The gate is not an MLflow feature.
- Gate rule: `f1_chal > f1_champ + 0.02 AND far_chal <= far_champ * 1.05`.
- Cold start loads `PUMPAD_MODEL_DIR` first, then MLflow `@champion`.
- Optional alias-refresh scheduler exists but is off by default.

**Speaker cue:** Separate the tool from the project policy. MLflow is the registry; the gate is our code.

## Slide 19 — RouteMQ: Connect It All Together

**Title:** RouteMQ: Connect It All Together

**Key message:** One MQTT application framework wires ingest, ML inference, storage, and the operator view into a single flow.

**Presentation: hero architecture map (`routemq-connect-map.png`), no bullets on the slide; the map carries the story.**

- Sync path: MQTT telemetry -> Router -> Middleware -> Controller -> publish anomaly, then persist (Redis, ClickHouse).
- Async path: Queue/Worker runs MLOps jobs (drift, retrain) and writes the MLflow registry; the Controller loads the champion from MLflow.
- Dashboard and observability read the same Redis/ClickHouse evidence.

**Speaker cue:** Use the map to set up the next four layer slides; everything downstream shares one source of truth.

## Slide 20 — Layer 1: Router (Ingest and Dispatch)

**Title:** Layer 1 - Router: Ingest and Dispatch

**Key message:** The Router is the entry point of the synchronous path; it only resolves topic to handler.

**Slide bullets:**

- Subscribes factory/skab/{station}/telemetry at QoS 1.
- Declarative routing dispatches the matched topic to anomaly_controller.ingest.
- A per-station wildcard keeps one route for every pump station.
- Nothing heavy here; routing is just topic-to-handler resolution.

**Speaker cue:** Keep this short; it is the doorway, not the work.

## Slide 21 — Layer 2: Middleware (Validate, Rate-limit, Correlate)

**Title:** Layer 2 - Middleware: Validate, Rate-limit, Correlate

**Key message:** An ordered middleware chain runs before the controller and protects inference.

**Slide bullets:**

- ValidateTelemetryMiddleware rejects non-JSON payloads and empty sensor maps.
- InMemoryRateLimitMiddleware caps 50 messages per second per station.
- CorrelationLoggingMiddleware attaches a correlation id for end-to-end tracing.
- Order matters: bad or excess traffic is dropped before it reaches the model.

**Speaker cue:** Stress that this is a guardrail layer, not business logic.

## Slide 22 — Layer 3: Controller, Inference, Persistence

**Title:** Layer 3 - Controller, Inference, Persistence

**Key message:** The synchronous core decides, publishes, and stores in one pass.

**Slide bullets:**

- anomaly_controller calls the inference service, which loads the champion and scores the window (PCA T2/Q).
- Inference is in-path and synchronous; it emits an AnomalyVerdict, not a queued job.
- It publishes to factory/skab/{station}/anomaly at QoS 1.
- Persistence is Redis latest-state first, then ClickHouse history.

**Speaker cue:** This is where the anomaly decision actually happens; everything else supports it.

## Slide 23 — Layer 4: Queue/Worker (MLOps Jobs)

**Title:** Layer 4 - Queue/Worker: MLOps Jobs

**Key message:** The asynchronous lane is where continuous learning lives, off the inference path.

**Slide bullets:**

- DriftReportJob and RetrainingJob run off the queue, not in the anomaly inference path.
- They read ClickHouse/SKAB data and write the MLflow registry plus Redis report keys.
- The Controller picks up the new champion; promotion is gated by the custom champion-challenger rule.
- The scheduler is off by default (env flags), so the loop is opt-in.

**Speaker cue:** Tie back to slides 13-14: cheap deterministic retraining is what makes this lane affordable.

## Slide 24 — Operator Dashboard (Streamlit)

**Title:** Operator Dashboard (Streamlit)

**Key message:** The operator dashboard has seven pages with live monitoring, model evidence, drift reports, health, and runbook support.

**Slide bullets:**

- Page titles: `PDAM Pump Sentinel`, `Live Sensor Monitoring`, `Anomaly History`, `Model Registry`, `Drift & Retrain Reports`, `System Health`, `Operator Runbook`.
- `Live Sensor Monitoring` refreshes every 5 seconds.
- `System Health` refreshes every 10 seconds.
- Other pages are manual or cache-TTL based.
- Operator actions ack, mute, and note write operator action state to Redis plus ClickHouse, not training labels.
- ack/note TTL is 30 days, mute TTL is 15 minutes.
- A separate label route exists for training labels, so don't conflate operator notes with labels.

**Speaker cue:** Show the screenshot from the appendix. Say this is an operator console, not just a chart page.

## Slide 25 — What Is Observability

**Title:** What Is Observability

**Key message:** Observability means reading system state from outside the running service through metrics, logs, and traces.

**Slide bullets:**

- Metrics: numeric time-series signals for health, rate, latency, freshness, and errors.
- Logs: timestamped event records for investigation.
- Traces: request or message paths across components.
- The goal is to answer new questions about a live system without changing code first.

**Speaker cue:** This slide defines the concept before judging what the project actually proves.

## Slide 26 — Observability in Our AIoT Case

**Title:** Observability in Our AIoT Case

**Key message:** The implementation proves metrics, dashboards, and local alert rules for the AIoT demo path.

**Slide bullets:**

- Prometheus collects `pumpad_*` metrics plus RouteMQ `routemq_*` metrics.
- The `pumpad_*` catalog has 15 metrics.
- Prometheus has 2 scrape jobs: `routemq-app` on app:8080 and `mosquitto-exporter` on 9234.
- Grafana dashboards cover RouteMQ Observability, MLOps Loop, System Health, and MQTT Broker, all refreshing every 10 seconds.
- Local alert group `pumpad-local-observability` has 7 rules: AppMetricsDown, TelemetryStale, InferenceErrors, PersistenceWriteErrors, DriftReportStale, ActiveModelStale, and HighSeverityAnomalyEvents.
- Honest boundary: this is metrics plus dashboards plus local alert rules only. Centralized logs, distributed traces, Alertmanager/PagerDuty routing, Kubernetes, and cloud production are future work.
- Use the phrase local Docker Compose evidence when presenting this slide.

**Speaker cue:** Show the Grafana screenshot from the appendix and keep the claim precise.

## Slide 27 — Closing: Limitations + Roadmap

**Title:** Closing: Limitations + Roadmap

**Key message:** The MVP is a working local anomaly-detection and MLOps demo, with clear production gaps and a concrete roadmap.

**Slide bullets:**

- Current limits: synchronous inference, schedulers behind env flags, process-local hot-swap, and local Docker Compose deployment.
- Scheduler defaults: retrain 30 minutes, drift 15 minutes, model refresh 5 minutes, all off by default unless enabled.
- Retraining path is PCA by default for the demo, while code also supports lstm_ae, xgboost, lightgbm, and live_clickhouse options.
- Roadmap: operator labels, supervised promotion gates, incident routing, centralized logs/traces, and production deployment hardening.
- Closing thesis: the project proves the anomaly-detection lifecycle honestly, from data split to champion artifact to operator evidence.

**Speaker cue:** Close with confidence and boundaries: it works locally, it is honest about what it doesn't prove yet, and the next steps are specific.

## Canonical Screenshots Used

- `docs/presentation/screenshots/t9-observability-streamlit-observability-snapshot-20260628T0535Z.png`
- `docs/presentation/screenshots/t9-observability-grafana-pipeline-observability-20260609T130723Z.png`
