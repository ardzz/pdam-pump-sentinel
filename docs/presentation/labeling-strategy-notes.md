# Labeling Strategy & Hybrid Roadmap — Speaker Notes

Supporting material for the deck section defending the PCA-spectral champion choice and the multi-stage roadmap toward supervised learning. Every claim is backed either by industry documentation, peer-reviewed academic citation, or an exact `file:line` in this repository.

## Core argument (slide headline)

> Champion PCA-spectral dipilih bukan karena PCA paling akurat, tapi karena **deployable dari Day-1 tanpa labeled data** — kondisi realistis PDAM. Arsitektur project sudah memiliki hooks untuk evolusi smooth menuju supervised begitu labeling pipeline matang. Ini pattern standar industri, bukan workaround.

## 1. Industry evidence — 6 vendor PdM platforms converge

| Vendor | URL | Cold-start | Labeling source |
|---|---|---|---|
| AWS Lookout for Equipment | https://docs.aws.amazon.com/lookout-for-equipment/latest/ug/understanding-labeling.html | Asumsi input = normal operation, learn deviations dari baseline | SME + work orders; labels OPTIONAL |
| Hitachi Lumada | https://www.hitachi.com/en/press/articles/2018/10/1004/ | ART clustering pada past normal operation | Operator feedback loop |
| GE Vernova SmartSignal | https://www.gevernova.com/software/products/asset-performance-management/equipment-downtime-predictive-analytics | Digital-twin blueprint vs predicted normal (explicit: "not by thresholds") | Failure-mode knowledge + fleet history; $1.6B losses avoided, 3.41-month average ROI, $60M average loss avoided per catch |
| Siemens MindSphere PSA | https://press.siemens.com/global/en/pressrelease/mindsphere-application-predictive-service-assistance-uses-artificial-intelligence | Evolved from KPI limit values to neural-network AI module | Not explicitly documented |
| Asystom Advisor | https://www.asystom.com/solutions/asystomadvisor | Automated learning per device (vibration, ultrasound, temperature) | Probable root cause output; claims detection up to 2 months in advance |
| Uptake Fleet | https://uptake.com/blog/planning-for-a-predictive-maintenance-program/ | Dashboarding + connector verification first | Work-order + parts-order mining; label-correction engine fixes ~15% root-cause bin shift; up to 10 years WO history |

### Cross-cutting patterns (docs converge)

- **Normal-baseline-first adalah convention, bukan shortcut** — AWS, Hitachi, GE, dan Asystom semua belajar "normal" sebelum mendeteksi deviation.
- **Human / CMMS labels datang BELAKANGAN** — AWS SMEs, Hitachi operator decisions, Uptake WO-mining, Augury analyst-labeled vibration datasets.
- **Thresholds sering coexist atau precede AI** — Siemens explicit migrasi KPI limits → neural network; Uptake masih pakai thresholds pada survival curves; GE contrasts model output dengan simple thresholds.
- **Docs TIDAK converge** pada lifecycle bulan-spesifik (month-0 / month-6 / month-12) — itu framing kita sendiri, bukan klaim industri.

### Watchouts (untuk Q&A defense)

- **Work-order labels NOISY.** AWS sendiri menyebut WO "subjective and inconsistent"; Uptake's label-correction engine bisa shift root-cause bin ~15%.
- **Supervised model per failure mode UNREALISTIC.** Uptake explicit: failure modes sisanya tetap di-handle dengan conventional maintenance, bukan ML.
- **Evidence gaps:** tidak ada vendor publik yang quote angka spesifik "F1 improved from X to Y after N months". Klaim lifecycle adalah inferensi pattern, bukan claim numerik.

## 2. Academic backing — 4 peer-reviewed papers

| Sub-topic | Citation | Quotable claim | Relevance to project |
|---|---|---|---|
| Weak supervision for PdM | Martínez-Heredia, *WIREs Data Mining and Knowledge Discovery* 2025 — https://doi.org/10.1002/widm.70022 | Weak supervision is a practical PdM strategy when labels are incomplete, imprecise, or noisy rather than fully curated. | Pump alarms, maintenance logs, dan operator notes bisa jadi weak labels untuk TS-AD windows sebelum full supervision tersedia. |
| Active learning streaming TS-AD | Holtz, *Flexible Services and Manufacturing Journal* 2025 — https://doi.org/10.1007/s10696-024-09588-0 | Budgeted expert feedback can improve industrial time-series anomaly detection over unsupervised baselines while reducing labeling burden. | Cocok untuk operator-in-the-loop roadmap di mana hanya selected PCA T² alerts yang di-review. |
| Self-training pseudo-labels | Yoon (SPADE), *TMLR* 2023 — https://arxiv.org/abs/2212.00173 | Ensembles of one-class detectors can act as pseudo-labelers for semi-supervised anomaly detection under limited labels. | PCA T², IF, atau autoencoder scores bisa bootstrap pseudo-labels untuk supervised pump classifier, tapi WAJIB ada drift checks. |
| TS-AD survey | Schmidl, *PVLDB* 2022 — https://doi.org/10.14778/3538598.3538602 | No single TS-AD algorithm dominates across datasets, metrics, and anomaly types. | Justifies keeping PCA T² sebagai champion baseline sambil benchmarking supervised successors. |

### Risk acknowledgement (literature-confirmed)

- **Operator labeling fatigue** — Holtz 2025 mendokumentasikan label expense dan butuh budgeted expert feedback.
- **LF coverage gaps** — Martínez-Heredia 2025 mendokumentasikan incomplete/imprecise/noisy weak labels dalam PdM.
- **Confirmation bias in self-training** — Yoon 2023 mendokumentasikan pseudo-label bias under distribution mismatch.

## 3. SKAB methodology — bahkan academic benchmark butuh labeling effort

> NOTE: Section ini berbasis dataset README + general knowledge tentang SKAB, **bukan** librarian-verified deep dive (sub-agent untuk SKAB-spesifik timeout).

**SKAB (Skoltech Anomaly Benchmark)** — Iurii Katser & Vyacheslav Kozitsin (Skoltech). Repository: https://github.com/waico/SKAB.

- Closed-loop water-circulation testbed di laboratorium Skoltech.
- Multivariate sensor data: pressure, flow, temperature, vibration, motor electrical.
- Label granularity: range-based — fault start/end timestamp di-record per fault episode.
- **Source of truth untuk labels: controlled fault injection di lab**. Fault categories (`valve1`, `valve2`, `other`, `anomaly-free`) merefleksikan injection regime — operator/peneliti sengaja closure valve, induce cavitation, dsb sesuai eksperimen design.

**Implikasi metodologis:** SKAB labels BUKAN ground truth magis. Mereka berasal dari **strategi labeling no. 5 (lab/digital twin simulation)** — paling clean tapi paling unrepresentative untuk noisy production environment. Untuk PDAM real deployment, labels akan datang dari **strategi 1 (operator-in-the-loop)** + **strategi 4 (CMMS mining)**, yang lebih noisy tapi reflektif kondisi real.

### Cross-benchmark labeling (general knowledge)

| Benchmark | Domain | Labeling source |
|---|---|---|
| MIMII | Industrial machine sounds | Controlled fault injection di lab |
| SMAP / MSL | NASA spacecraft telemetry | Engineer-confirmed anomaly windows dari ops log |
| SWaT / WADI | SUTD water treatment testbed | Controlled attack scenarios |
| NAB | Real production data (servers, sensors) | Domain expert post-hoc labeling |

**Konklusi:** Semua widely-used industrial TS-AD benchmarks memerlukan manual atau experimentally-controlled labeling effort. **Datasets with magical ground truth tidak eksis di space ini.** Klaim "kami tidak punya labels jadi tidak bisa pakai supervised" adalah valid bukan excuse — itu kondisi normal untuk industrial PdM.

## 4. Repo proof-points (exact file:line — bisa di-demo saat Q&A)

| Capability | Implementation | Status |
|---|---|---|
| Champion-challenger gate | `ml/monitoring/champion_challenger.py:8-32` — `should_promote(f1_margin=0.02, far_guard=1.05)` requires challenger F1 > champion + 0.02 AND FAR <= champion_far * 1.05 | wired |
| Gate invocation in retrain flow | `app/jobs/retraining_job.py:33-44` | wired |
| Family-aware inference loader | `ml/inference/loader.py:17-35` — dispatch by `metadata.model_family` (pca, isolation_forest, lstm_ae, xgboost, lightgbm) | wired |
| Hot-swap mechanism | `app/services/inference.py:22-37` + `app/jobs/retraining_job.py:114-121` | wired |
| MLflow `@champion` alias promotion | `app/jobs/retraining_job.py:124-143`; seed `scripts/seed_initial_models.py:32-47` | wired |
| Generic alias setter | `ml/registry/mlflow_client.py:469-490` | wired |
| MLflow 3.12 version-resolution workaround | `ml/registry/mlflow_client.py:493-503` — `MlflowClient.search_model_versions(filter_string="run_id='...'")` because `model_info.registered_model_version is None` in MLflow 3.12 | wired |
| Hot-swap consumer (live inference) | `app/controllers/anomaly_controller.py:26-33` | wired |
| APScheduler retraining job | `ml/monitoring/scheduler.py:15-40`; enabled in `bootstrap/app.py:185-187` | wired |
| Dataset path for retrain | `app/jobs/retraining_job.py:69-90` — `PUMPAD_SKAB_INPUT_PATH` env, optional validation/manifest paths | wired |
| Evidently drift report | `ml/monitoring/drift_check.py:21-40` — `Report([DataDriftPreset(drift_share=...)])` | wired |
| Drift triggers retrain | `app/jobs/drift_report_job.py:41-43` dispatches `RetrainingJob()` | wired |
| Per-family training entry points | `ml/training/train_pca.py:87-91`, `train_lstm_ae.py:70-74`, `train_isoforest.py:69-73`, `train_supervised.py:82-86` | wired |

## 5. Honest gaps (untuk slide "Future work" — JUJUR DI DEPAN REVIEWER)

| Gap | Suggested implementation location |
|---|---|
| Operator label intake (dashboard buttons / Redis queue / ClickHouse table) | `app/controllers/label_controller.py` (NEW) + extend `infra/clickhouse/init.sql` + add `dashboard/pages/2_anomaly_history.py` triage UI |
| Scheduled retrain hanya PCA-family; supervised promotion deferred | `app/jobs/retraining_job.py:76-90` extension untuk dispatch ke supervised training |
| Supervised training menerima `alias` arg tapi belum set alias di MLflow registry | `ml/training/train_supervised.py:816-850` |
| MLflow alias changes belum di-poll oleh live app (manual hot-swap only) | `app/services/inference.py` add reload polling thread |
| `DriftReportJob` ada tapi belum di-schedule via APScheduler | `ml/monitoring/scheduler.py` add drift hook |

**Honest framing for the slide:** "Roadmap arsitektur lengkap; 5 hook implementasi adalah follow-up post-MVP. Setiap gap punya specific file:line target — bukan handwave."

## 6. Suggested slide narrative (speaker script)

> "Kenapa PCA champion, bukan XGBoost yang F1-nya 0.91?
>
> Karena 0.91 itu **in-distribution** — labeled examples dari setiap fault type harus ada di train. Di PDAM real, day-1 tidak ada labeled fault inventory. Industri-pun — AWS Lookout, Hitachi Lumada, GE SmartSignal — semua mulai dengan normal-baseline learning, baru supervised setelah labeling pipeline jalan berbulan-bulan.
>
> Kami pilih PCA-spectral karena: deployable Day-1, decomposable per-sensor signature (operator bisa tahu sensor mana yang anomali), inference latency µs-scale.
>
> Tapi kami **tidak terjebak di PCA**. Lihat `champion_challenger.py:8-32` — gate `should_promote` siap auto-promote LSTM-AE atau XGBoost begitu metric-nya beat PCA dengan margin 0.02 + FAR ratio < 1.05. `loader.py:17-35` family dispatch sudah siap untuk 5 famili.
>
> Hook yang BELUM ada — operator label intake — kami list eksplisit sebagai future work. Bukan ditutupi. Roadmap-nya: month-3 dashboard triage button → ClickHouse labels table → retraining job pickup labels → supervised challenger trained → gate auto-promote.
>
> Honest, defensible, dan in line dengan industry practice."

## References

- Industry: see Section 1 URL list.
- Academic: Martínez-Heredia 2025; Holtz 2025; Yoon (SPADE) 2023; Schmidl et al. 2022 PVLDB.
- SKAB: https://github.com/waico/SKAB (dataset repository + paper).
- Project file refs: see Section 4 — all paths relative to repo root.
