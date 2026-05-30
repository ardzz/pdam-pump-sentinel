# PDAM Pump Sentinel — Design Document

**Judul Project:** PDAM Pump Sentinel: Platform MLOps untuk Predictive Maintenance Pompa Distribusi Air berbasis MQTT Framework RouteMQ dengan Continuous Training

**Tanggal:** 2026-05-29
**Status:** Design — validated through evidence-driven brainstorming
**Tim:** 4 anggota
**Durasi:** 5 minggu pengerjaan + 1 minggu buffer
**Konteks:** Tugas Besar Project AI/IoT — kelompok 3–4 mahasiswa

---

## 1. Ringkasan Eksekutif

PDAM Pump Sentinel adalah platform AIoT end-to-end untuk **predictive maintenance pompa distribusi air bersih**, mengkombinasikan tiga pilar engineering yang dipresentasikan dalam rasio 40/60:

1. **DevOps (40%)** — Containerized multi-service architecture berbasis **RouteMQ**, sebuah MQTT application framework custom-built bergaya Laravel/Django dalam Python 3.12+. Mencakup Docker Compose orchestration, GitHub Actions CI, observability hooks, dan horizontal scaling via MQTT shared subscriptions.

2. **AI/ML & MLOps (60%)** — Multivariate anomaly detection menggunakan **PCA T²/Q** sebagai champion dan **LSTM Autoencoder** sebagai challenger, dibungkus dalam continuous training lifecycle: MLflow Model Registry dengan alias `@champion`/`@challenger`, Evidently AI untuk drift monitoring, dan scheduled retraining setiap 30 menit (demo mode) dengan champion-challenger evaluation otomatis.

Dataset surrogate: **SKAB (Skoltech Anomaly Benchmark)** — water circulation testbed industri nyata dari Skolkovo Institute of Science and Technology, dengan 8 channel sensor (vibration, pressure, current, flow rate, temperature) dan skenario fault realistis (valve closing, leak, rotor imbalance, cavitation).

Framing aplikasi: **sistem pemantauan kondisi pompa pada Instalasi Pengolahan Air (IPA) PDAM**.

---

## 2. Latar Belakang Masalah

### 2.1 Masalah Industri Nyata

Pompa sentrifugal adalah aset kritis dalam sistem distribusi air bersih PDAM. Kegagalan tak terduga dari pompa menyebabkan:

- Ribuan pelanggan kehilangan akses air bersih dalam hitungan jam
- Biaya emergency dispatch maintenance yang signifikan
- Cascade impact pada tekanan pipa downstream
- Reputasi layanan publik menurun

Konteks industri lebih luas: **Siemens True Cost of Downtime 2024** melaporkan downtime industri otomotif mencapai USD 2.3 juta per jam, dengan kerugian tahunan Fortune Global 500 akibat unplanned downtime mendekati USD 1.4 triliun.

### 2.2 Keterbatasan Threshold Konvensional

Sistem monitoring berbasis batas min/max gagal mendeteksi tiga kelas anomali penting:

- **Contextual anomalies** — nilai sensor "normal" secara batas, tapi abnormal pada konteks operasinya (mis. suhu tinggi saat beban rendah)
- **Sensor drift** — pembacaan menyimpang perlahan karena aging sensor
- **Collective anomalies** — pola degradasi gradual lintas beberapa sensor sekaligus

SKAB secara eksplisit memisahkan outlier detection dan changepoint/collective anomaly sebagai dua kelas masalah berbeda.

### 2.3 Concept Drift dalam ML Production

Model anomaly detection yang dilatih sekali akan mengalami **concept drift** seiring usia sensor dan perubahan musim. Tanpa pipeline MLOps untuk retraining otomatis, akurasi model menurun dalam hitungan minggu — phenomenon yang didokumentasikan di banyak literatur MLOps [Kreuzberger 2022, Bayram 2024].

---

## 3. Tujuan & Scope

### 3.1 Tujuan Utama

1. Membangun MQTT application framework (**RouteMQ**) sebagai backbone ingestion sensor dengan routing DSL, middleware chain, queue workers, Redis/MySQL integration, dan shared-subscription horizontal scaling.
2. Mengimplementasikan multivariate anomaly detection dengan PCA T²/Q (champion) dan LSTM Autoencoder (challenger) pada 8 channel sensor SKAB.
3. Membangun MLOps loop lengkap: offline training pipeline, MLflow Model Registry, Evidently drift monitoring, scheduled retraining dengan champion-challenger evaluation.
4. Mendemonstrasikan **continuous training loop** end-to-end: drift detection → retraining trigger → model promotion → hot-swap inference.
5. Menyediakan dashboard operator real-time + interface MLflow registry.

### 3.2 Scope Demo MVP

Yang harus jalan saat presentasi:

- SKAB CSV replay → MQTT broker → RouteMQ routes → MySQL/Redis persistence
- PCA champion + LSTM-AE challenger inference jobs
- Scheduled retraining (30 menit demo cadence)
- Dashboard Streamlit dengan 4 halaman: live sensors, anomaly history, model registry, drift reports
- Demo skenario MLOps lengkap: drift injection → retraining trigger → model v2 promoted → akurasi recover

### 3.3 Out of Scope

- Real ESP32 sensor (kami pakai dataset replayer)
- Kubernetes deployment (cukup Docker Compose)
- A/B testing real production traffic (cukup champion-challenger in-registry swap)
- Multi-tenant operations

---

## 4. Use Case & Manfaat Stakeholder

| Stakeholder | Manfaat Konkret |
|---|---|
| Operator IPA | Alert dini sebelum kegagalan; dashboard real-time kondisi pompa |
| Manager PDAM | Laporan mingguan kondisi aset; reduksi unplanned downtime |
| Pelanggan rumah tangga | Continuity layanan air bersih |
| Engineer maintenance | Identifikasi mode kegagalan akurat → perbaikan tepat sasaran |
| Data scientist internal | MLflow registry → eksperimen model baru aman dengan staging promotion |

---

## 5. Arsitektur Sistem

### 5.1 Arsitektur 3-Pilar

```text
┌─────────────────────────────────────────────────────────────┐
│                      DEVOPS LAYER                            │
│   Docker Compose · GitHub Actions CI · .env · Healthcheck   │
│   Prometheus + Grafana observability                        │
└─────────────────────────────────────────────────────────────┘
        │
        ↓
┌─────────────────────────────────────────────────────────────┐
│                  ROUTEMQ APPLICATION                         │
│  Router → Middleware → Controller → Queue → Worker          │
│  + observability hooks → Prometheus → Grafana               │
└─────────────────────────────────────────────────────────────┘
        ↑                                       ↓
        │ inference                             │ metrics
        │                                       │
┌─────────────────────────────────────────────────────────────┐
│                      MLOPS LAYER                             │
│                                                              │
│  Offline:   SKAB training data                              │
│             ↓                                                │
│             Feature pipeline                                 │
│             ↓                                                │
│             Train PCA T²/Q + LSTM-AE → MLflow tracking      │
│             ↓                                                │
│             Model Registry (MLflow @champion/@challenger)   │
│                                                              │
│  Online:    InferenceJob loads model from registry           │
│             ↓                                                │
│             Score + drift check (Evidently)                  │
│             ↓                                                │
│             Publish anomaly + drift metrics                  │
│                                                              │
│  Loop:      Scheduled retrain (30m) → champion-challenger    │
│             → if F1↑ promote v+1 → hot-swap via Redis        │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Data Flow End-to-End

```text
SKAB Dataset Replayer
   ↓ MQTT publish 1 Hz (demo cepat: 10 Hz)
[ Eclipse Mosquitto Broker ]
   ↓
[ RouteMQ Application ]
   ├─ Router.dispatch → factory/skab/{station}/telemetry
   ├─ Middleware chain
   │    ├─ ClientRateLimitMiddleware (Redis sliding window)
   │    ├─ Payload validation
   │    └─ Logging + correlation ID
   ├─ SensorController.ingest()
   │    ├─ Redis: latest reading per sensor (set_json)
   │    └─ MySQL: append historical row (sensor_readings table)
   └─ QueueManager.dispatch(AnomalyDetectionJob)
        ↓
[ QueueWorker (Redis driver) ]
   ├─ AnomalyDetectionJob.handle()
   │    ├─ Load active model URI dari Redis cache
   │    ├─ Load PCA T²/Q model (champion) dari MLflow Registry
   │    ├─ Compute T² + Q statistics dari window
   │    ├─ Adaptive EWMA threshold
   │    └─ Publish anomaly ke factory/skab/{station}/anomaly
   ↓
[ Streamlit Dashboard ]
   ├─ Live sensor charts + anomaly markers
   ├─ Anomaly score timeline + threshold
   ├─ Model registry status (PCA v1 → v2 → ...)
   └─ Evidently drift report viewer
```

### 5.3 Komponen DevOps Layer

- **Orchestration**: Docker Compose (production-ish) + Compose dev (Redis/MySQL only)
- **CI/CD**: GitHub Actions — lint (Ruff) + unit tests (unittest) + build images
- **Observability**: Prometheus metrics scraped dari RouteMQ `observability.py` hooks; Grafana dashboards untuk system metrics + ML metrics (anomaly score, model version, drift score) dalam satu pipeline
- **Config**: `.env` based environment variables, `ENABLE_REDIS`/`ENABLE_MYSQL` feature flags
- **Healthchecks**: RouteMQ built-in `/health` dan `/ready` endpoints

### 5.4 Komponen RouteMQ Application

- **Router** (`routemq/router.py`) — Laravel-style DSL: `router.on("factory/skab/{station}/telemetry", ...)`
- **Dynamic discovery** (`routemq/router_registry.py`) — auto-load semua module dari `app/routers/`
- **Middleware** — `RateLimitMiddleware`, `TopicRateLimitMiddleware`, custom validation
- **Controllers** (`app/controllers/`) — async static handlers
- **Queue** (`routemq/queue/`) — Redis + Database driver, allowlist-secured `Job.register`
- **Worker scaling** (`routemq/worker_manager.py`) — `multiprocessing.Process` per shared subscription
- **Observability** (`routemq/observability.py`) — backend-neutral metric/trace hooks
- **CLI**: `routemq run`, `routemq queue-work`, `routemq tinker`

### 5.5 Komponen MLOps Layer

- **Offline training**: SKAB normal data → feature engineering → train PCA + LSTM-AE → MLflow tracking + registry push
- **Online inference**: AnomalyDetectionJob load model via `models:/PumpAD@champion`, hot-swap via Redis pub/sub invalidation
- **Drift monitoring**: Evidently `Report(metrics=[DataDriftPreset()])` dijalankan setiap batch
- **Retraining**: APScheduler 30-menit cron → RetrainingJob → champion-challenger evaluation
- **Promotion**: jika F1_challenger > F1_champion + 0.02, panggil `client.set_registered_model_alias("PumpAD", "champion", v2)`
- **Rollback**: keep 3 versi terakhir; manual rollback button di dashboard

---

## 6. Stack Teknologi

| Layer | Tools |
|---|---|
| MQTT broker | Eclipse Mosquitto 2.x |
| Application framework | **RouteMQ** (custom, Python 3.12+) |
| Database | MySQL 8 (historis sensor + queue jobs) |
| Cache + queue | Redis 7 (latest readings + active model URI + Redis queue) |
| ML — Champion | scikit-learn (PCA, RobustScaler) + custom T²/Q calculator |
| ML — Challenger | TensorFlow 2.15+ / Keras (LSTM Autoencoder) |
| Experiment tracking + registry | **MLflow** (SQLite backend untuk demo) |
| Drift monitoring | **Evidently AI** |
| Dashboard | **Streamlit** multi-page |
| Observability | Prometheus + Grafana |
| Orchestration | Docker Compose |
| CI/CD | GitHub Actions (lint + unit tests + image build) |
| Scheduler | APScheduler (in-process RouteMQ) |
| Dataset | SKAB (GPL-3.0) |

---

## 7. Algoritma Anomaly Detection

### 7.1 Pemilihan Algoritma — Justifikasi Berbasis Evidence

Kami memilih **PCA T²/Q sebagai champion** dan **LSTM Autoencoder sebagai challenger** berdasarkan tiga pertimbangan:

1. **SKAB official leaderboard** (commit `b2c0d46c` di `waico/SKAB`) menunjukkan:
   - PCA T²+Q: F1 = 0.76
   - LSTM-AE: F1 = 0.74
   - Conv-AE: F1 = 0.78
   - Isolation Forest: F1 = 0.29 (ditolak sebagai pilihan utama)

2. **Bukti water domain langsung**: [Githinji et al. 2023, IEEE AFRICON] secara eksplisit menggunakan deep LSTM Autoencoder pada IoT water level sensors deployed on a water catchment — prior art sempurna untuk framing PDAM.

3. **PCA T²/Q sangat matang** di process monitoring industri: [Bakdi et al. 2018], [Kazemi et al. 2024], dan banyak lainnya.

### 7.2 Champion: PCA T²/Q

- **Algoritma**: Hotelling T² (statistik sub-space principal) + Q-statistic (Squared Prediction Error pada residual sub-space)
- **Threshold**: adaptive via EWMA-based control limit [Bakdi 2018]
- **Training**: fit pada data normal-only (sliding window 60 samples)
- **Inference**: O(features × components) — sangat cepat
- **Explainability**: T² chart + Q chart langsung dipresentasikan di dashboard operator
- **Library**: `sklearn.decomposition.PCA` + custom T²/Q calculator (~30 baris)
- **Retraining cost**: detik, cocok untuk scheduled cron

### 7.3 Challenger: LSTM Autoencoder

- **Arsitektur**: encoder (LSTM → bottleneck) → decoder (LSTM → output)
- **Loss**: MSE reconstruction error
- **Anomaly score**: reconstruction error per window
- **Threshold**: 99th percentile dari error pada validation set normal
- **Training**: pada data normal-only, epochs 20–50, batch 32, CPU-feasible
- **Explainability**: per-sensor reconstruction error breakdown
- **Library**: TensorFlow 2.15 / Keras (CPU build)
- **Retraining cost**: 5–15 menit per cycle (acceptable untuk 30-menit cron)

### 7.4 Feature Engineering

- **Sliding window**: 60 samples per window (≈1 menit @ 1 Hz)
- **Stride**: 30 samples (overlap 50%)
- **Per-sensor normalization**: `RobustScaler` fitted pada normal training data
- **Rolling statistics tambahan untuk PCA**: mean, std, lag-1, lag-2

### 7.5 Champion-Challenger Decision Rule

```python
# Pseudo-code
hold_out = labeled_skab_segments[20%]
f1_champion = evaluate(current_champion_model, hold_out)
f1_challenger = evaluate(retrained_challenger_model, hold_out)

if f1_challenger > f1_champion + 0.02:
    mlflow_client.set_registered_model_alias(
        "PumpAD", "champion", new_version
    )
    redis.publish("model:invalidate", new_uri)
    publish_event("mlops/retrain/promoted", ...)
else:
    publish_event("mlops/retrain/rejected", reason="no improvement")
```

---

## 8. MLOps Continuous Training Loop

### 8.1 Skema Loop Lengkap

```text
[ APScheduler tick — 30 menit ]
   ↓
[ RetrainingJob.handle() ]
   ├─ 1. Fetch sliding window 7 hari dari MySQL
   ├─ 2. Filter "normal-ish" (anomaly_score < threshold)
   ├─ 3. Feature engineering (window builder + normalizer)
   ├─ 4. Train challenger (PCA atau LSTM-AE, alternating)
   ├─ 5. Log eksperimen ke MLflow (params, metrics, artifacts)
   ├─ 6. Evaluate vs hold-out validation set → F1
   ├─ 7. Champion-vs-Challenger decision
   │      promote: client.set_registered_model_alias(...)
   │      reject: log alasan
   ├─ 8. Update Redis active_model_uri
   └─ 9. Publish event ke mlops/retrain/result
```

### 8.2 Model Registry Pattern (MLflow Aliases)

Implementasi mengikuti pola MLflow Model Registry resmi:

```python
# Promotion
client.set_registered_model_alias("PumpAD", "champion", new_version)

# Load di InferenceJob
model = mlflow.sklearn.load_model("models:/PumpAD@champion")
```

Reference: MLflow Model Registry docs (commit `1aad416a`), aliases section.

### 8.3 Drift Monitoring (Evidently)

```python
# Setiap N batch baru
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=training_ref, current_data=last_24h)
metrics = report.as_dict()
drift_share = metrics["metrics"][0]["result"]["dataset_drift"]

# Log ke MLflow + dashboard
mlflow.log_metric("drift_share", drift_share)
mlflow.log_artifact("drift_report.html")
```

Pattern reference: [Evidently mlflow_integration.ipynb](https://github.com/evidentlyai/evidently/blob/ad71e132d59ac3a84fce6cf27bd50b12b10d9137/examples/integrations/mlflow_logging/mlflow_integration.ipynb).

### 8.4 Hot-Swap Mechanism

- InferenceJob caches loaded model in-memory dengan TTL 5 menit
- Pada promotion, RetrainingJob publish ke Redis channel `model:invalidate`
- InferenceJob subscribe channel ini → invalidate cache pada next call
- Tidak perlu restart RouteMQ worker

### 8.5 Rollback Strategy

- Keep 3 model version terakhir di MLflow Registry
- Manual rollback button di Streamlit dashboard → memanggil `set_registered_model_alias` ke versi sebelumnya
- Auto-rollback trigger: anomaly rate > 5x baseline selama 10 menit berturut-turut → revert ke previous champion

---

## 9. Dataset & Study Case

### 9.1 SKAB Surrogate Dataset

- **Sumber**: SKAB v0.9 (Skoltech Anomaly Benchmark), GPL-3.0, DOI `10.34740/KAGGLE/DSV/1693952`
- **Repo**: https://github.com/waico/SKAB
- **Karakteristik**: 35 CSV files, ~1 Hz sampling, water circulation testbed nyata
- **8 channel sensor**:
  - `Accelerometer1RMS`, `Accelerometer2RMS` (vibration)
  - `Current` (motor current signature)
  - `Voltage` (supply quality)
  - `Pressure` (hydraulic load)
  - `Temperature`, `Thermocouple` (thermal)
  - `Volume Flow RateRMS` (process flow)
- **Skenario fault**: valve closing, pipe leak, liquid addition, rotor imbalance, water level change, cavitation
- **Label**: `anomaly` + `changepoint` columns per row

### 9.2 PDAM Framing untuk Proposal

> "Kami menggunakan dataset publik **SKAB (Skoltech Anomaly Benchmark)** sebagai surrogate data dari sistem pompa air bersih. Dataset ini berasal dari water circulation testbed di Skolkovo Institute of Science and Technology, yang merepresentasikan karakteristik tipikal sistem pompa sentrifugal pada Instalasi Pengolahan Air (IPA) PDAM — sensor getaran, tekanan, arus motor, suhu, dan flow rate. Skenario fault dalam dataset (valve fault, pipe leak, rotor imbalance, cavitation) merepresentasikan mode kegagalan paling umum pada operasi pompa air bersih."

**Validasi akademis**: [Pau et al. 2021, IEEE ICCE-Berlin] secara eksplisit menggunakan SKAB untuk anomaly detection di water distribution systems (WDS). Framing ini bukan rekayasa — sudah dilakukan peer-reviewed researcher.

### 9.3 Pemetaan Skenario ke Operasi PDAM

| SKAB Scenario | Cerita Operasional PDAM | Expected Model Behavior |
|---|---|---|
| Normal segment | Operasi normal pompa distribusi | Score rendah, semua hijau |
| Valve closing | Valve distribusi macet | Score naik, alert |
| Pipe leak | Kebocoran pipa distribusi | Score naik, alert |
| Rotor imbalance | Bearing pompa aus | Score naik perlahan, alert kritis |
| Cavitation | Tekanan suction tidak mencukupi | Score naik tajam, alert paling kritis |
| Synthetic drift inject | Sensor kalibrasi drift (aging) | v1 mulai false positive → retrain → v2 promoted |

---

## 10. Repo Structure

```text
pdam-pump-sentinel/
├── README.md
├── docs/
│   ├── proposal.md                    # 8 bagian sesuai rubrik
│   ├── architecture.md
│   ├── demo-script.md
│   ├── adr/
│   │   ├── 0001-routemq-as-mqtt-framework.md
│   │   ├── 0002-pca-lstm-ae-algorithm-choice.md
│   │   ├── 0003-mlflow-aliases-champion-challenger.md
│   │   └── 0004-scheduled-vs-trigger-retraining.md
│   └── presentation/
│       └── slides.pdf
│
├── app/
│   ├── routers/
│   │   ├── pump_sensors.py
│   │   └── mlops_events.py
│   ├── controllers/
│   │   ├── sensor_controller.py
│   │   └── mlops_event_controller.py
│   ├── middleware/
│   │   ├── rate_limit.py
│   │   ├── validate_payload.py
│   │   └── correlation.py
│   ├── models/
│   │   ├── sensor_reading.py
│   │   ├── anomaly_event.py
│   │   └── retraining_log.py
│   └── jobs/
│       ├── anomaly_detection_job.py
│       ├── retraining_job.py
│       └── drift_report_job.py
│
├── ml/
│   ├── training/
│   │   ├── train_pca.py
│   │   ├── train_lstm_ae.py
│   │   └── evaluate.py
│   ├── features/
│   │   ├── window_builder.py
│   │   └── normalizer.py
│   ├── monitoring/
│   │   ├── drift_check.py
│   │   └── champion_challenger.py
│   ├── registry/
│   │   └── mlflow_client.py
│   └── datasets/
│       ├── skab_loader.py
│       └── synthetic_drift.py
│
├── bootstrap/
│   └── app.py
│
├── dashboard/
│   ├── app.py
│   └── pages/
│       ├── 1_live_sensors.py
│       ├── 2_anomaly_history.py
│       ├── 3_model_registry.py
│       └── 4_drift_reports.py
│
├── scripts/
│   ├── replay_skab.py
│   ├── inject_drift.py
│   ├── trigger_retrain.py
│   └── seed_initial_models.py
│
├── infra/
│   ├── docker-compose.yml
│   ├── docker-compose.dev.yml
│   ├── mosquitto/mosquitto.conf
│   ├── prometheus/prometheus.yml
│   ├── grafana/dashboards/
│   │   ├── system.json
│   │   └── ml-metrics.json
│   └── mlflow/Dockerfile
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── data/                              # SKAB CSV (gitignored)
├── mlruns/                            # MLflow artifacts (gitignored)
├── logs/                              # gitignored
│
├── .env.example
├── .github/workflows/ci.yml
├── pyproject.toml
└── Makefile
```

---

## 11. Roadmap 5 Minggu + 1 Buffer

| Minggu | Tema | Deliverable | Demo Akhir Minggu |
|:---:|---|---|---|
| **1** | Foundation & Infra | Repo init, Docker Compose stack lengkap (mosquitto + mysql + redis + mlflow + prometheus + grafana), SKAB download script, RouteMQ scaffold, proposal v1 | `docker compose ps` semua services healthy |
| **2** | Ingestion Pipeline | SKAB replayer publishing 8 channel sensor, sensor router/controller, MySQL schema, Redis cache, middleware suite, Streamlit dashboard v0 (live charts) | Live sensor data flowing end-to-end |
| **3** | ML Baseline | Window features, train PCA T²/Q, train LSTM-AE, evaluate vs labeled anomalies, MLflow tracking integrated, first model versions registered | MLflow UI menampilkan experiments + registered models |
| **4** | MLOps Loop | AnomalyDetectionJob pakai registered model + hot-swap via Redis, Evidently drift report, RetrainingJob scheduled (APScheduler), champion-challenger evaluation, synthetic drift injector | Manual trigger retrain → model v2 promoted ke `@champion` |
| **5** | Observability + Polish | Prometheus metrics dari RouteMQ hooks, Grafana dashboards (system + ML), demo script automation, slide deck, ADR docs, proposal final, README polish | Full dry-run presentation timed 13–14 menit |
| **6** | Buffer | Bug fixing, dokumentasi final, latihan presentasi multiple kali, rekam video demo backup | Presentasi final |

---

## 12. Pembagian Tugas 4 Anggota

| Anggota | Role | Tanggung Jawab Konkret |
|---|---|---|
| **A** | DevOps + RouteMQ Backend | Docker Compose, RouteMQ scaffold, routes, controllers, middleware suite, CI pipeline, observability wiring, Grafana dashboard |
| **B** | ML + MLOps Engineer | SKAB preprocessing, feature engineering, train PCA + LSTM-AE, MLflow integration + registry, Evidently drift, champion-challenger evaluation logic |
| **C** | Backend Integrator | Queue jobs (anomaly, retraining, drift), MySQL schema, Redis cache + hot-swap, SKAB replayer, drift injector, scheduler |
| **D** | Frontend + Presenter | Streamlit dashboard 4 halaman, slide deck, demo script, dokumentasi proposal, presentasi utama |

Untuk tim 3 orang: gabungkan **A + C** menjadi satu role "Backend Engineer", pertahankan B dan D.

---

## 13. Demo Storyboard 15 Menit

| Menit | Bagian | Konten |
|:---:|---|---|
| 0–1 | Hook industri | Downtime cost USD 2.3M/jam + ML model degrade tanpa MLOps → MLOps is non-negotiable |
| 1–2 | Problem framing | Threshold gagal + ML model tanpa MLOps juga gagal (concept drift) |
| 2–5 | Highlight 1 (40%) — DevOps + RouteMQ | DSL routing, middleware pipeline, queue workers, Docker Compose multi-service, CI/CD, observability hooks |
| 5–9 | Highlight 2a (30%) — Anomaly Detection | SKAB pump loop dataset, PCA T²/Q champion + LSTM-AE challenger, hasil training |
| 9–12 | Highlight 2b (30%) — MLOps Loop | MLflow registry workflow, scheduled retraining, Evidently drift, champion-challenger promotion |
| 12–14 | **Live Demo** | Replay SKAB → dashboard live → inject drift → trigger retrain → model v2 promoted → dashboard recover |
| 14–15 | Q&A | |

### 13.1 Skenario Live Demo MLOps Loop

```text
T+0   : Model v1 (PCA) aktif. Dashboard hijau.
T+1   : Replay SKAB normal segment → anomaly score rendah.
T+2   : Replay SKAB valve closing → anomaly score naik, alert publish ke MQTT.
T+3   : Inject synthetic drift (pressure mean +15%) untuk simulasi sensor aging.
T+4   : False positive rate naik. Dashboard menampilkan drift warning.
T+5   : "Normally scheduled retrain setiap 30 menit. Untuk demo, kita trigger manual sekarang."
T+6   : MLflow UI menampilkan eksperimen baru. F1_challenger > F1_champion + 0.02.
T+7   : Model v2 promoted via set_registered_model_alias. Redis active_model_uri update.
T+8   : Worker hot-swap. Anomaly score recover, dashboard hijau lagi.
```

---

## 14. Pemetaan ke Kriteria Penilaian Tugas

| Aspek | Bobot | Dipenuhi Oleh |
|---|---:|---|
| Ide & inovasi | 20% | Framework custom + data-driven anomaly + full MLOps lifecycle |
| Implementasi teknis | 30% | RouteMQ pipeline penuh + queue worker + Redis/MySQL + 2 ML model + MLflow + Evidently + Docker + CI |
| Fungsi & demo sistem | 25% | Live SKAB replay → dashboard live → MLOps loop dramatis |
| Presentasi | 15% | Dual-highlight 40/60 narrative, hook industri kuat, demo storyboard 8-step |
| Dokumentasi & laporan | 10% | Proposal 8 bagian, ADR docs, GitBook docs RouteMQ, README polished |

---

## 15. Risiko & Mitigasi

| Risiko | Probability | Mitigasi |
|---|---|---|
| Docker Compose multi-service tidak align | Medium | Healthcheck wajib + startup ordering + Week 1 fokus penuh ke infra |
| MLflow setup rumit di Docker | Medium | Fallback SQLite-backed MLflow yang lebih simple |
| LSTM training lambat di CPU | Medium | Batch size kecil, epochs limited, model arsitektur ringkas (<50k params) |
| Demo race condition (drift + retrain timing) | High | Tombol manual trigger di dashboard untuk demo controlled |
| Anggota belum familiar RouteMQ codebase | High | Week 1 pair-programming reading session wajib |
| MLOps loop "tidak benar-benar otomatis" | Medium | Scheduled retrain + manual trigger button — keduanya tersedia, mana yang berjalan = real automation |
| Dashboard styling pucat | Low | Streamlit theming + Plotly charts |

---

## 16. Pengembangan Lanjutan

Untuk slide "Kendala & Pengembangan":

1. **Real device deployment** — ESP32 dengan sensor vibration + pressure, MQTT-SN bridge ke Mosquitto
2. **Kafka scaling pattern** — production-grade IIoT examples seperti [Kapoor 2025] dan HiveMQ + Kafka pair MQTT dengan Kafka untuk scale; kami implementasi sebagai future work
3. **Multi-pompa multi-tenant** — extend ke beberapa station IPA sekaligus dengan namespacing
4. **Conv-AE / Transformer model** — Conv-AE F1 0.78 di SKAB, lebih tinggi dari LSTM-AE; tambahkan sebagai third challenger
5. **Federated learning** — train lintas IPA tanpa share raw data
6. **Edge inference** — quantize LSTM-AE ke TensorFlow Lite, deploy di edge gateway

---

## 17. Referensi

### Algoritma Anomaly Detection
- Bakdi et al. 2018 — *An improved plant-wide fault detection scheme based on PCA and adaptive threshold for reliable process monitoring*, Journal of Chemometrics. https://consensus.app/papers/details/25e7d9d1d1d358c68af8f960f2535ba1/
- Kazemi et al. 2024 — *Fault Detection and Isolation for Time-Varying Processes Using Neural-Based Principal Component Analysis*, Processes. https://consensus.app/papers/details/a6f3c1b15f5f516bb0f7fe7d76d037d8/
- Githinji et al. 2023 — *Anomaly Detection on Time Series Sensor Data Using Deep LSTM-Autoencoder*, IEEE AFRICON. https://consensus.app/papers/details/b118e85c341e55778ed624ded460411c/
- Maleki et al. 2021 — *Unsupervised anomaly detection with LSTM autoencoders using statistical data-filtering*, Applied Soft Computing. https://consensus.app/papers/details/94e182cfe37d54c2a9d27e184c38bf9b/

### SKAB Dataset & Water Domain
- SKAB Official Repository — https://github.com/waico/SKAB
- Pau et al. 2021 — *Online learning on tiny micro-controllers for anomaly detection in water distribution systems*, IEEE ICCE-Berlin. https://consensus.app/papers/details/d3d6336724485c039e56bdce6abf5c4b/
- Katser & Kozitsin — *Skoltech Anomaly Benchmark (SKAB)*, Kaggle, DOI 10.34740/KAGGLE/DSV/1693952

### MLOps & Continuous Training
- Myakala et al. 2025 — *AutoDrift: A Forecast-Aware Concept Drift Detection and Retraining Pipeline in MLOps with CMAPSS*, IEEE BigDataService. https://consensus.app/papers/details/c9896033e91c5172b879feea738b70f5/
- Dr. Abhay 2025 — *Automated Drift Detection and Retraining Pipeline for ML Models*, IJSREM. https://consensus.app/papers/details/7033b5ce320f5f03a12da5754b9e5764/
- Kreuzberger et al. 2022 — *Machine Learning Operations (MLOps): Overview, Definition, and Architecture*, IEEE Access. https://consensus.app/papers/details/51bd196a657551beb31942ecca8d2c0f/
- Bayram et al. 2024 — *Towards Trustworthy Machine Learning in Production: An Overview of the Robustness in MLOps Approach*, ACM Computing Surveys. https://consensus.app/papers/details/9dbd10d002855c2dbbbbc41758bd9291/

### MQTT + IIoT Architecture
- Kapoor et al. 2025 — *Analyzing the impact of edge, fog and cloud computing on predictive maintenance in the Industrial Internet of Things*, Discover Computing. https://consensus.app/papers/details/68bedcf423e751848d68782f714b9ffb/
- Sathupadi et al. 2024 — *Edge-Cloud Synergy for AI-Enhanced Sensor Network Data: A Real-Time Predictive Maintenance Framework*, Sensors. https://consensus.app/papers/details/489c937a2341547c81e0a39d30fac5e5/
- Kolar et al. 2022 — *Condition Monitoring of Rotary Machinery Using Industrial IOT Framework*, Tehnički glasnik. https://consensus.app/papers/details/17f52e6018c75b6abb8f5ea8edc094e1/

### Tools Documentation
- MLflow Model Registry — https://github.com/mlflow/mlflow/blob/1aad416a6b24c24d34845f2767d37fa22c955105/docs/docs/classic-ml/model-registry/index.mdx
- Evidently MLflow Integration — https://github.com/evidentlyai/evidently/blob/ad71e132d59ac3a84fce6cf27bd50b12b10d9137/examples/integrations/mlflow_logging/mlflow_integration.ipynb
- RouteMQ Framework — https://github.com/ardzz/RouteMQ
- SKAB Algorithms Reference Notebooks — https://github.com/waico/SKAB/blob/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/notebooks/README.md

### Industry Context
- Siemens *True Cost of Downtime 2024* — https://assets.new.siemens.com/siemens/assets/api/uuid:1b43afb5-2d07-47f7-9eb7-893fe7d0bc59/TCOD-2024_original.pdf

---

## 18. Status & Next Action

- [x] Brainstorming complete
- [x] Evidence verification via Consensus MCP + librarian cross-check
- [x] Algorithm selection finalized (PCA T²/Q champion + LSTM-AE challenger)
- [x] Design document written
- [ ] Approval untuk mulai implementasi Week 1
- [ ] Repo `pdam-pump-sentinel` di-init
- [ ] Docker Compose stack di-bring up

**Ready to set up for implementation?**
