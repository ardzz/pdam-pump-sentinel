# PDAM Pump Sentinel

Platform MLOps untuk **Predictive Maintenance Pompa Distribusi Air** berbasis MQTT Framework RouteMQ dengan Continuous Training.

> Tugas Besar Project AI/IoT, kombinasi **DevOps (40%)** + **AI/ML & MLOps (60%)**.

## Apa Ini?

Sistem pemantauan kondisi pompa pada Instalasi Pengolahan Air (IPA) yang:

- Menerima data sensor pompa via **MQTT** (RouteMQ application framework)
- Mendeteksi anomali multivariate dengan **PCA T²/Q** (champion) + **LSTM Autoencoder** (challenger)
- Menjalankan **MLOps continuous training loop**: MLflow registry, Evidently drift monitoring, scheduled retraining, champion-challenger promotion
- Menampilkan dashboard operator real-time (Streamlit)
- Mengekspos observability lokal: Prometheus metrics, Grafana SLO-style panels, Streamlit observability snapshot, drift/retrain evidence, dan runbook operator

Dataset surrogate: **SKAB (Skoltech Anomaly Benchmark)**, public water circulation testbed. SKAB **bukan data operasional PDAM nyata** dan dipakai untuk demo akademik, verifikasi pipeline, serta pengembangan metodologi.

## Arsitektur 3-Pilar

```text
DevOps (Docker Compose, CI, observability)
        ↓
RouteMQ Application (Router → Middleware → Controller sinkron + Queue/Worker MLOps)
        ↕
MLOps (Training → MLflow Registry → Inference → Evidently Drift → Retraining)
```

Detail lengkap: lihat [`docs/plans/design.md`](docs/plans/design.md).

## Stack

| Layer | Tools |
|---|---|
| MQTT broker | Eclipse Mosquitto |
| App framework | RouteMQ (Python 3.12+) |
| Telemetry store | ClickHouse 24 |
| Cache + queue | Redis 7 |
| ML | scikit-learn (PCA) + TensorFlow/Keras (LSTM-AE) |
| MLOps | MLflow + Evidently AI |
| Dashboard | Streamlit |
| Observability | Prometheus + Grafana + Streamlit operator runbook |
| Orchestration | Docker Compose |

## Bukti Observability Portfolio

Observability yang sudah terimplementasi berada pada stack lokal Docker Compose, bukan klaim deployment produksi cloud/Kubernetes. Bukti teknisnya meliputi:

- Endpoint `/metrics` aplikasi RouteMQ yang menggabungkan metrik framework dan `pumpad_*`.
- Local Prometheus alert rules untuk app scrape health, telemetry freshness, inference errors, persistence write errors, drift report age, dan active model age.
- Metrik bounded-label untuk inference, persistence writes, anomaly severity, telemetry freshness, drift report age, retrain duration, active model age, dan observability schema/build marker.
- Dashboard Grafana `pumpad-observability`, `pumpad-mlops`, `pumpad-system-health`, dan `pumpad-mqtt-broker` untuk membaca pipeline, MLOps SLIs, dependency health, dan broker health.
- Streamlit Overview/System Health/Runbook yang menampilkan `Observability Snapshot`, app metric freshness, service checks, dan triage berbasis metrik.
- Demo T+0–T+8, optional T+9 observability evidence check, dan target screenshot observability pada [`docs/presentation/screenshot-checklist.md`](docs/presentation/screenshot-checklist.md).

Spec upgrade lengkap: [`docs/plans/2026-06-08-portfolio-observability-upgrades.md`](docs/plans/2026-06-08-portfolio-observability-upgrades.md).

## Quick Start

```bash
# Install dependencies
uv sync

# Bring up infra (Redis + ClickHouse + MQTT + MLflow)
make dev

# Run RouteMQ app
make run

# Replay SKAB dataset to MQTT
make replay

# Launch dashboard
make dashboard
```

Perintah tambahan:

| Perintah | Fungsi |
|---|---|
| `make mlflow-report` | Server report MLflow di port 5050 dari `data/mlflow_live.db`, berisi eksperimen perbandingan lintas-famili. |
| `make demo` | Orkestrasi demo E2E T+0–T+8; tambahkan `DEMO_EXTRA_ARGS="--observability-evidence"` untuk optional T+9 evidence check. |
| `make screenshots TAG=<tag>` | Tangkapan layar dashboard via headless Chrome (`TAG` adalah alias untuk `SCREENSHOT_TAG`). |

### Topik MQTT

- Telemetry: `factory/skab/{station}/telemetry`
- Anomaly: `factory/skab/{station}/anomaly`

Demo vertical slice awal: lihat [`docs/demo-vertical-slice.md`](docs/demo-vertical-slice.md).
Notebook EDA dan persiapan data SKAB: [`notebooks/skab_eda_and_data_prep.ipynb`](notebooks/skab_eda_and_data_prep.ipynb). Notebook ini mengecek missing policy, timestamp quality, korelasi dan heatmap `sns`, distribusi, rolling stats, label ranges/overlay, split manifest metadata/base_dir, held-out test metrics, changepoint separation, dan provenance pada fixture kecil tanpa klaim benchmark.

## Keterbatasan / Known Limitations

- Inferensi anomali berjalan sinkron di `app/controllers/anomaly_controller.py`, bukan pola Queue -> Worker. Infrastruktur queue/job dipakai hanya untuk job MLOps, yaitu drift report dan retraining.
- Scheduler retraining dan drift ada di balik env flag `ENABLE_DRIFT_SCHEDULER` dan `DRIFT_INTERVAL_MINUTES`, serta tidak aktif secara default pada compose.
- Scheduled retraining saat ini PCA-only dan berbasis path SKAB, bukan rolling-window dari ClickHouse, belum melatih LSTM-AE atau supervised secara otomatis.
- Hot-swap model bersifat process-local. Belum ada runtime polling alias MLflow, hanya cold-start load champion.
- Live PCA inference melayani fitur raw saja. Mode spectral/enriched masih untuk offline-eval.
- Sebagian halaman dashboard refresh manual: `live_sensors` 5 detik, `system_health` 10 detik, halaman lain memakai cache TTL.

## Struktur Project

```text
pdam-pump-sentinel/
├── app/            # RouteMQ application (routers, controllers, middleware, jobs, models)
├── ml/             # ML/MLOps domain (training, features, monitoring, registry, datasets)
├── bootstrap/      # RouteMQ bootstrap entry
├── dashboard/      # Streamlit dashboard
├── scripts/        # SKAB replayer, drift injector, retrain trigger
├── infra/          # Docker Compose + Mosquitto/Prometheus/Grafana/MLflow config
├── tests/          # Unit + integration tests
└── docs/           # Design, proposal, ADR, presentation
```

## Tim

| Role | Tanggung Jawab |
|---|---|
| DevOps + RouteMQ Backend | Docker, RouteMQ scaffold, CI, observability |
| ML + MLOps Engineer | Training, MLflow, Evidently, champion-challenger |
| Backend Integrator | Queue jobs, schema, Redis hot-swap, replayer |
| Frontend + Presenter | Dashboard, slides, demo, dokumentasi |

## Lisensi

Academic project. Dataset SKAB: GPL-3.0 (https://github.com/waico/SKAB).
