# PDAM Pump Sentinel

Platform MLOps untuk **Predictive Maintenance Pompa Distribusi Air** berbasis MQTT Framework RouteMQ dengan Continuous Training.

> Tugas Besar Project AI/IoT — kombinasi **DevOps (40%)** + **AI/ML & MLOps (60%)**.

## Apa Ini?

Sistem pemantauan kondisi pompa pada Instalasi Pengolahan Air (IPA) yang:

- Menerima data sensor pompa via **MQTT** (RouteMQ application framework)
- Mendeteksi anomali multivariate dengan **PCA T²/Q** (champion) + **LSTM Autoencoder** (challenger)
- Menjalankan **MLOps continuous training loop**: MLflow registry, Evidently drift monitoring, scheduled retraining, champion-challenger promotion
- Menampilkan dashboard operator real-time (Streamlit)

Dataset surrogate: **SKAB (Skoltech Anomaly Benchmark)** — water circulation testbed industri nyata.

## Arsitektur 3-Pilar

```text
DevOps (Docker Compose, CI, observability)
        ↓
RouteMQ Application (Router → Middleware → Controller → Queue → Worker)
        ↕
MLOps (Training → MLflow Registry → Inference → Evidently Drift → Retraining)
```

Detail lengkap: lihat [`docs/plans/design.md`](docs/plans/design.md).

## Stack

| Layer | Tools |
|---|---|
| MQTT broker | Eclipse Mosquitto |
| App framework | RouteMQ (Python 3.12+) |
| Database | MySQL 8 + Redis 7 |
| ML | scikit-learn (PCA) + TensorFlow/Keras (LSTM-AE) |
| MLOps | MLflow + Evidently AI |
| Dashboard | Streamlit |
| Observability | Prometheus + Grafana |
| Orchestration | Docker Compose |

## Quick Start

```bash
# Install dependencies
uv sync

# Bring up infra (Redis + MySQL + MQTT + MLflow)
make dev

# Run RouteMQ app
make run

# Replay SKAB dataset to MQTT
make replay

# Launch dashboard
make dashboard
```

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
