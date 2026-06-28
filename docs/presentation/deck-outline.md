# PDAM Pump Sentinel — Presentation Deck Outline

First-draft structure for a 13-14 minute presentation. The deck is intentionally honest: SKAB is a surrogate dataset, the observability stack is local Docker Compose evidence, and several production-grade items remain roadmap work.

## Source Materials

- `README.md` — project summary, implemented observability portfolio, commands, known limitations.
- `docs/plans/design.md` — original project scope, architecture, algorithms, demo framing.
- `docs/presentation/labeling-strategy-notes.md` — PCA-spectral champion defense, industry/academic support, speaker script.
- `docs/presentation/screenshot-checklist.md` — verified demo screenshot sequence and canonical evidence images.
- `docs/plans/sprint-remaining.md` — remaining work, evaluation spectrum, final readiness caveats.

## Timing Guide

| Slide | Target time |
|---:|---:|
| 1 | 0:45 |
| 2 | 1:00 |
| 3 | 1:00 |
| 4 | 1:10 |
| 5 | 1:00 |
| 6 | 1:15 |
| 7 | 1:15 |
| 8 | 1:10 |
| 9 | 1:00 |
| 10 | 1:00 |
| 11 | 0:55 |
| 12 | 1:00 |
| 13 | 1:15 |

Total target: roughly 13-14 minutes, leaving a little buffer for demo transitions.

## Slide 1 — PDAM Pump Sentinel

**Key message:** Platform MLOps untuk predictive maintenance pompa distribusi air berbasis MQTT RouteMQ, PCA T²/Q, MLflow, Grafana, dan Streamlit.

**Bullets:**

- End-to-end AIoT slice: ingestion, anomaly detection, model lifecycle, dashboard, observability.
- Demo akademik berbasis SKAB water-circulation benchmark sebagai surrogate data.
- Framing kontribusi: 40% DevOps/RouteMQ, 60% AI/ML + MLOps.

**Speaker cue:** Buka dengan masalah aset pompa kritis dan kebutuhan sistem yang tidak berhenti di notebook ML.

**Sources:** `README.md`, `docs/plans/design.md §1`.

## Slide 2 — Why This Problem Matters

**Key message:** Threshold monitoring saja tidak cukup untuk degradasi pompa yang multivariate, gradual, dan contextual.

**Bullets:**

- Pompa sentrifugal adalah aset kritis distribusi air bersih.
- Kegagalan mendadak berdampak ke pelanggan, biaya dispatch, tekanan pipa, dan reputasi layanan.
- Threshold min/max lemah terhadap contextual anomaly, sensor drift, dan collective anomaly.

**Speaker cue:** Tekankan bahwa masalahnya bukan sekadar membuat alarm, tetapi membaca pola lintas sensor.

**Sources:** `docs/plans/design.md §2.1-2.3`.

## Slide 3 — Scope and Honest Data Framing

**Key message:** SKAB dipakai sebagai surrogate water-circulation testbed, bukan klaim data operasional PDAM nyata.

**Bullets:**

- Dataset: SKAB, public water-circulation benchmark dengan fault injection terkontrol.
- MVP tidak memakai sensor ESP32 asli atau deployment Kubernetes.
- Tujuan presentasi: membuktikan pipeline, metodologi, dan readiness demo lokal.

**Speaker cue:** Ini slide penting untuk membangun trust reviewer sebelum masuk ke hasil.

**Sources:** `README.md`, `docs/plans/design.md §3.2-3.3`, `docs/presentation/labeling-strategy-notes.md §3`.

## Slide 4 — Architecture: Three-Layer Slice

**Key message:** Sistem dibagi menjadi DevOps layer, RouteMQ application layer, dan MLOps layer.

**Bullets:**

- DevOps: Docker Compose, Prometheus, Grafana, health checks.
- RouteMQ: MQTT router, middleware, controllers, Redis/ClickHouse persistence.
- MLOps: training, MLflow registry, Evidently drift, retraining, champion-challenger promotion.

**Speaker cue:** Jelaskan arsitektur sebagai satu vertical slice, bukan kumpulan tools terpisah.

**Sources:** `README.md`, `docs/plans/design.md §5.1-5.5`.

## Slide 5 — What Is Implemented Today

**Key message:** Project sudah punya evidence stack lokal yang cukup untuk demo portfolio, tapi tidak diklaim sebagai production cloud deployment.

**Bullets:**

- `/metrics` RouteMQ + `pumpad_*` metrics.
- Grafana dashboards: RouteMQ observability, MLOps, system health, MQTT broker.
- Streamlit Overview/System Health/Runbook dengan observability snapshot dan triage evidence.
- Demo `make demo` mencakup T+0 sampai T+8, optional T+9 observability evidence.

**Speaker cue:** Pakai kata “local observability evidence” agar klaimnya tepat.

**Sources:** `README.md`, `docs/presentation/screenshot-checklist.md`, `docs/plans/sprint-remaining.md`.

## Slide 6 — Demo Storyboard: T+0 to T+9

**Key message:** Alur demo menunjukkan baseline, anomaly, drift, retraining, promotion, recovery, dan observability evidence.

**Bullets:**

- T+0 baseline: champion model dan dashboard sehat.
- T+1/T+2: normal replay lalu anomalous replay.
- T+3/T+4: drift injection dan detection.
- T+5-T+8: retrain, verify challenger, promote alias, hot-swap inference.
- T+9 optional: observability evidence capture.

**Speaker cue:** Ini peta jalan presentasi live; jangan tenggelam dalam detail command.

**Sources:** `docs/presentation/screenshot-checklist.md §Per-phase capture sequence`, `scripts/run_e2e_demo.py` referenced by README.

## Slide 7 — Why PCA-Spectral as Day-1 Champion

**Key message:** PCA-spectral dipilih bukan karena “paling akurat”, melainkan karena deployable dari Day-1 tanpa labeled fault inventory.

**Bullets:**

- Normal-baseline-first sesuai pola industrial predictive maintenance.
- T² membaca jarak sepanjang pola normal; Q/SPE membaca residual yang tidak bisa dijelaskan PCA.
- PCA cepat, decomposable, dan cocok sebagai champion awal.
- Supervised model tetap disiapkan sebagai challenger ketika label cukup matang.

**Speaker cue:** Gunakan narasi: “0.91 butuh fault labels di train; PDAM Day-1 belum punya itu.”

**Sources:** `docs/presentation/labeling-strategy-notes.md §Core argument`, `docs/plans/sprint-remaining.md §Honest evaluation spectrum`.

## Slide 8 — Honest Evaluation Spectrum

**Key message:** Angka evaluasi harus dipresentasikan sesuai split, bukan dicampur sebagai satu klaim performa.

**Bullets:**

- PCA-spectral normal-only: deployable untuk novel-fault setting, F1 sekitar 0.58.
- Supervised cross-group: labeled anomalies membantu, tetapi tetap novel-fault constrained.
- In-distribution supervised: XGB/LGBM sekitar 0.90, tetapi perlu fault type yang sudah muncul di train.
- Random-window dengan leakage tidak boleh dijual sebagai generalization claim.

**Speaker cue:** Ini melindungi deck dari overclaim. Tekankan “jujur tapi defensible”.

**Sources:** `docs/plans/sprint-remaining.md §Honest evaluation spectrum`, `docs/presentation/labeling-strategy-notes.md §6`.

## Slide 9 — MLOps Loop Evidence

**Key message:** ML lifecycle bukan konsep saja; ada registry, champion alias, drift/retrain evidence, dan operator-action observability.

**Bullets:**

- MLflow model registry memakai alias `@champion` / challenger promotion path.
- Champion-challenger gate menjaga F1 margin dan FAR guard.
- Grafana MLOps dashboard menampilkan loop dan operator action evidence.

**Screenshot:** `docs/presentation/screenshots/t9-observability-grafana-mlops-observability-20260609T130723Z.png`

**Sources:** `README.md`, `docs/presentation/labeling-strategy-notes.md §4`, `docs/presentation/screenshot-checklist.md §Verified T+9 capture`.

## Slide 10 — Pipeline Observability Evidence

**Key message:** Pipeline ingestion → inference → persistence dapat diamati dari dashboard RouteMQ/Grafana.

**Bullets:**

- Metrics mencakup dispatch, inference latency, anomaly score, persistence writes, freshness.
- Observability bersifat lokal Docker Compose, bukan production SRE claim.
- Tujuan slide: membuktikan sistem demo bisa dioperasikan dan diaudit.

**Screenshot:** `docs/presentation/screenshots/t9-observability-grafana-pipeline-observability-20260609T130723Z.png`

**Sources:** `README.md §Bukti Observability Portfolio`, `docs/presentation/screenshot-checklist.md`.

## Slide 11 — System Health and SLO Evidence

**Key message:** Sistem menyediakan health signal untuk dependency dan active model freshness.

**Bullets:**

- Grafana system-health dashboard membaca scrape health dan freshness.
- Alert rules lokal mencakup telemetry freshness, inference errors, persistence errors, active model age.
- Ini memberi dasar runbook, bukan klaim PagerDuty production routing.

**Screenshot:** `docs/presentation/screenshots/t9-observability-grafana-slo-health-20260609T130723Z.png`

**Sources:** `README.md §Bukti Observability Portfolio`, `docs/plans/sprint-remaining.md §Architectural follow-ups`.

## Slide 12 — Operator Console and Runbook

**Key message:** Streamlit bukan hanya chart; operator mendapat observability snapshot dan runbook triage.

**Bullets:**

- Overview menampilkan status, model aktif, dan freshness.
- Runbook mengarahkan triage berbasis metrik.
- Operator action evidence tersimpan di ClickHouse dan dipantau di Grafana.

**Screenshots:**

- `docs/presentation/screenshots/t9-observability-streamlit-observability-snapshot-20260628T0535Z.png`
- `docs/presentation/screenshots/t9-observability-streamlit-runbook-observability-20260628T0535Z.png`

**Sources:** `README.md`, `docs/presentation/screenshot-checklist.md`.

## Slide 13 — Limitations, Roadmap, Close

**Key message:** MVP sudah demo-ready, tetapi roadmap production masih jelas dan tidak ditutup-tutupi.

**Bullets:**

- Known limitations: synchronous inference, scheduler behind env flags, scheduled retraining PCA-only, process-local hot-swap.
- Future work: operator triage labels, supervised promotion gates, supervised alias setter, incident routing.
- Closing thesis: platform ini menunjukkan kemampuan end-to-end MLOps, bukan sekadar model notebook.

**Speaker cue:** Tutup dengan “honest engineering”: sudah jalan, tahu batasnya, dan tahu next step spesifik.

**Sources:** `README.md §Known Limitations`, `docs/plans/sprint-remaining.md §Architectural follow-ups`.

## Canonical Screenshots Used

- `docs/presentation/screenshots/t9-observability-grafana-mlops-observability-20260609T130723Z.png`
- `docs/presentation/screenshots/t9-observability-grafana-pipeline-observability-20260609T130723Z.png`
- `docs/presentation/screenshots/t9-observability-grafana-slo-health-20260609T130723Z.png`
- `docs/presentation/screenshots/t9-observability-streamlit-observability-snapshot-20260628T0535Z.png`
- `docs/presentation/screenshots/t9-observability-streamlit-runbook-observability-20260628T0535Z.png`
