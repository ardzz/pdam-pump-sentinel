# AI Methodology: PDAM Pump Sentinel

## Objective

Project memakai SKAB sebagai surrogate sistem pompa distribusi air PDAM. Model belajar pola normal multivariate dari vibration, current, pressure, temperature, voltage, dan flow, bukan memakai batas min/max per sensor.

Keputusan model:

| Role | Model | Alasan |
|---|---|---|
| Champion | PCA T²/Q | Cepat, interpretatif, sesuai process monitoring, dan tercatat di official SKAB proposed leaderboard. |
| Challenger | LSTM Autoencoder | Menangkap pola temporal nonlinear; dipakai jika PCA memberi false alarm tinggi. |
| Negative baseline | Isolation Forest | Pembanding sederhana, bukan kandidat utama. |

## Dataset Evidence

Sumber utama: SKAB official repository dan DOI Kaggle [R1][R2]. SKAB v0.9 berisi 35 file CSV. Setiap file mewakili satu eksperimen pada water-circulation testbed dan berisi satu anomaly event [R1]. Dataset mendukung outlier detection dan changepoint detection [R1].

Kolom aktual CSV:

```text
datetime
Accelerometer1RMS
Accelerometer2RMS
Current
Pressure
Temperature
Thermocouple
Voltage
Volume Flow RateRMS
anomaly
changepoint
```

Catatan schema: README menyebut `RateRMS`, tetapi header CSV aktual memakai `Volume Flow RateRMS` [R3]. Kode harus mengikuti header CSV, bukan ringkasan README.

Pemetaan fault ke framing PDAM:

| Folder | Scenario | Makna PDAM |
|---|---|---|
| `anomaly-free` | normal operation | operasi pompa sehat |
| `valve1` | inlet valve closing | gangguan suplai masuk pompa |
| `valve2` | outlet valve closing | gangguan distribusi keluar pompa |
| `other` | leak/addition, rotor imbalance, cavitation, high-temperature water | kebocoran, imbalance, cavitation, gangguan fluida |

Official proposed leaderboard mencatat `T-squared+Q` dengan F1 0.76. `Conv-AE` dan `MSET` berada di sekitar F1 0.78 [R4]. Angka ini menjadi dasar pemilihan PCA T²/Q sebagai champion awal, tanpa klaim bahwa model ini selalu unggul di semua konteks.

## Data Preparation

Split harus berbasis file eksperimen, bukan random row shuffle.

| Split | Isi | Tujuan |
|---|---|---|
| Train | normal-only windows dari `anomaly-free` dan bagian normal file lain jika segmentasinya aman | belajar pola sehat |
| Validation | normal windows + beberapa anomaly file | tuning threshold dan parameter |
| Test | held-out anomaly files | evaluasi akhir tanpa leakage |

Aturan anti-leakage:

1. Jangan shuffle row lintas file.
2. Jangan fit scaler pada validation/test.
3. Jangan pakai `anomaly` atau `changepoint` sebagai fitur.
4. Jangan memilih threshold dari test set.
5. Simpan daftar file train/validation/test sebagai artifact MLflow.

Scaler awal:

```text
RobustScaler
fit: train normal only
transform: train, validation, test, online inference
```

RobustScaler dipilih karena fault dapat memicu lonjakan pressure, current, vibration, atau flow. Jika PCA sulit ditafsirkan, jalankan eksperimen pembanding dengan StandardScaler.

## Windowing

MQTT replay mengirim row per timestamp. Model inference memakai rolling buffer per station.

Parameter awal:

| Parameter | Nilai awal | Catatan |
|---|---:|---|
| `window_size` | 60 samples | kira-kira 1 menit jika sampling sekitar 1 Hz |
| `stride_train` | 1 | memperbanyak sample normal |
| `stride_eval` | 1 | deteksi secepat mungkin |
| padding | none | drop window tidak lengkap |

SKAB notebook memakai sequence helper untuk overlapping windows [R12]. Untuk demo PDAM, window 60 dipakai karena mewakili dinamika pompa sekitar satu menit dan lebih mudah dijelaskan ke operator daripada window 10.

## Visualization Plan

Visualisasi sebelum training final:

1. Sensor time series dengan anomaly overlay.
2. Correlation heatmap antar sensor.
3. PC1/PC2 scatter untuk normal vs anomaly.
4. Hotelling T² score timeline.
5. Q/SPE residual score timeline.
6. LSTM-AE reconstruction error timeline.
7. Per-sensor reconstruction error ranking.
8. Precision, recall, F1, false alarm rate comparison.
9. Drift chart antara reference window dan current window.

Visualisasi dipakai untuk menjelaskan bahwa model membaca kombinasi sensor, bukan batas statis per sensor.

## Champion: PCA T²/Q

PCA T²/Q memisahkan deviasi ke dua ruang:

| Statistic | Makna |
|---|---|
| Hotelling T² | deviasi di principal-component subspace |
| Q / SPE | residual di luar subspace PCA |

Alur training:

1. Fit scaler pada normal train.
2. Fit PCA pada scaled normal train.
3. Pilih `n_components` agar explained variance sekitar 90%.
4. Hitung T² dan Q untuk train/validation.
5. Tentukan threshold dari validation normal atau formula control limit.
6. Simpan scaler, PCA, threshold, sensor order, dan metadata ke MLflow.

Parameter awal:

```text
n_components = 0.90
t2_threshold = p95(validation_normal_t2)
q_threshold = p95(validation_normal_q)
decision = t2 > t2_threshold OR q > q_threshold
```

SKAB menyediakan implementasi T²/Q dengan control limit dan residual transform [R5]. Gunakan sebagai referensi konsep. Jangan salin kode karena repository SKAB memakai GPL-3.0 [R13]. Reimplementasi dilakukan di project ini dengan scikit-learn dan numpy.

Eksperimen setelah MVP:

```text
EWMA smoothing on T²/Q
KDE threshold for non-Gaussian residual
sensor contribution plot for root-cause hint
```

## Challenger: LSTM Autoencoder

LSTM-AE belajar merekonstruksi normal window. Window anomali diharapkan menghasilkan reconstruction error tinggi.

Arsitektur awal:

| Component | Nilai awal |
|---|---|
| Encoder | LSTM 64 |
| Bottleneck | Dense 32 |
| Decoder | LSTM 64 + RepeatVector |
| Output | TimeDistributed Dense |
| Loss | MAE first, compare MSE later |
| Optimizer | Adam, learning rate 1e-3 |
| Batch size | 32 atau 64 |
| Epochs | max 100 |
| Early stopping | patience 10 |
| Dropout | 0.2 |

Referensi LSTM-AE SKAB memakai Keras LSTM encoder, RepeatVector, decoder LSTM, TimeDistributed Dense, MAE loss, dan early stopping [R6]. Gunakan sebagai pola arsitektur, bukan sumber kode.

Threshold awal:

```text
ae_threshold = p95(validation_reconstruction_error)
```

Jika false alarm terlalu tinggi:

```text
try p99
add EWMA smoothing
use per-sensor weighted reconstruction error
```

## Evaluation Protocol

Point-wise F1 tidak cukup. Operator PDAM perlu tahu apakah event fault terdeteksi, seberapa cepat alert muncul, dan berapa banyak alarm palsu.

Metrik wajib:

| Metric | Tujuan |
|---|---|
| Precision | mengukur alarm palsu |
| Recall | mengukur missed fault |
| F1 | ringkasan precision/recall |
| Event-level F1 | apakah event anomaly terdeteksi |
| False alarm rate | alarm palsu per waktu/station |
| Detection delay | jarak waktu dari anomaly start ke alert pertama |
| Missed anomaly rate | event anomaly yang tidak terdeteksi |

Event-level rule:

```text
An anomaly event is detected if at least one alert appears inside the labeled anomaly interval.
Merge detections separated by <= 10 samples.
```

Evaluasi time-series anomaly bisa bias jika hanya memakai point-wise metric. Garg et al. dan Schmidl et al. membahas risiko ini untuk multivariate time-series anomaly detection [R8][R9].

## Online Inference Design

Runtime flow:

```text
SKAB CSV row
→ scripts/replay_skab.py
→ MQTT topic factory/skab/{station}/telemetry
→ RouteMQ router
→ rolling buffer per station
→ scaler transform
→ champion model inference
→ anomaly score + label
→ MQTT topic factory/skab/{station}/anomaly
→ dashboard
```

State per station:

```text
last 60 scaled sensor rows
last anomaly score
last model version
last alert timestamp
```

Payload anomaly:

```json
{
  "station": "valve1-1",
  "ts": "...",
  "model": "PumpAD@champion",
  "model_version": "...",
  "score_t2": 0.0,
  "score_q": 0.0,
  "threshold_t2": 0.0,
  "threshold_q": 0.0,
  "is_anomaly": false,
  "top_sensors": ["Pressure", "Current"]
}
```

## MLOps Design

Pipeline:

```text
training data snapshot
→ train champion/challenger candidate
→ log run to MLflow
→ validate against fixed validation split
→ compare with champion
→ promote or reject
→ update Redis active model pointer
→ inference reloads model
```

MLflow model naming:

```text
registered_model = PumpAD
alias champion
alias challenger
```

Redis pointer:

```text
model:active_uri = models:/PumpAD@champion
```

Promotion gate:

```text
promote if:
  f1_challenger > f1_champion + 0.02
  false_alarm_rate_challenger <= false_alarm_rate_champion * 1.05
  model_load_test == pass
  inference_latency_p95 <= target
  schema_validation == pass
```

Rollback gate:

```text
rollback if:
  model_load_error
  inference_error_rate increases
  false_alarm_rate > 1.5x champion baseline
  prediction distribution collapses
  operator rejects repeated alerts
```

Monitoring signals:

| Signal | Example |
|---|---|
| data drift | PSI, KS, Wasserstein per sensor |
| score drift | p95/p99 anomaly score shift |
| prediction drift | anomaly ratio per station |
| data quality | missing rate, stale timestamp, constant sensor |
| model quality | F1, precision, recall when labels exist |
| system health | inference latency, model version, retraining status |

Referensi MLOps dipakai untuk membatasi desain pada lifecycle training, model registry, kontrol technical debt, dan kesiapan deployment [R14][R15][R16][R17].

## Implementation Order

Urutan implementasi:

1. `scripts/replay_skab.py`
2. `ml/datasets/skab_loader.py`
3. `ml/features/windowing.py`
4. `ml/training/train_pca.py`
5. `ml/training/evaluate.py`
6. `ml/registry/model_registry.py`
7. `app/routers/pump_sensors.py`
8. `app/jobs/anomaly_detection_job.py`
9. LSTM-AE challenger
10. drift + retraining jobs

## References

| ID | Source |
|---|---|
| R1 | waico/SKAB README, dataset overview and proposed leaderboard: https://github.com/waico/SKAB/blob/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/README.md |
| R2 | Katser, I. D., & Kozitsin, V. O. (2020). Skoltech Anomaly Benchmark (SKAB). Kaggle. https://doi.org/10.34740/KAGGLE/DSV/1693952 |
| R3 | SKAB labeled CSV header: https://github.com/waico/SKAB/blob/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/data/valve1/1.csv#L1 |
| R4 | SKAB proposed outlier leaderboard: https://github.com/waico/SKAB/blob/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/README.md#L56-L74 |
| R5 | SKAB PCA T²/Q implementation: https://github.com/waico/SKAB/blob/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/core/t2.py |
| R6 | SKAB LSTM-AE implementation: https://github.com/waico/SKAB/blob/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/core/LSTM_AE.py |
| R7 | Harrou et al. (2016). Amalgamation of anomaly-detection indices for enhanced process monitoring. https://consensus.app/papers/details/5508b6fee62358b1b4d601ef0bca964d/ |
| R8 | Garg et al. (2021). An Evaluation of Anomaly Detection and Diagnosis in Multivariate Time Series. https://consensus.app/papers/details/e6f62a5e92065cdca4ab474429b8662f/ |
| R9 | Schmidl et al. (2022). Anomaly Detection in Time Series: A Comprehensive Evaluation. https://consensus.app/papers/details/37fd4988ab1d514b96e42fbe1b56f93c/ |
| R10 | Maleki et al. (2021). Unsupervised anomaly detection with LSTM autoencoders using statistical data-filtering. https://consensus.app/papers/details/94e182cfe37d54c2a9d27e184c38bf9b/ |
| R11 | Githinji et al. (2023). Anomaly Detection on Time Series Sensor Data Using Deep LSTM-Autoencoder. https://consensus.app/papers/details/b118e85c341e55778ed624ded460411c/ |
| R12 | SKAB sequence helper: https://github.com/waico/SKAB/blob/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/core/utils.py#L59-L64 |
| R13 | SKAB GPL-3.0 license: https://github.com/waico/SKAB/blob/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/LICENSE |
| R14 | Kreuzberger et al. (2022). Machine Learning Operations (MLOps): Overview, Definition, and Architecture. https://consensus.app/papers/details/51bd196a657551beb31942ecca8d2c0f/ |
| R15 | Testi et al. (2022). MLOps: A Taxonomy and a Methodology. https://consensus.app/papers/details/1a14edbc89895e91a9ce5952a3568ca8/ |
| R16 | Zaharia et al. (2018). Accelerating the Machine Learning Lifecycle with MLflow. https://people.eecs.berkeley.edu/~matei/papers/2018/ieee_mlflow.pdf |
| R17 | Sculley et al. (2015). Hidden Technical Debt in Machine Learning Systems. https://papers.nips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf |
