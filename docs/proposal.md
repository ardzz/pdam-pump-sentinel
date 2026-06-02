# Proposal Proyek v0

# PDAM Pump Sentinel: Platform MLOps untuk Predictive Maintenance Pompa Distribusi Air berbasis Framework MQTT RouteMQ

## 1. Judul dan identitas tim

**Judul proyek:** PDAM Pump Sentinel: Platform MLOps untuk Predictive Maintenance Pompa Distribusi Air berbasis Framework MQTT RouteMQ
**Jenis kegiatan:** Tugas besar proyek AI/IoT
**Nama tim:** diisi kemudian
**Kelas:** diisi kemudian
**Dosen pengampu:** diisi kemudian

| Anggota | Peran utama | Tanggung jawab awal |
|---|---|---|
| Anggota 1 | RouteMQ, DevOps, dan Backend Integrator | Docker Compose, Mosquitto, scaffold RouteMQ, routing, middleware, queue job, Redis/ClickHouse, model hot-swap, SKAB replayer, scheduler, observability |
| Anggota 2 | ML, MLOps, Dashboard, dan Dokumentasi | SKAB preprocessing, PCA T²/SPE Q, LSTM Autoencoder, MLflow, Evidently, evaluasi model, Streamlit dashboard, skenario demo, slide, proposal, laporan akhir |

Integrasi akhir, pengujian demo, dan presentasi dilakukan bersama agar kedua sisi proyek tetap tersambung.

## 2. Ringkasan proyek

PDAM Pump Sentinel adalah rancangan sistem AIoT untuk mendeteksi anomali pada sistem pompa distribusi air. Proyek ini tidak memakai data operasional PDAM asli. Dataset publik SKAB atau Skoltech Anomaly Benchmark dipakai sebagai surrogate karena berasal dari water circulation testbed industri dengan sensor getaran, arus, tekanan, suhu, tegangan, dan flow.

Bobot pekerjaan dibagi 40/60 antara RouteMQ/DevOps dan AI/ML/MLOps. Bagian 40 persen berfokus pada RouteMQ sebagai framework aplikasi MQTT di Python, bukan sekadar konfigurasi broker. RouteMQ dirancang memiliki router, middleware, controller, queue, worker, integrasi Redis/ClickHouse, dan observability hook. Bagian 60 persen berada pada anomaly detection dan MLOps, yaitu training model, registry MLflow, monitoring drift, retraining terjadwal, dan evaluasi champion-challenger.

Model awal yang dipakai sebagai champion adalah PCA Hotelling T²/SPE Q karena inference cepat, dapat dijelaskan, dan sesuai untuk process monitoring multivariat. LSTM Autoencoder menjadi challenger untuk membaca pola temporal nonlinier. Isolation Forest hanya dipakai sebagai baseline lemah atau pembanding negatif jika waktu pengerjaan masih cukup.

Demo utama dimulai dari replay dataset SKAB ke MQTT, pemrosesan data melalui RouteMQ, inference model anomaly detection, publikasi hasil anomali ke MQTT, visualisasi dashboard, lalu simulasi retraining dan promosi model melalui MLflow.

## 3. Latar belakang

Pompa distribusi air merupakan aset penting dalam operasi penyediaan air bersih. Gangguan pada pompa dapat menurunkan tekanan distribusi, mengganggu layanan pelanggan, dan menaikkan biaya perawatan darurat. Pada pemantauan sederhana, alarm sering dibuat dari batas minimum dan maksimum per sensor. Cara ini mudah diterapkan, tetapi kurang cocok untuk gangguan yang muncul dari kombinasi beberapa sensor.

Contohnya, tekanan masih berada dalam rentang normal, tetapi arus motor dan getaran naik bersamaan. Sistem berbasis threshold tunggal dapat melewatkan kondisi seperti itu. Pada sistem pompa, anomali juga dapat berupa valve closing, kebocoran, cavitation, atau rotor imbalance. Pola tersebut lebih tepat dibaca sebagai masalah multivariate time-series anomaly detection.

Dari sisi implementasi, banyak contoh proyek IoT berhenti pada pengiriman data sensor ke MQTT dan dashboard. Proyek ini menempatkan MQTT sebagai protokol transport. Pekerjaan rekayasanya berada pada RouteMQ, yaitu framework aplikasi untuk mengelola routing topic, validasi payload, middleware, queue job, worker, cache, penyimpanan historis, dan observability.

Dari sisi AI, model yang dilatih satu kali dapat menurun kualitasnya ketika distribusi data berubah. Perubahan sensor, karakteristik fluida, atau pola operasi dapat memicu data drift maupun concept drift. Karena itu, MLOps dimasukkan sejak desain awal melalui MLflow Model Registry, monitoring drift, retraining terjadwal, dan aturan promosi model yang jelas.

## 4. Rumusan masalah

Pertanyaan kerja dalam proyek ini adalah:

1. Bagaimana membangun framework aplikasi MQTT, yaitu RouteMQ, yang mendukung routing topic, middleware, controller, queue worker, Redis/ClickHouse, dan observability untuk use case sensor industri?
2. Bagaimana memetakan dataset publik SKAB sebagai surrogate yang masuk akal untuk telemetry pompa distribusi air PDAM tanpa mengklaim adanya data PDAM asli?
3. Bagaimana menerapkan PCA Hotelling T²/SPE Q sebagai champion model untuk mendeteksi anomali multivariat pada sensor pompa?
4. Bagaimana membandingkan champion PCA dengan LSTM Autoencoder sebagai challenger dalam skema evaluasi yang adil?
5. Bagaimana merancang loop MLOps sederhana yang mencakup tracking eksperimen, registry model `PumpAD`, retraining, promosi model, dan guard terhadap kenaikan false alarm?
6. Bagaimana menyajikan hasil deteksi anomali dan status model kepada operator melalui dashboard yang cukup untuk demo tugas besar?

## 5. Batasan masalah

Batasan proyek ditetapkan agar pengerjaan tetap sesuai skala tugas besar.

1. Dataset utama adalah SKAB. Proyek tidak mengklaim memiliki data sensor PDAM asli.
2. Data sensor fisik dari ESP32 atau PLC tidak menjadi target wajib. Replay CSV SKAB dipakai sebagai sumber telemetry.
3. Deployment menggunakan Docker Compose. Kubernetes dan cloud deployment tidak menjadi target v0.
4. Model utama hanya PCA Hotelling T²/SPE Q dan LSTM Autoencoder. Model lain, termasuk Isolation Forest, hanya opsional untuk baseline sederhana.
5. Promosi model dilakukan di lingkungan demo melalui MLflow dan Redis pointer. A/B testing pada trafik produksi tidak dibahas.
6. Sistem dirancang untuk satu atau beberapa station SKAB dalam demo, bukan operasi multi-cabang PDAM.
7. Evaluasi dilakukan pada label SKAB. Klaim performa baru boleh dibuat setelah eksperimen dijalankan.

## 6. Tujuan dan manfaat

### 6.1 Tujuan

Tujuan proyek ini adalah:

1. Membangun RouteMQ sebagai framework aplikasi MQTT untuk pola industrial sensor ingestion.
2. Membuat pipeline replay SKAB ke topic MQTT `factory/skab/{station}/telemetry`.
3. Membuat job inference yang menerapkan PCA Hotelling T²/SPE Q pada rolling window sensor.
4. Membandingkan model champion PCA dengan LSTM Autoencoder sebagai challenger.
5. Mendaftarkan model anomaly detection sebagai `PumpAD` pada MLflow Model Registry.
6. Menerapkan aturan promosi model berbasis `F1_challenger > F1_champion + 0.02` dengan guard false alarm.
7. Menerbitkan hasil deteksi ke topic MQTT `factory/skab/{station}/anomaly`.
8. Menyediakan dashboard operator untuk data sensor, riwayat anomali, versi model, dan ringkasan drift.

### 6.2 Manfaat

Manfaat yang ditargetkan bersifat akademik dan praktis.

| Penerima manfaat | Manfaat |
|---|---|
| Mahasiswa pengembang | Memahami integrasi MQTT framework, DevOps, anomaly detection, dan MLOps dalam satu proyek |
| Operator sistem air | Mendapat contoh dashboard awal untuk membaca indikasi gangguan pompa |
| Engineer maintenance | Mendapat referensi pola sensor yang terkait dengan fault seperti valve closing, leak, cavitation, dan rotor imbalance |
| Pengajar | Mendapat bahan penilaian tugas besar yang menggabungkan rekayasa perangkat lunak dan AI, bukan model saja |

## 7. Tinjauan pustaka singkat

SKAB adalah dataset benchmark untuk anomaly detection pada water circulation testbed. Dataset ini tersedia secara publik dengan lisensi GPL-3.0 dan DOI Kaggle `10.34740/KAGGLE/DSV/1693952`. Dataset berisi 35 file CSV, masing-masing mewakili eksperimen dengan satu event anomali. Kolom sensor yang digunakan meliputi `Accelerometer1RMS`, `Accelerometer2RMS`, `Current`, `Pressure`, `Temperature`, `Thermocouple`, `Voltage`, dan `Volume Flow RateRMS`. Kolom `anomaly` dan `changepoint` dipakai sebagai label evaluasi, bukan sebagai fitur input.

PCA Hotelling T²/SPE Q banyak dipakai pada process monitoring. Hotelling T² mengukur deviasi pada subspace principal component, sedangkan SPE Q mengukur residual di luar subspace tersebut. Kombinasi ini sesuai untuk mendeteksi perubahan pola multivariat pada proses industri. Pada referensi SKAB, pendekatan T²/Q termasuk metode yang layak sebagai baseline kuat untuk outlier detection.

LSTM Autoencoder sering dipakai untuk time-series anomaly detection. Model belajar merekonstruksi window normal. Ketika window berisi pola abnormal, reconstruction error cenderung naik. Pada proyek ini LSTM Autoencoder tidak langsung dianggap lebih baik dari PCA. Model tersebut dipakai sebagai challenger karena biaya training lebih tinggi dan interpretasinya lebih sulit, tetapi dapat menangkap pola temporal yang tidak selalu terbaca oleh PCA.

MLOps membahas siklus hidup model setelah training, termasuk eksperimen, registry, validasi, monitoring, retraining, dan rollback. MLflow menyediakan tracking dan Model Registry, sedangkan Evidently dapat dipakai untuk laporan drift. Proyek ini mengambil subset yang cukup untuk tugas besar, yaitu tracking eksperimen, registry model `PumpAD`, alias champion/challenger, retraining terjadwal, dan promosi dengan guard kualitas.

MQTT adalah protokol publish-subscribe yang sering dipakai dalam IoT. Namun, protokol saja belum memberi struktur aplikasi. RouteMQ diusulkan sebagai lapisan framework agar aplikasi MQTT punya pola yang tertata, seperti route definition, middleware, controller, queue job, worker, dan observability hook.

## 8. Solusi yang diusulkan

Solusi proyek dibagi menjadi dua bagian besar.

Pertama, RouteMQ dibuat sebagai framework aplikasi MQTT. RouteMQ memberi cara deklaratif untuk mendaftarkan handler topic, misalnya `factory/skab/{station}/telemetry`. Payload telemetry divalidasi, diberi correlation ID, disimpan ke Redis sebagai latest reading, disimpan ke ClickHouse sebagai data historis pada tabel `telemetry_observations`, lalu diteruskan ke queue job untuk inference. ClickHouse dipakai sebagai columnar store untuk multivariate sensor time-series dan sudah menjadi jalur TSDB first-class di RouteMQ 0.24. Dengan pola ini, bagian DevOps/RouteMQ menjadi kontribusi framework, bukan hanya konfigurasi broker Mosquitto.

Kedua, sistem anomaly detection dan MLOps dibangun di atas SKAB. Replayer membaca CSV SKAB, mengirim data sensor ke MQTT, lalu RouteMQ menjalankan job anomaly detection. PCA Hotelling T²/SPE Q menjadi champion model awal. LSTM Autoencoder menjadi challenger yang dibandingkan melalui metrik precision, recall, F1, event-level F1, false alarm rate, dan detection delay. Model didaftarkan ke MLflow dengan nama `PumpAD`.

Hasil inference dipublikasikan kembali ke MQTT melalui topic `factory/skab/{station}/anomaly`. Payload anomali berisi station, timestamp, versi model, skor T², skor Q, threshold, status anomali, dan sensor yang paling berkontribusi bila tersedia. Dashboard Streamlit membaca data hasil inference dan menampilkan grafik sensor, riwayat alert, status model, dan laporan drift.

Aturan promosi model dibuat eksplisit. Challenger hanya dipromosikan jika `F1_challenger > F1_champion + 0.02` dan false alarm tidak naik melewati batas penjaga. Guard awal yang dipakai adalah `false_alarm_rate_challenger <= false_alarm_rate_champion * 1.05`, ditambah pemeriksaan model load, schema validation, dan latency inference.

## 9. Arsitektur sistem

### 9.1 Arsitektur tiga lapis

```text
SKAB CSV Replayer
    |
    | publish telemetry
    v
Eclipse Mosquitto
    |
    | factory/skab/{station}/telemetry
    v
RouteMQ Application
    |-- Router
    |-- Middleware
    |-- Controller
    |-- Queue
    |-- Worker
    |-- Redis cache
    |-- ClickHouse telemetry storage
    |
    | anomaly inference job
    v
MLOps Layer
    |-- PCA T²/SPE Q champion
    |-- LSTM Autoencoder challenger
    |-- MLflow Model Registry: PumpAD
    |-- Evidently drift report
    |-- Retraining scheduler
    |
    | publish anomaly
    v
factory/skab/{station}/anomaly
    |
    v
Streamlit Dashboard
```

### 9.2 Alur data online

1. `scripts/replay_skab.py` membaca file SKAB dan mengirim row sensor ke MQTT.
2. Mosquitto menerima telemetry pada `factory/skab/{station}/telemetry`.
3. RouteMQ router mencocokkan topic dan memanggil middleware validasi payload.
4. Controller menyimpan latest reading ke Redis dan data historis ke ClickHouse.
5. Queue mengirim `AnomalyDetectionJob` ke worker.
6. Worker mengambil rolling buffer 60 sample per station.
7. Model champion dari MLflow `models:/PumpAD@champion` menghitung skor T² dan Q.
8. Worker mengirim hasil anomali ke `factory/skab/{station}/anomaly`.
9. Dashboard menampilkan sensor, skor anomali, status model, dan riwayat alert.

### 9.3 Komponen dan stack

| Lapisan | Komponen |
|---|---|
| MQTT broker | Eclipse Mosquitto |
| Framework aplikasi | RouteMQ, Python 3.12+ |
| Storage | ClickHouse 24 untuk telemetry historis sensor, Redis 7 untuk cache, queue, dan active model pointer |
| ML | scikit-learn, numpy, TensorFlow/Keras bila challenger dikerjakan |
| MLOps | MLflow, Evidently, APScheduler |
| Dashboard | Streamlit |
| Observability | Prometheus dan Grafana |
| Orchestration | Docker Compose |

### 9.4 Konfigurasi utama

Konfigurasi mengikuti pola `.env.example` agar demo dapat dijalankan ulang.

| Variabel | Peran |
|---|---|
| `MQTT_BROKER`, `MQTT_PORT`, `MQTT_QOS` | Koneksi ke Mosquitto |
| `ENABLE_TELEMETRY`, `TELEMETRY_CONNECTION=clickhouse`, `TELEMETRY_URL` | Penyimpanan telemetry historis sensor ke ClickHouse melalui RouteMQ 0.24 |
| `ENABLE_REDIS`, `REDIS_HOST`, `REDIS_DB` | Cache latest reading, queue, dan pointer model aktif |
| `MLFLOW_TRACKING_URI` | Lokasi MLflow tracking server |
| `MLFLOW_REGISTERED_MODEL=PumpAD` | Nama model anomaly detection di registry |
| `RETRAIN_INTERVAL_MINUTES=30` | Jadwal retraining demo |
| `PROMOTION_F1_DELTA=0.02` | Selisih F1 minimum untuk promosi challenger |
| `DRIFT_SHARE_THRESHOLD=0.30` | Ambang awal peringatan drift |
| `SKAB_DATA_DIR=./data/skab` | Lokasi dataset SKAB lokal |
| `REPLAY_RATE_HZ=1` | Laju replay telemetry default |

## 10. Metodologi AI dan MLOps

### 10.1 Dataset dan preprocessing

Dataset SKAB ditempatkan pada `data/skab/` dan tidak di-commit ke repository. Struktur yang diharapkan mencakup folder `anomaly-free`, `valve1`, `valve2`, dan `other`. Data dibagi berbasis file eksperimen, bukan random row shuffle, agar tidak terjadi leakage antar waktu.

Fitur input mencakup delapan channel sensor. Kolom `anomaly` dan `changepoint` hanya dipakai untuk evaluasi. Scaler dilatih pada data normal train saja. Parameter awal window adalah 60 sample per station, tanpa padding untuk window yang belum lengkap.

### 10.2 Champion model: PCA Hotelling T²/SPE Q

Langkah training champion:

1. Fit scaler pada data normal train.
2. Fit PCA pada data normal yang sudah diskalakan.
3. Pilih jumlah komponen untuk explained variance sekitar 90 persen.
4. Hitung Hotelling T² dan SPE Q pada train dan validation.
5. Tentukan threshold awal dari percentile validation normal, misalnya p95.
6. Simpan scaler, PCA, threshold, urutan sensor, dan metadata ke MLflow.

Keputusan anomali awal:

```text
is_anomaly = t2_score > t2_threshold OR q_score > q_threshold
```

PCA dipilih sebagai champion karena inference cepat, mudah dijelaskan melalui score timeline, dan cocok untuk process monitoring multivariat.

### 10.3 Challenger model: LSTM Autoencoder

LSTM Autoencoder dilatih pada window normal. Output model adalah rekonstruksi window sensor. Skor anomali dihitung dari reconstruction error, misalnya MAE. Threshold awal dapat memakai p95 atau p99 dari validation normal, tergantung false alarm pada validation.

Challenger ini tidak otomatis dipromosikan. Ia harus mengalahkan champion pada validation atau hold-out split yang sama dan tidak menaikkan false alarm secara berlebihan.

### 10.4 Baseline opsional

Isolation Forest dapat dicatat sebagai baseline lemah jika waktu pengerjaan cukup. Perannya hanya untuk menunjukkan bahwa metode sederhana belum tentu cocok untuk SKAB. Baseline ini tidak menjadi kandidat utama dalam promosi model.

### 10.5 Evaluasi model

Metrik yang dipakai:

| Metrik | Tujuan |
|---|---|
| Precision | Mengukur proporsi alert yang benar |
| Recall | Mengukur fault yang berhasil terdeteksi |
| F1 | Ringkasan precision dan recall |
| Event-level F1 | Menilai apakah event anomali terdeteksi setidaknya sekali |
| False alarm rate | Mengontrol alarm palsu per waktu atau per station |
| Detection delay | Mengukur jarak dari awal anomali ke alert pertama |

Point-wise F1 tidak cukup untuk use case operator. Karena itu, event-level evaluation dan false alarm rate tetap dilaporkan.

### 10.6 MLOps loop

Loop MLOps v0 dirancang sebagai berikut.

```text
Training data snapshot
    -> train candidate model
    -> log run ke MLflow
    -> evaluate pada split validasi tetap
    -> compare dengan champion
    -> promote atau reject
    -> update Redis active model pointer
    -> inference worker reload model
```

Model didaftarkan sebagai:

```text
registered_model = PumpAD
active_alias = champion
candidate_alias = challenger
active_uri = models:/PumpAD@champion
```

Aturan promosi:

```text
promote if:
  F1_challenger > F1_champion + 0.02
  false_alarm_rate_challenger <= false_alarm_rate_champion * 1.05
  model_load_test == pass
  schema_validation == pass
  inference_latency_p95 <= target
```

Rollback manual tetap disiapkan melalui MLflow alias jika model baru gagal load, error inference naik, atau false alarm meningkat jauh dari baseline champion.

## 11. Rencana pengujian

Pengujian dibagi menjadi empat kelompok.

| Kelompok uji | Target | Bukti yang dikumpulkan |
|---|---|---|
| Unit test RouteMQ | Router topic matching, middleware, controller, queue dispatch | Test result dan coverage sederhana |
| Integration test ingestion | SKAB replayer ke Mosquitto, RouteMQ ingest, Redis/ClickHouse write | Log pipeline dan isi storage |
| ML evaluation | PCA, LSTM-AE, threshold, metric evaluation | MLflow run, confusion matrix, metric table |
| MLOps flow test | Retraining, compare, promotion, model hot-swap | MLflow registry version, Redis active URI, dashboard status |

Skenario demo pengujian akhir:

1. Jalankan Docker Compose untuk Mosquitto, Redis, ClickHouse, MLflow, Prometheus, dan Grafana.
2. Jalankan RouteMQ application dan worker.
3. Replay segment normal SKAB dan pastikan dashboard menunjukkan skor rendah.
4. Replay segment anomali, misalnya valve closing, lalu pastikan alert muncul pada `factory/skab/{station}/anomaly`.
5. Jalankan training candidate dan log ke MLflow.
6. Jalankan evaluasi champion-challenger.
7. Jika rule promosi terpenuhi, ubah alias `PumpAD@champion` dan pastikan worker memakai model baru tanpa restart penuh.
8. Catat false alarm rate dan detection delay untuk laporan.

## 12. Jadwal pengerjaan

Rencana waktu mengikuti skala lima minggu pengerjaan dan satu minggu buffer.

| Minggu | Fokus | Luaran mingguan |
|---|---|---|
| 1 | Foundation dan infra | Docker Compose awal, Mosquitto, Redis, ClickHouse, MLflow, struktur RouteMQ, proposal v1 |
| 2 | Ingestion pipeline | SKAB replayer, route `factory/skab/{station}/telemetry`, validasi payload, storage Redis/ClickHouse, dashboard sensor v0 |
| 3 | Model anomaly detection | Loader SKAB, windowing, PCA T²/SPE Q, LSTM-AE awal, evaluasi metric, MLflow tracking |
| 4 | MLOps loop | Model registry `PumpAD`, inference job, Evidently drift report, retraining job, champion-challenger evaluation |
| 5 | Observability dan demo | Prometheus/Grafana metric, dashboard final, demo script, slide, laporan akhir |
| 6 | Buffer | Bug fixing, latihan presentasi, rekaman demo cadangan, revisi dokumen |

## 13. Pembagian tugas

Pembagian tugas awal mengikuti dua anggota tim berikut.

| Anggota | Tugas teknis | Output |
|---|---|---|
| Anggota 1 | Docker Compose, Mosquitto, RouteMQ scaffold, routing DSL, middleware, queue job, Redis active model URI, ClickHouse telemetry schema, SKAB replayer, drift injector, scheduler, CI, Prometheus hook | RouteMQ application dapat menerima telemetry, menyimpan data, menjalankan job inference, dan dieksekusi dalam stack lokal |
| Anggota 2 | SKAB preprocessing, windowing, PCA T²/SPE Q, LSTM-AE, MLflow logging, metric evaluation, Evidently drift report, Streamlit dashboard, proposal, slide, laporan | Model `PumpAD`, metric evaluasi, artifact MLflow, dashboard operator, dan bahan presentasi |

Koordinasi dilakukan melalui issue atau task board sederhana. Setiap akhir minggu perlu ada demo kecil agar risiko integrasi tidak menumpuk di akhir.

## 14. Luaran proyek

Luaran yang ditargetkan pada v0:

1. Source code RouteMQ application untuk ingestion dan processing telemetry.
2. Docker Compose stack untuk Mosquitto, Redis, ClickHouse, MLflow, Prometheus, Grafana, dan aplikasi.
3. Script replay SKAB ke MQTT.
4. Pipeline training PCA Hotelling T²/SPE Q dan LSTM Autoencoder.
5. MLflow experiment dan registered model `PumpAD`.
6. Job inference yang menerbitkan hasil ke `factory/skab/{station}/anomaly`.
7. Retraining job dengan rule promosi champion-challenger.
8. Dashboard Streamlit untuk sensor, anomali, registry model, dan drift.
9. Dokumen proposal, laporan akhir, dan slide presentasi.
10. Demo script untuk presentasi tugas besar.

## 15. Risiko dan mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Docker Compose multi-service tidak stabil | Demo gagal dijalankan | Selesaikan stack pada minggu pertama, pakai healthcheck, sediakan mode dev minimal |
| LSTM Autoencoder terlalu lambat di CPU | Challenger tidak selesai tepat waktu | Batasi arsitektur, pakai early stopping, prioritaskan PCA champion lebih dulu |
| Data leakage pada split SKAB | Evaluasi terlihat terlalu baik | Split berbasis file eksperimen, scaler fit hanya pada train normal, test tidak dipakai untuk threshold |
| False alarm terlalu tinggi | Dashboard terlalu banyak alert | Tuning threshold di validation, laporkan false alarm rate, gunakan guard promosi |
| MLOps loop terlalu besar untuk tugas besar | Scope melebar | Targetkan subset: MLflow registry, retraining job, compare metric, alias promotion, Redis pointer |
| Integrasi RouteMQ dan ML terlambat | Demo tidak utuh | Buat vertical slice sejak minggu kedua: satu topic, satu station, satu model champion |
| Dataset SKAB belum tersedia di mesin demo | Replay gagal | Dokumentasikan download di `data/README.md`, siapkan data lokal sebelum presentasi |
| Dashboard belum selesai | Presentasi sulit diikuti | Mulai dashboard v0 sejak ingestion pipeline, tidak menunggu semua model selesai |

## 16. Daftar pustaka

1. I. D. Katser dan V. O. Kozitsin, "Skoltech Anomaly Benchmark (SKAB)," Kaggle, DOI: 10.34740/KAGGLE/DSV/1693952.
2. waico, "SKAB: Skoltech Anomaly Benchmark," GitHub repository, https://github.com/waico/SKAB.
3. M. Kreuzberger, N. Kühl, dan S. Hirschl, "Machine Learning Operations (MLOps): Overview, Definition, and Architecture," IEEE Access, 2022.
4. D. Sculley et al., "Hidden Technical Debt in Machine Learning Systems," NeurIPS, 2015.
5. M. Zaharia et al., "Accelerating the Machine Learning Lifecycle with MLflow," IEEE Data Engineering Bulletin, 2018.
6. A. Harrou et al., "Amalgamation of anomaly-detection indices for enhanced process monitoring," Journal of Loss Prevention in the Process Industries, 2016.
7. S. Garg et al., "An Evaluation of Anomaly Detection and Diagnosis in Multivariate Time Series," IEEE Transactions on Neural Networks and Learning Systems, 2021.
8. S. Schmidl, P. Wenig, dan T. Papenbrock, "Anomaly Detection in Time Series: A Comprehensive Evaluation," Proceedings of the VLDB Endowment, 2022.
9. A. Maleki et al., "Unsupervised anomaly detection with LSTM autoencoders using statistical data-filtering," Applied Soft Computing, 2021.
10. J. Githinji et al., "Anomaly Detection on Time Series Sensor Data Using Deep LSTM-Autoencoder," IEEE AFRICON, 2023.
11. MLflow, "Model Registry Documentation," https://mlflow.org/docs/latest/model-registry.html.
12. Evidently AI, "Evidently Documentation," https://docs.evidentlyai.com.
13. RouteMQ Framework, https://github.com/ardzz/RouteMQ.
