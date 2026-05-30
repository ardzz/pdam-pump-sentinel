# Data Directory

Dataset SKAB tidak di-commit ke repo (lihat `.gitignore`). Download manual:

## SKAB (Skoltech Anomaly Benchmark)

- Repo: https://github.com/waico/SKAB
- License: GPL-3.0
- Kaggle DOI: 10.34740/KAGGLE/DSV/1693952
- Catatan metodologi: SKAB adalah surrogate public water circulation testbed, bukan data operasional PDAM nyata. Pakai untuk demo akademik, kontrak pipeline, dan validasi metode, bukan klaim performa lapangan PDAM.

```bash
git clone https://github.com/waico/SKAB.git /tmp/skab
cp -r /tmp/skab/data ./data/skab
```

## Struktur yang Diharapkan

```text
data/skab/
├── anomaly-free/
│   └── anomaly-free.csv
├── valve1/
├── valve2/
└── other/
```

## EDA sebelum training

EDA SKAB dipakai sebelum training, visualisasi artefak model, dan dashboard. Tujuannya membaca kualitas data, cakupan label, rentang sensor, dan ringkasan split agar pilihan training PCA tidak langsung lompat ke model.

Contoh untuk satu CSV:

```bash
uv run python scripts/generate_skab_eda.py --input data/skab/anomaly-free/anomaly-free.csv --output-dir /tmp/pdam-skab-eda --no-plots
```

Contoh untuk split manifest:

```bash
uv run python scripts/generate_skab_eda.py --split-manifest data/manifests/skab-demo.json --output-dir /tmp/pdam-skab-eda
```

Output CLI berupa JSON berisi path artefak EDA dari API core `ml.datasets.skab_eda.generate_skab_eda_report(input_path, split_manifest_path, output_dir, include_plots)`.

Artefak EDA research-grade yang diharapkan:

- `summary.json`: jumlah baris, label, rentang waktu, statistik sensor, constant-column flags, dan ringkasan split bila ada.
- `report.md`: laporan ringkas untuk dokumentasi dan demo.
- `sensor_statistics.csv`: mean, std, min, max, dan kuantil sensor.
- `missingness.csv`: missing count dan missing rate per kolom. Training memakai kebijakan ketat, sensor kosong ditolak kecuali eksperimen eksplisit mengizinkan missing.
- `timestamp_quality.json`: jumlah timestamp valid, invalid, duplikat, monotonicity, cadence mode, dan max gap.
- `correlation_matrix.csv`: korelasi antar sensor untuk membaca redundansi dan hubungan proses.
- `sensor_distributions.csv`: kuantil p01 sampai p99 dan IQR per sensor.
- `rolling_statistics.csv`: rolling mean dan rolling std untuk melihat stabilitas sinyal.
- `label_ranges.csv`: range kontigu untuk `anomaly` dan `changepoint`.
- `label_overlay.csv`: baris timestamp, label, dan sensor untuk overlay anomaly/changepoint.

## Split Manifest Training

Training split-manifest memakai daftar file eksplisit, bukan pembagian baris acak dari satu CSV. Simpan manifest sebagai JSON dengan tiga array wajib dan metadata opsional berikut.

```json
{
  "schema_version": 1,
  "dataset_kind": "skab-surrogate-demo",
  "name": "skab-demo",
  "base_dir": "files",
  "notes": {
    "purpose": "academic demo, not real PDAM operations"
  },
  "train": ["normal.csv"],
  "validation": ["validation.csv"],
  "test": ["test.csv"]
}
```

`base_dir` harus relatif terhadap folder manifest dan semua entry split harus tetap berada di bawah `base_dir`. Loader menyimpan `_base_dir` hasil resolve untuk audit lokal, sedangkan artifact `split_manifest.json` menyimpan path relatif agar run dapat direproduksi.

Aturan metodologi:

- Tiap split berisi daftar file. Jangan shuffle baris antar file karena urutan waktu dipakai untuk windowing.
- Fitur model hanya kolom sensor. Kolom label seperti `anomaly` dan `changepoint` tidak boleh masuk ke fitur.
- `changepoint` dipakai sebagai transient mask untuk metrik `_excluding_transient`, bukan sebagai target anomaly utama.
- Fit scaler dan PCA hanya dari window normal pada split `train`.
- Kalibrasi threshold memakai split `validation`.
- Evaluasi akhir dilakukan sekali pada split `test` yang ditahan dari fitting dan kalibrasi. Metrik test memakai prefix `test_`.
- `metrics.json` mencatat precision, recall, F1, false alarm rate, PR-AUC, ROC-AUC, event recall, missed events, false alarm events, dan detection delay windows.
- `metadata.json` mencatat `metric_protocol`, `test_split_held_out`, `split`, `artifact_paths`, dan `provenance` berisi konfigurasi, input files, hash, dan versi paket.
- Saat MLflow aktif, log manifest, daftar file ter-resolve, metrics, metadata, dan provenance sebagai artifact agar run dapat diaudit.

Contoh command split-manifest:

```bash
uv run python -m ml.training.train_pca /tmp/pdam-pca-split --split-manifest data/manifests/skab-demo.json --window-size 1 --stride 1 --threshold-quantile 0.95
```

## Kolom Sensor

- `datetime`: timestamp
- `Accelerometer1RMS`, `Accelerometer2RMS`: vibration
- `Current`: motor current
- `Pressure`: hydraulic pressure
- `Temperature`, `Thermocouple`: thermal
- `Voltage`: supply
- `Volume Flow RateRMS`: flow
- `anomaly`: label (0/1)
- `changepoint`: changepoint label (0/1)
