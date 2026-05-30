# Data Directory

Dataset SKAB tidak di-commit ke repo (lihat `.gitignore`). Download manual:

## SKAB (Skoltech Anomaly Benchmark)

- Repo: https://github.com/waico/SKAB
- License: GPL-3.0
- Kaggle DOI: 10.34740/KAGGLE/DSV/1693952

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

## Split Manifest Training

Training split-manifest memakai daftar file eksplisit, bukan pembagian baris acak dari satu CSV. Simpan manifest sebagai JSON dengan tiga array berikut.

```json
{
  "train": ["relative/path/to/normal.csv"],
  "validation": ["relative/path/to/validation.csv"],
  "test": ["relative/path/to/test.csv"]
}
```

Aturan metodologi:

- Tiap split berisi daftar file. Jangan shuffle baris antar file karena urutan waktu dipakai untuk windowing.
- Fitur model hanya kolom sensor. Kolom label seperti `anomaly` dan `changepoint` tidak boleh masuk ke fitur.
- Fit scaler dan PCA hanya dari window normal pada split `train`.
- Kalibrasi threshold memakai split `validation`.
- Evaluasi akhir dilakukan sekali pada split `test` yang ditahan dari fitting dan kalibrasi.
- Saat MLflow aktif, log manifest dan daftar file ter-resolve sebagai artifact agar run dapat diaudit.

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
