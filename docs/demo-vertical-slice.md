# Demo vertical slice

Dokumen ini mencatat cara menjalankan potongan awal sistem: EDA SKAB, Mosquitto, SKAB replayer, training PCA, dan RouteMQ ingestion.

SKAB dipakai sebagai surrogate public water circulation testbed untuk demo akademik. SKAB bukan data operasional PDAM nyata, jadi hasil pada fixture kecil tidak boleh dipakai sebagai klaim performa lapangan atau benchmark.

## Verifikasi tanpa service berjalan

Perintah berikut sudah cukup untuk memastikan kontrak awal dan konfigurasi lokal valid.

```bash
uv run pytest tests/unit
uv run ruff check .
docker compose -f infra/docker-compose.dev.yml config
uv run python scripts/generate_skab_eda.py --input tests/fixtures/skab_tiny.csv --output-dir /tmp/pdam-skab-eda --no-plots
uv run python scripts/replay_skab.py --input tests/fixtures/skab_tiny.csv --station ipa_01 --limit 1 --dry-run
uv run python -m ml.training.train_pca tests/fixtures/skab_tiny.csv /tmp/pdam-pca-smoke --window-size 1 --stride 1 --threshold-quantile 0.95
```

Dry-run mencetak telemetry untuk topic berikut.

```text
factory/skab/ipa_01/telemetry
```

## EDA SKAB sebelum training

Jalankan EDA sebelum training PCA, visualisasi artefak model, dan dashboard. Langkah ini memberi ringkasan data, label, sensor, timestamp, dan file split agar keputusan training bisa diaudit dari artefak awal.

```bash
uv run python scripts/generate_skab_eda.py --input tests/fixtures/skab_tiny.csv --output-dir /tmp/pdam-skab-eda --no-plots
```

Dengan Makefile:

```bash
make skab-eda SKAB_EDA_INPUT=tests/fixtures/skab_tiny.csv SKAB_EDA_OUTPUT_DIR=/tmp/pdam-skab-eda SKAB_EDA_EXTRA_ARGS=--no-plots
```

Untuk manifest beberapa file, tambahkan `SKAB_EDA_SPLIT_MANIFEST=data/manifests/skab-demo.json` atau panggil CLI dengan `--split-manifest`. CLI mencetak JSON berisi path artefak EDA dari API core `ml.datasets.skab_eda.generate_skab_eda_report(input_path, split_manifest_path, output_dir, include_plots)`.

Artefak EDA utama:

- `summary.json` dan `report.md` untuk ringkasan jumlah baris, label, rentang waktu, dan statistik sensor.
- `missingness.csv` untuk missing policy aktual. Training default menolak missing sensor.
- `timestamp_quality.json` untuk parsed timestamps, invalid timestamps, duplicate timestamps, monotonicity, cadence mode, dan max gap.
- `correlation_matrix.csv`, `correlation_heatmap.png`, `sensor_distributions.csv`, dan `rolling_statistics.csv` untuk korelasi, heatmap `sns`, kuantil/IQR, rolling mean, dan rolling std.
- `label_ranges.csv` dan `label_overlay.csv` untuk range anomaly/changepoint dan overlay label terhadap sensor.

## Training PCA offline

Jalankan training PCA T²/SPE Q champion dari CSV SKAB. Contoh berikut memakai fixture kecil untuk smoke test dan menulis artefak lokal ke `/tmp/pdam-pca-smoke`.

```bash
uv run python -m ml.training.train_pca tests/fixtures/skab_tiny.csv /tmp/pdam-pca-smoke --window-size 1 --stride 1 --threshold-quantile 0.95
```

Artefak yang dihasilkan:

- `pca_detector.joblib`
- `metadata.json`
- `metrics.json`
- `scores.csv`

`metadata.json` menyimpan `metric_protocol` dan `provenance`, termasuk konfigurasi, input files, hash, dan versi paket. Pada mode single CSV, metrik dihitung pada input scoring yang sama dengan contoh smoke test, jadi jangan dibaca sebagai hasil final.

Untuk mencatat run ke MLflow lokal, pastikan stack dev berjalan dan tambahkan flag berikut.

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python -m ml.training.train_pca tests/fixtures/skab_tiny.csv /tmp/pdam-pca-smoke --window-size 1 --stride 1 --threshold-quantile 0.95 --log-mlflow --register-model --registered-model-name PumpAD --alias champion
```

## Training PCA dengan split manifest

Workflow split-manifest mengganti contoh single CSV saat memakai beberapa file eksperimen. Manifest berisi daftar file untuk `train`, `validation`, dan `test`, ditambah metadata opsional seperti `schema_version`, `dataset_kind`, `name`, `base_dir`, dan `notes`. Contoh format:

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

`base_dir` harus relatif terhadap lokasi manifest. Loader menolak path absolut, path yang keluar dari `base_dir`, file hilang, dan duplikasi file antar split.

Catatan metodologi:

- Split dilakukan pada level file. Jangan shuffle baris dari file berbeda.
- Kolom label seperti `anomaly` dan `changepoint` dipakai untuk kalibrasi atau evaluasi saja, bukan sebagai fitur model.
- `changepoint` dipisahkan dari target anomaly dan dipakai sebagai transient mask untuk metrik dengan suffix `_excluding_transient`.
- Scaler dan PCA di-fit hanya dari window normal pada split `train`.
- Threshold dikalibrasi pada `validation`.
- Metrik validation berada di top level `metrics.json`. Metrik final dihitung pada `test` yang held-out dan diberi prefix `test_`.
- Event/range metrics meliputi `event_count`, `event_recall`, `missed_events`, `false_alarm_events`, dan `mean_detection_delay_windows`.
- `metadata.json` mencatat `test_split_held_out`, detail `split`, `metric_protocol`, artifact paths, dan `provenance`.
- Jika `--log-mlflow` aktif, simpan manifest, daftar file final tiap split, metrics, metadata, dan provenance sebagai MLflow artifact.

Command split-manifest:

```bash
uv run python -m ml.training.train_pca /tmp/pdam-pca-split --split-manifest data/manifests/skab-demo.json --window-size 1 --stride 1 --threshold-quantile 0.95
```

Dengan Makefile, path manifest dan output tetap dapat dioverride:

```bash
make train-pca-split PCA_SPLIT_MANIFEST=data/manifests/skab-demo.json PCA_SPLIT_OUTPUT_DIR=/tmp/pdam-pca-split
```

Artefak tambahan mode split:

- `split_manifest.json`: manifest resolved relatif untuk audit run.
- `test_scores.csv`: skor held-out test, bila split `test` tidak kosong.
- `metadata.json`: protocol evaluasi, split counts, flag `test_split_held_out`, dan provenance.

## Demo dengan broker lokal

Jalankan infrastruktur lokal.

```bash
make dev
```

Jalankan aplikasi RouteMQ di terminal lain.

```bash
make run
```

Kirim replay SKAB dari CSV lokal.

```bash
uv run python scripts/replay_skab.py --input tests/fixtures/skab_tiny.csv --station ipa_01 --limit 3
```

Untuk mengamati hasil anomali dari terminal terpisah, gunakan subscriber MQTT.

```bash
mosquitto_sub -h localhost -p 1883 -t 'factory/skab/+/anomaly' -v
```

Pada vertical slice v0, payload anomali RouteMQ masih memakai label `anomaly` dari telemetry bila tersedia. PCA T²/SPE Q sudah tersedia sebagai pipeline training offline, tetapi belum dipakai oleh handler MQTT real-time. LSTM Autoencoder belum masuk di tahap ini.
