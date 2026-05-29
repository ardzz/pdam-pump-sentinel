# Demo vertical slice

Dokumen ini mencatat cara menjalankan potongan awal sistem: Mosquitto, SKAB replayer, dan RouteMQ ingestion.

## Verifikasi tanpa service berjalan

Perintah berikut sudah cukup untuk memastikan kontrak awal dan konfigurasi lokal valid.

```bash
uv run pytest tests/unit
uv run ruff check .
docker compose -f infra/docker-compose.dev.yml config
uv run python scripts/replay_skab.py --input tests/fixtures/skab_tiny.csv --station ipa_01 --limit 1 --dry-run
uv run python -m ml.training.train_pca tests/fixtures/skab_tiny.csv /tmp/pdam-pca-smoke --window-size 1 --stride 1 --threshold-quantile 0.95
```

Dry-run mencetak telemetry untuk topic berikut.

```text
factory/skab/ipa_01/telemetry
```

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

Untuk mencatat run ke MLflow lokal, pastikan stack dev berjalan dan tambahkan flag berikut.

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python -m ml.training.train_pca tests/fixtures/skab_tiny.csv /tmp/pdam-pca-smoke --window-size 1 --stride 1 --threshold-quantile 0.95 --log-mlflow --register-model --registered-model-name PumpAD --alias champion
```

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
