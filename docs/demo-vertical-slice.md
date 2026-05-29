# Demo vertical slice

Dokumen ini mencatat cara menjalankan potongan awal sistem: Mosquitto, SKAB replayer, dan RouteMQ ingestion.

## Verifikasi tanpa service berjalan

Perintah berikut sudah cukup untuk memastikan kontrak awal dan konfigurasi lokal valid.

```bash
uv run pytest tests/unit
uv run ruff check .
docker compose -f infra/docker-compose.dev.yml config
uv run python scripts/replay_skab.py --input tests/fixtures/skab_tiny.csv --station ipa_01 --limit 1 --dry-run
```

Dry-run mencetak telemetry untuk topic berikut.

```text
factory/skab/ipa_01/telemetry
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

Pada vertical slice v0, payload anomali masih memakai label `anomaly` dari telemetry bila tersedia. PCA T²/SPE Q dan LSTM Autoencoder belum masuk di tahap ini.
