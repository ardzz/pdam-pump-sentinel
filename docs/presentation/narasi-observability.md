# Narasi Observability PDAM Pump Sentinel

Dokumen ini adalah catatan belajar sekaligus naskah bicara untuk pemilik bagian observability. Fokusnya adalah DevOps, observability, dan plumbing MLOps pada project akademik PDAM Pump Sentinel. Bagian modeling hanya disebut seperlunya agar alur presentasi nyambung.

Prinsip utama saat bicara: stack ini adalah bukti lokal Docker Compose yang bisa diuji, bukan klaim deployment produksi cloud, Kubernetes, atau operasi SRE penuh.

## 1. Konsep inti yang wajib dikuasai

### Observability dalam bahasa sederhana

Observability adalah kemampuan sistem untuk menjawab tiga pertanyaan saat sesuatu terjadi:

- Apa yang sedang terjadi?
- Bagian mana yang terdampak?
- Seberapa parah dan apa langkah triage berikutnya?

Di project ini, observability dibaca lewat tiga sinyal utama:

| Sinyal | Arti | Contoh di project |
|---|---|---|
| Metrics | Angka yang berubah dari waktu ke waktu | latency inference, anomaly score, telemetry freshness, drift share |
| Logs | Jejak kejadian dan urutan proses | banner fase demo, error runtime, hasil job retraining atau drift |
| Health | Status hidup dan siapnya service | `/health`, `/ready`, Prometheus scrape health, service checks di Streamlit |

Kalimat pembelaan yang aman: observability di sini membantu reviewer melihat pipeline berjalan, membaca gejala error, dan menghubungkan anomaly, drift, retraining, serta tindakan operator pada satu stack lokal.

### Tiga lapis arsitektur

Saat menjelaskan observability, jangan mulai dari tool. Mulai dari lapisan sistem.

| Lapis | Isi | Peran observability |
|---|---|---|
| DevOps | Docker Compose, Prometheus, Grafana, healthchecks | Menjaga service lokal bisa dipantau dan dicek statusnya |
| RouteMQ application | MQTT router, middleware, controllers, Redis, ClickHouse | Mengekspos `/metrics`, menyimpan telemetry, memproses inference |
| MLOps | training, MLflow registry, Evidently drift, retraining, champion-challenger promotion | Memberi bukti model lifecycle, drift, retrain, promotion, dan hot-swap demo |

Ini yang dimaksud vertical slice: data sensor masuk, diproses, disimpan, diberi skor anomali, dipantau, lalu dipakai sebagai bukti untuk demo MLOps.

### Prometheus scrape dan endpoint `/metrics`

Prometheus tidak mendorong data ke aplikasi. Prometheus mengambil data secara berkala dari target yang punya endpoint metrics.

Di project ini:

- Aplikasi RouteMQ mengekspos endpoint `/metrics`.
- Endpoint itu menggabungkan metrik framework RouteMQ dan metrik domain `pumpad_*`.
- Prometheus scrape aplikasi RouteMQ.
- Prometheus juga scrape `mosquitto-exporter` untuk kesehatan broker MQTT.
- Grafana membaca data dari Prometheus untuk dashboard.

Cara menjelaskannya: RouteMQ membuka papan angka di `/metrics`, Prometheus memotret papan itu berkala, lalu Grafana mengubah angka itu jadi panel yang bisa dibaca manusia.

### Metrik `pumpad_*` yang perlu diingat

Tidak perlu menghafal semua nama internal. Hafalkan maknanya dan sebutkan metrik kunci ini:

- Inference latency: waktu yang dibutuhkan inference, dibaca sebagai p50, p95, dan p99.
- Anomaly score: skor anomali dari model saat sensor diproses.
- Drift share: porsi data yang terdeteksi drift oleh Evidently.
- Drift detected atau drift state: status apakah drift sedang terdeteksi.
- Telemetry freshness: seberapa baru data telemetry yang terakhir diterima.
- Retraining jobs by result: jumlah job retraining berdasarkan hasilnya.
- Persistence writes: bukti write ke storage berhasil atau error.
- `pumpad_operator_action_state`: status tindakan operator seperti ack, mute, dan note.
- Observability schema/build marker: penanda versi skema observability dan build agar screenshot bisa dikaitkan dengan versi stack.

Kalimat pembelaan: metrik ini dipilih karena mencakup jalur utama dispatch, inference, persistence, drift, retrain, dan respons operator.

### Grafana dashboards

Grafana dipakai sebagai bukti visual untuk reviewer. Dashboard yang boleh disebut adalah:

| Dashboard | Fungsi |
|---|---|
| `pumpad-observability` | RouteMQ pipeline: dispatch -> inference -> persistence |
| `pumpad-mlops` | MLOps loop, latency, anomaly score, drift, retraining, operator action evidence |
| `pumpad-system-health` | scrape health, dependency health, freshness, active model age |
| `pumpad-mqtt-broker` | broker MQTT: clients, throughput, uptime |

Poin penting: dashboard ini bukan bukti production SRE. Dashboard ini bukti bahwa stack demo lokal punya telemetry yang bisa diaudit.

### Streamlit operator console

Streamlit adalah sisi operator. Grafana kuat untuk engineer, Streamlit dibuat agar operator bisa melihat status dan menjalankan triage sederhana.

Halaman yang perlu disebut:

- Overview: status banner, Observability Snapshot, KPI, station picker.
- Live Sensors: auto-refresh 5 detik untuk status sensor dan score.
- Anomaly History: severity buckets 1h, 24h, 7d, drilldown, tombol operator ack, mute, note.
- Model Registry: name, version, activated_at.
- Drift Reports: ringkasan drift dan training.
- System Health: auto-refresh 10 detik, probe MLflow, Redis, ClickHouse, MQTT.
- Runbook: runbook insiden dan triage berbasis metrik.

Operator actions disimpan di ClickHouse tabel `operator_actions`. Aksi itu juga muncul sebagai metric `pumpad_operator_action_state` dan terlihat di panel Grafana MLOps.

### Alert rules lokal

Alert rules yang sudah ada berada di Prometheus lokal. Yang perlu disebut:

- app scrape health.
- telemetry freshness.
- inference errors.
- persistence write errors.
- drift report age.
- active model age.
- rule anomali `PDAMHighSeverityAnomalyEvents`.

Jawaban jujur jika ditanya incident routing: rule lokal sudah ada, tetapi Slack, email, webhook, Alertmanager, dan PagerDuty belum ada. Itu future work.

### ClickHouse telemetry dan Redis cache

ClickHouse dan Redis punya peran berbeda.

ClickHouse adalah penyimpanan telemetry historis:

- `telemetry_observations` menyimpan observasi telemetry dan anomaly score.
- `operator_actions` menyimpan action operator seperti ack, mute, dan note.
- Data ini bisa dipakai untuk panel dashboard dan bukti demo.

Redis adalah cache dan queue lokal:

- latest reading untuk pembacaan sensor terbaru.
- active model URI atau metadata model aktif.
- queue untuk jalur MLOps seperti drift dan retraining.

Kalimat aman: ClickHouse menjawab riwayat, Redis menjawab state cepat yang dibutuhkan aplikasi saat demo berjalan.

### Beda bukti lokal dan klaim production

Ini bagian yang harus diulang beberapa kali.

Yang sudah bisa diklaim:

- Docker Compose lokal dapat menaikkan Mosquitto, Redis, ClickHouse, MLflow, Prometheus, Grafana, dan app.
- RouteMQ expose `/metrics`.
- Prometheus scrape app dan `mosquitto-exporter`.
- Grafana dan Streamlit menampilkan bukti pipeline, MLOps loop, health, dan operator action.
- Demo T+0 sampai T+8 berjalan, dengan optional T+9 untuk evidence observability.

Yang tidak boleh diklaim:

- Sudah production cloud.
- Sudah Kubernetes.
- Sudah SRE on-call process.
- Sudah ada Slack, email, webhook, Alertmanager, atau PagerDuty routing.
- Sudah memakai sensor PDAM nyata.

Kalimat siap pakai: "Kami sengaja menyebut ini local observability evidence. Artinya stack observability sudah bisa dibuktikan di Docker Compose, tetapi belum kami jual sebagai deployment produksi."

### Perintah demo yang relevan

Perintah yang boleh disebut saat menjelaskan readiness:

```bash
make dev
make run
make dashboard
make replay
make demo
DEMO_EXTRA_ARGS="--observability-evidence" make demo
make screenshots TAG=<tag>
make mlflow-report
```

Untuk slide, jangan tenggelam di command. Pakai command hanya untuk menunjukkan bahwa bukti bisa direproduksi.

## 2. Narasi presentasi (speaker script)

### Slide 4: Architecture, bagian DevOps dan observability

Target waktu: sekitar 1 menit 10 detik.

Naskah:

"Di slide ini saya fokus ke sisi DevOps dan observability. Sistem kami dibagi menjadi tiga lapis. Lapis pertama adalah DevOps: Docker Compose, Prometheus, Grafana, dan healthchecks. Ini yang membuat semua service demo bisa dinaikkan dan dipantau secara lokal.

Lapis kedua adalah RouteMQ application. Di sini data MQTT masuk lewat router, melewati middleware dan controller, lalu state cepat masuk ke Redis dan telemetry historis masuk ke ClickHouse. RouteMQ juga expose `/metrics`, jadi pipeline aplikasinya tidak menjadi black box.

Lapis ketiga adalah MLOps. Bagian modeling dilatih dan didaftarkan ke MLflow, drift dibaca oleh Evidently, retraining menghasilkan kandidat model, lalu champion-challenger promotion mengubah alias model.

Jadi observability kami bukan dashboard tempelan. Observability membaca vertical slice dari sensor replay, ingestion, inference, persistence, drift, retraining, sampai tindakan operator. Tapi framingnya tetap lokal Docker Compose, bukan production cloud."

Catatan bicara:

- Tekankan "tiga lapis" sebelum menyebut tool.
- Jangan bilang Kubernetes atau SRE production.
- Kalau waktunya pendek, tutup dengan: "yang kami buktikan adalah sistem demo bisa diamati dari ujung ke ujung di stack lokal."

### Slide 5: What is implemented today

Target waktu: sekitar 1 menit.

Naskah:

"Yang sudah terimplementasi hari ini adalah local observability evidence. Pertama, aplikasi RouteMQ punya endpoint `/metrics` yang menggabungkan metrik framework dan metrik domain `pumpad_*`. Ini mencakup latency inference, anomaly score, drift, freshness telemetry, retraining job, persistence writes, dan operator action state.

Kedua, Prometheus scrape aplikasi dan `mosquitto-exporter`. Dari situ Grafana punya empat dashboard utama: `pumpad-observability` untuk pipeline RouteMQ, `pumpad-mlops` untuk bukti MLOps loop, `pumpad-system-health` untuk health dan freshness, serta `pumpad-mqtt-broker` untuk broker MQTT.

Ketiga, Streamlit menjadi operator console. Ada Overview, Live Sensors, Anomaly History, Model Registry, Drift Reports, System Health, dan Runbook. Operator action seperti ack, mute, dan note disimpan di ClickHouse, lalu juga muncul sebagai metric `pumpad_operator_action_state`.

Yang penting, saya tidak mengklaim ini production alerting. Alert rules lokal sudah ada, tetapi routing Slack, email, Alertmanager, atau PagerDuty belum ada."

Catatan bicara:

- Pakai istilah "sudah terimplementasi" hanya untuk fitur yang benar ada.
- Ulangi "lokal" saat menyebut Prometheus alert rules.

### Slide 6: Demo storyboard T+0 sampai T+9

Target waktu: sekitar 1 menit 15 detik.

Naskah:

"Slide ini adalah peta demo. Di T+0, model champion sudah aktif dan dashboard hijau. Di T+1, kami replay segment normal dari SKAB, jadi anomaly score rendah. Di T+2, kami replay segment anomalous seperti valve closing, score naik, dan alert anomali dipublish ke MQTT.

Di T+3, kami inject drift sintetis untuk mensimulasikan sensor aging. Di T+4, Evidently menghasilkan `DriftResult`, dan state drift mulai terlihat di evidence. Di T+5, retraining dijalankan. Dalam operasi penuh ini bisa dijadwalkan, tetapi untuk demo prosesnya dibuat deterministik agar reviewer bisa melihat fasenya.

Di T+6, kita verifikasi versi challenger di MLflow. Di T+7, alias `@champion` dipromote ke challenger. Di T+8, inference hot-swap dalam proses demo dan dashboard kembali hijau.

Opsional T+9 adalah fase observability evidence. Di sini kami capture Grafana pipeline, Grafana MLOps, system health, Streamlit Observability Snapshot, dan Runbook. Jadi T+9 bukan fitur baru, tetapi bukti bahwa semua sinyal tadi bisa ditunjukkan setelah demo berjalan."

Catatan bicara:

- Jangan terlalu lama di T+5 sampai T+8 karena modeling owner akan membahas detail model.
- Untuk observability, tekankan T+9 sebagai bukti audit.

### Slide 9: MLOps loop evidence

Target waktu: sekitar 1 menit.

Naskah:

"Dari sisi observability, slide MLOps ini saya baca sebagai bukti lifecycle, bukan hanya bukti model akurat. Ada beberapa sinyal yang harus kelihatan: model aktif di registry, drift share dan drift state, retraining job berdasarkan result, latency inference, anomaly score, dan action operator.

MLflow menunjukkan versi dan alias model. Grafana MLOps menunjukkan apakah loop itu meninggalkan jejak di metrics. Streamlit Model Registry membantu operator melihat name, version, dan activated_at tanpa membuka detail MLflow.

Bagian yang paling penting untuk observability adalah operator action evidence. Kalau anomali severity tinggi muncul, operator bisa ack, mute, atau note dari Anomaly History. Aksi itu masuk ke ClickHouse tabel `operator_actions`, muncul sebagai `pumpad_operator_action_state`, dan terlihat di row Grafana MLOps. Jadi respons manusia juga masuk ke evidence, tidak hanya angka model."

Catatan bicara:

- Sebut "lifecycle evidence" agar tidak mengambil porsi modeling.
- Jawaban jika ditanya otomatisasi: scheduler drift dan retraining ada di balik env flag dan tidak aktif default.

### Slide 10: Pipeline observability evidence

Target waktu: sekitar 1 menit.

Naskah:

"Slide ini menjawab pertanyaan: kalau data sensor masuk, bagaimana kita tahu pipeline benar-benar jalan? Dashboard `pumpad-observability` membaca jalur dispatch, inference, dan persistence.

Di jalur dispatch, kita melihat traffic dari MQTT ke RouteMQ. Di jalur inference, kita membaca latency dan anomaly score. Di jalur persistence, kita melihat write ke ClickHouse dan freshness telemetry. Kalau telemetry berhenti masuk, freshness akan memburuk. Kalau write error muncul, alert rule lokal bisa menangkapnya.

Jadi observability pipeline ini bukan hanya chart cantik. Panelnya dipilih untuk menjawab kegagalan praktis: app tidak terscrape, broker tidak sehat, inference error, persistence error, atau data yang sudah basi. Sekali lagi, ini adalah bukti lokal Docker Compose. Belum ada klaim bahwa ini sudah dipasang ke production cloud."

Catatan bicara:

- Pakai urutan dispatch, inference, persistence.
- Sebut freshness karena itu mudah dipahami reviewer.

### Slide 11: System health dan SLO evidence

Target waktu: sekitar 55 detik.

Naskah:

"Di slide ini, kata SLO kami pakai sebagai gaya pembacaan health, bukan sebagai kontrak SLA produksi. Dashboard `pumpad-system-health` menunjukkan apakah target Prometheus terscrape, apakah dependency sehat, apakah telemetry masih fresh, dan apakah active model terlalu tua.

Alert rules lokal mencakup app scrape health, telemetry freshness, inference errors, persistence write errors, drift report age, active model age, dan `PDAMHighSeverityAnomalyEvents`. Ini cukup untuk demo akademik karena reviewer bisa melihat sinyal gejala dan runbook triage.

Yang belum ada adalah incident routing produksi. Belum ada Alertmanager, Slack, email, webhook, atau PagerDuty. Jadi kalau ada pertanyaan, jawaban jujurnya: rule lokal ada, proses eskalasi produksi belum."

Catatan bicara:

- Jangan menyamakan SLO panel dengan SRE production process.
- Tekankan "dasar runbook".

### Slide 12: Operator console dan runbook

Target waktu: sekitar 1 menit.

Naskah:

"Streamlit di project ini bukan cuma visualisasi sensor. Kami pakai sebagai operator console ringan. Di Overview, operator melihat status banner, Observability Snapshot, KPI, dan pilihan station. Di Live Sensors, halaman auto-refresh 5 detik supaya kondisi terbaru cepat terlihat. Di Anomaly History, operator bisa melihat severity bucket 1 jam, 24 jam, dan 7 hari, lalu masuk ke drilldown.

Bagian triage ada di tombol operator: ack, mute, dan note. Aksi ini tidak hanya tampilan. Aksi disimpan di ClickHouse `operator_actions`, diekspos sebagai metric `pumpad_operator_action_state`, lalu muncul di Grafana MLOps.

System Health auto-refresh 10 detik dan mengecek MLflow, Redis, ClickHouse, serta MQTT. Runbook memberi langkah triage berbasis metrik. Jadi saat demo ada anomali, presenter bisa menjelaskan bukan hanya modelnya mendeteksi, tetapi operator punya tempat untuk membaca status dan mencatat respons."

Catatan bicara:

- Kalau reviewer bertanya soal label operator, bedakan ack/mute/note dari supervised label intake.
- Jangan klaim triage UI sudah menjadi workflow incident management produksi.

### Slide 13: Limitations, bagian observability

Target waktu untuk bagian observability: sekitar 30 sampai 45 detik.

Naskah:

"Untuk bagian limitation, saya ingin jujur. Observability kami kuat untuk bukti lokal Docker Compose, tetapi belum production SRE. Belum ada Slack, email, webhook, Alertmanager, atau PagerDuty routing.

Secara arsitektur, inference anomali masih sinkron di controller, bukan Queue -> Worker untuk anomaly. Queue dipakai untuk job MLOps seperti drift dan retraining. Scheduler drift dan retraining ada di balik `ENABLE_DRIFT_SCHEDULER` dan `DRIFT_INTERVAL_MINUTES`, serta tidak aktif default. Scheduled retraining saat ini PCA-only berbasis path SKAB, bukan rolling-window dari ClickHouse. Hot-swap demo masih process-local, belum polling alias runtime penuh. Beberapa halaman dashboard juga memakai refresh manual atau cache TTL.

Menurut saya ini tetap defensible karena limitation-nya spesifik. Kami tidak menutup-nutupi batas MVP, dan next step produksi sudah jelas."

Catatan bicara:

- Tutup dengan nada tenang: "sudah jalan, tahu batasnya, tahu next step".
- Jangan minta maaf berlebihan. Limitation yang jelas justru membuat klaim lebih dipercaya.

## 3. Antisipasi Q&A

### Apakah sistem ini sudah production-ready?

Belum. Yang kami buktikan adalah stack lokal Docker Compose dengan observability yang bisa direproduksi. Production-ready masih butuh deployment cloud atau on-prem yang benar, hardening security, incident routing, scaling, backup, dan operasional SRE.

### Apakah sudah pakai Kubernetes?

Belum. Scope project ini Docker Compose. Di slide, Docker Compose adalah bukti orchestration lokal, bukan klaim Kubernetes.

### Apakah alert sudah masuk Slack atau PagerDuty?

Belum. Prometheus alert rules lokal sudah ada, termasuk `PDAMHighSeverityAnomalyEvents`, tetapi belum ada Alertmanager, Slack, email, webhook, atau PagerDuty routing.

### Kalau belum ada incident routing, kenapa tetap disebut observability?

Karena observability tidak harus berarti on-call production lengkap. Di MVP ini, observability berarti sistem mengekspos metrics, health, dashboard, local alert rules, dan evidence yang bisa dipakai untuk triage lokal.

### Apa fungsi `/metrics`?

`/metrics` adalah endpoint RouteMQ yang dibaca Prometheus. Isinya gabungan metrik framework dan metrik `pumpad_*`, seperti latency inference, anomaly score, drift, freshness, retraining job, persistence writes, operator action state, dan schema/build marker.

### Prometheus scrape apa saja?

Prometheus scrape aplikasi RouteMQ dan `mosquitto-exporter`. Aplikasi memberi metrics pipeline dan domain PDAM Pump Sentinel. `mosquitto-exporter` memberi health broker MQTT seperti clients, throughput, dan uptime.

### Apa bedanya Grafana dan Streamlit?

Grafana untuk observability engineer: metrics, dashboards, health, alert evidence. Streamlit untuk operator console: status banner, live sensors, anomaly history, model registry, drift reports, system health, dan runbook.

### Mengapa perlu ClickHouse kalau sudah ada Prometheus?

Prometheus menyimpan time series metrics untuk monitoring. ClickHouse menyimpan telemetry dan event domain seperti `telemetry_observations` dan `operator_actions`. Jadi Prometheus bagus untuk sinyal operasional, ClickHouse bagus untuk riwayat data aplikasi dan bukti operator action.

### Redis dipakai untuk apa?

Redis dipakai untuk latest reading, active model URI atau metadata model aktif, dan queue MLOps. Redis memberi state cepat agar dashboard dan service tidak selalu membaca storage historis.

### Apakah inference anomali berjalan lewat queue worker?

Untuk kondisi saat ini, tidak. README menyatakan inferensi anomali berjalan sinkron di `anomaly_controller.py`. Queue atau worker dipakai untuk job MLOps seperti drift report dan retraining. Ini limitation yang harus disebut jujur.

### Apakah scheduler drift dan retraining selalu aktif?

Tidak. Scheduler ada di balik env flag `ENABLE_DRIFT_SCHEDULER` dan `DRIFT_INTERVAL_MINUTES`, serta tidak aktif default pada compose. Untuk demo, beberapa fase dibuat deterministik agar reviewer bisa melihat alurnya.

### Apakah retraining sudah semua model?

Belum. Scheduled retraining saat ini PCA-only berbasis path SKAB. Belum otomatis melatih LSTM-AE atau supervised model dari rolling-window ClickHouse.

### Apakah hot-swap model sudah penuh seperti production?

Belum. Demo hot-swap bersifat process-local. Repo juga mencatat bahwa polling alias runtime penuh belum menjadi klaim utama MVP. Untuk presentasi, sebut bahwa hot-swap dibuktikan di proses demo, bukan sistem production multi-instance.

### Apakah operator action sama dengan label supervised?

Tidak sama. Ack, mute, dan note adalah action triage operator yang tersimpan di `operator_actions`. Itu memberi evidence respons operator. Supervised promotion berbasis label operator masih future work.

### Apa bukti paling kuat untuk slide observability?

Bukti paling kuat adalah kombinasi `/metrics`, Prometheus scrape, empat dashboard Grafana, Streamlit Observability Snapshot, operator action yang tersimpan di ClickHouse, dan T+9 screenshot capture setelah demo berjalan.

### Apa yang harus dilakukan jika reviewer menekan soal local-only?

Jawab langsung: "Benar, ini local-only Docker Compose evidence. Kami tidak mengklaim production cloud. Nilai kontribusinya adalah pipeline observability sudah dirancang, diekspos lewat metrics, dan bisa dibuktikan end-to-end di demo akademik."

### Apa perintah untuk membuktikan stack bisa dijalankan?

Gunakan `make dev` untuk infra, `make run` untuk app, `make dashboard` untuk Streamlit, `make replay` untuk replay data, `make demo` untuk orkestrasi T+0 sampai T+8, `DEMO_EXTRA_ARGS="--observability-evidence" make demo` untuk T+9, `make screenshots TAG=<tag>` untuk screenshot, dan `make mlflow-report` untuk report MLflow.

### Kalau dashboard tidak update, apakah observability gagal?

Belum tentu. Beberapa halaman memakai refresh manual atau cache TTL. Live Sensors auto-refresh 5 detik dan System Health 10 detik, sedangkan halaman lain bisa membutuhkan reload atau menunggu cache. Ini juga masuk known limitation.

### Bagaimana menjawab pertanyaan "ini logs-nya mana"?

Jawab bahwa bukti utama observability di deck adalah metrics, health, telemetry store, dan dashboard. Logs tetap ada sebagai jejak runtime dan banner fase demo, tetapi project ini tidak mengklaim centralized logging production.

### Satu kalimat penutup untuk bagian observability?

"Observability kami menunjukkan bahwa demo ini bisa diaudit: data masuk, inference berjalan, hasil disimpan, drift dan retraining punya evidence, health terbaca, dan tindakan operator ikut tercatat, semuanya pada stack lokal yang jujur batasnya."

## Sumber di repo

- `README.md`: Bukti Observability Portfolio, Quick Start, perintah tambahan, dan Known Limitations.
- `docs/plans/design.md`: arsitektur tiga lapis, DevOps layer, MLOps loop, dan storyboard T+0 sampai T+8.
- `docs/presentation/screenshot-checklist.md`: canonical dashboards, URL lokal, per-phase capture, dan T+9 observability evidence.
- `docs/plans/sprint-remaining.md`: observability modules yang sudah landed dan gap arsitektur yang harus dibingkai jujur.
- `docs/presentation/deck-outline.md`: nomor slide, key message, dan batas narasi deck.
