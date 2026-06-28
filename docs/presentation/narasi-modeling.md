# Narasi Modeling PDAM Pump Sentinel

Dokumen ini untuk presenter bagian modeling. Tujuannya dua: kamu paham modelnya cukup dalam untuk menjawab reviewer, dan kamu punya naskah bicara yang siap dipakai di slide.

Scope dokumen ini hanya modeling: data, feature engineering, PCA T2/Q, evaluasi, dan roadmap model. Bagian observability cukup disentuh sebagai jembatan ke presenter lain.

## 1. Konsep inti yang wajib dikuasai

### 1.1 Gambaran besar model

Model champion kita adalah PCA T2/Q dengan fitur raw dan spectral. Champion berarti model utama yang dipakai sebagai rujukan awal. Challenger berarti kandidat yang bisa mengganti champion kalau metriknya terbukti lebih baik dan false alarm-nya tetap terkendali.

Di project ini:

| Peran | Model |
|---|---|
| Champion | PCA T2/Q, Hotelling T-squared plus Q/SPE residual |
| Challenger utama | LSTM Autoencoder |
| Extra families | Isolation Forest, XGBoost, LightGBM |

Alur singkatnya:

```text
sensor time-series
-> sliding window
-> fitur raw + spectral
-> scaling dari data normal
-> PCA membaca pola normal
-> hitung T2 dan Q/SPE
-> anomali jika T2 atau Q melewati threshold
```

Yang paling penting untuk diingat: PCA champion dipilih bukan karena selalu punya angka terbesar. PCA dipilih karena bisa jalan dari Day-1 tanpa labeled fault data. Ia hanya butuh baseline normal dulu.

### 1.2 Dataset dan framing data

Dataset yang dipakai adalah SKAB, Skoltech Anomaly Benchmark. Ini surrogate water-circulation testbed, bukan data operasional PDAM nyata.

SKAB punya 8 channel sensor:

| Channel | Makna modeling |
|---|---|
| 2 accelerometer atau vibration | tanda getaran mekanik |
| current | tanda beban motor |
| voltage | kondisi suplai listrik |
| pressure | beban hidrolik |
| temperature | kondisi termal |
| thermocouple | pembacaan termal tambahan |
| volume flow rate | aliran proses |

Kalimat aman untuk presentasi: "Kami memakai SKAB sebagai data surrogate untuk membuktikan metodologi dan pipeline. Ini belum boleh dibaca sebagai performa produksi PDAM nyata."

### 1.3 Sliding window

Sensor datang sebagai deret waktu. Model tidak membaca satu titik sensor sendirian, karena anomali pompa sering terlihat sebagai pola beberapa waktu.

Di project:

| Parameter | Nilai |
|---|---|
| Window | 60 samples, sekitar 1 menit pada 1 Hz |
| Stride | 30 samples |
| Overlap | 50% |

Artinya, model membaca potongan 60 sample, lalu bergeser 30 sample untuk potongan berikutnya. Overlap membuat model tidak terlalu lama menunggu window baru.

### 1.4 Standardisasi dan z-score

PCA melihat angka, bukan arti fisik satuan. Kalau fitur pressure dan vibration punya skala angka yang berbeda, fitur yang angkanya besar bisa mendominasi hanya karena satuannya.

Z-score adalah cara belajar yang mudah:

```text
z = (x - mean training normal) / standard deviation training normal
```

Maknanya:

| Nilai z | Arti sederhana |
|---:|---|
| 0 | tepat di pusat normal |
| 1 | satu langkah sebaran di atas pusat normal |
| -1 | satu langkah sebaran di bawah pusat normal |
| besar absolutnya | makin jauh dari kebiasaan normal |

Di implementasi project, scaler yang dipakai adalah RobustScaler dan fit dilakukan pada data normal. Ide besarnya sama: semua fitur dibuat lebih adil sebelum masuk PCA. Statistik scaling tidak boleh dihitung ulang dari test atau window inference, karena itu membuat ukuran normal ikut bergeser.

### 1.5 Fitur spektral dan FFT

FFT adalah cara melihat isi frekuensi dari satu window sinyal. Untuk presentasi, jangan masuk ke hitungan FFT besar. Jelaskan pertanyaannya saja:

```text
Di window ini, irama pompa paling kuat di frekuensi mana?
Energi getaran tersebar di band mana?
```

Fitur spektral yang perlu dikuasai:

| Fitur | Arti sederhana | Kenapa berguna |
|---|---|---|
| Energy | jumlah kekuatan spektrum | melihat seberapa kuat getaran atau sinyal dalam window |
| Dominant frequency | frekuensi dengan magnitude paling besar | membaca nada atau ritme yang paling dominan |
| Spectral centroid | pusat massa frekuensi | melihat apakah energi bergeser ke frekuensi lebih rendah atau tinggi |
| Band energy | energi pada rentang frekuensi tertentu | menangkap perubahan di band yang relevan untuk fault |

Mode feature yang perlu disebut hati-hati:

| Mode | Status |
|---|---|
| raw | didukung live inference |
| spectral | didukung live inference |
| enriched | masih offline |

Kalimat mudah: "Raw memberi level sensor langsung. Spectral memberi ringkasan irama sensor. Enriched masih area evaluasi offline, jadi tidak saya klaim sebagai live serving."

### 1.6 PCA sebagai pencari pola normal

PCA mencari arah utama variasi normal. Bayangkan titik-titik normal digambar di kertas. Kalau saat vibration naik, pressure atau flow juga punya pola tertentu, titik normal akan membentuk awan dengan arah.

PCA membuat "jalan normal" dari awan titik itu. Window baru lalu ditanya:

1. Apakah dia jauh di sepanjang jalan normal?
2. Apakah dia keluar dari jalan normal?

Pertanyaan pertama dijawab oleh T2. Pertanyaan kedua dijawab oleh Q/SPE.

### 1.7 Score atau proyeksi

Setelah PCA punya arah normal, satu window diproyeksikan ke arah itu. Hasil proyeksi disebut score.

Bahasa sederhananya:

```text
score = posisi bayangan window pada jalan PCA
```

Kalau score dekat nol, window dekat pusat normal. Kalau score besar positif atau negatif, window jauh di arah pola PCA yang dipelajari.

Score belum otomatis berarti anomali. Ia baru bahan untuk menghitung T2.

### 1.8 Rekonstruksi

Rekonstruksi berarti PCA mencoba menggambar ulang window dengan pola normal yang ia simpan.

Kalau PCA hanya menyimpan pola utama, ia akan menggambar ulang window seolah-olah window itu masih mengikuti pola utama. Hasil gambar ulang ini disebut rekonstruksi.

Kalimat mudah:

```text
rekonstruksi = versi window menurut bahasa PCA
```

Jika rekonstruksi mirip dengan window asli, berarti window cocok dengan pola normal. Jika tidak mirip, ada sisa yang perlu dibaca.

### 1.9 Residual

Residual adalah selisih antara window asli dan hasil rekonstruksi.

```text
residual = z asli - z rekonstruksi
```

Kalau residual kecil, PCA berhasil menjelaskan window. Kalau residual besar, ada pola yang tidak bisa dijelaskan oleh PCA.

Dalam konteks pompa, residual besar bisa berarti hubungan antar sensor berubah. Misalnya energi vibration naik, tetapi frekuensi dominan atau flow tidak ikut berubah seperti pola normal.

### 1.10 Hotelling T2

T2 mengukur jarak window di dalam subspace PCA yang disimpan. Bahasa jalan:

```text
T2 = seberapa jauh window melaju di jalan normal PCA
```

Secara konsep, T2 memakai score dan membaginya dengan eigenvalue. Eigenvalue memberi toleransi: arah yang memang biasa banyak berubah diberi ruang lebih besar.

Kalimat kunci untuk Q&A:

"T2 tinggi berarti window ekstrem di arah pola normal yang sudah dipelajari. T2 tidak mengukur keluar dari jalan, itu tugas Q/SPE."

### 1.11 Q/SPE

Q/SPE adalah jumlah kuadrat residual. SPE berarti squared prediction error.

Bahasa jalan:

```text
Q/SPE = seberapa jauh window keluar dari jalan normal PCA
```

Q besar berarti window sulit digambar ulang oleh pola PCA. Ini penting karena ada anomali yang tidak ekstrem secara level, tetapi merusak hubungan antar fitur.

Kalimat kunci:

"T2 melihat ekstrem di dalam pola normal. Q/SPE melihat sisa yang tidak bisa dijelaskan PCA. Karena itu keputusan memakai T2 atau Q."

### 1.12 Threshold

Threshold adalah pagar normal. Skor T2 dan Q baru punya makna setelah dibandingkan dengan threshold.

Aturan project untuk PCA champion:

```text
anomali jika T2 > threshold_T2 atau Q > threshold_Q
```

Threshold harus dikalibrasi dari data normal training atau validation normal. Jangan pilih threshold dari test, karena itu membuat evaluasi bocor.

Cara bicara yang aman:

"Threshold bukan angka sakral. Ia adalah batas yang dipasang dari baseline normal. Yang penting, threshold tidak diambil dari test dan dipakai konsisten saat inference."

### 1.13 Champion-challenger

Champion-challenger adalah mekanisme agar model tidak diganti hanya karena satu angka terlihat naik.

Gate project bernama `should_promote` memakai dua syarat:

| Syarat | Nilai |
|---|---|
| F1 challenger | harus lebih besar dari F1 champion + 0.02 |
| FAR challenger | harus <= champion_far * 1.05 |

Artinya, challenger baru boleh promote kalau F1 naik cukup jelas dan false alarm rate tidak memburuk lebih dari batas guard.

Kalimat mudah:

"Kami tidak asal mengganti model. Challenger harus menang F1 dengan margin 0.02 dan false alarm-nya tetap terkendali dengan FAR guard 1.05."

### 1.14 Kenapa PCA-spectral menjadi Day-1 champion

Alasan utamanya:

1. Deployable Day-1 tanpa labeled fault data.
2. Cocok untuk normal-baseline-first.
3. Cepat untuk inference.
4. Bisa diurai per sensor atau per fitur, jadi operator bisa diberi alasan.
5. Tetap punya jalur naik kelas melalui challenger.

Jangan bilang PCA paling canggih. Bilang PCA paling masuk akal untuk kondisi awal PDAM, ketika fault label belum matang.

## 2. Narasi presentasi (speaker script)

Bagian ini ditulis sebagai kata-kata yang bisa langsung diucapkan. Sesuaikan sedikit dengan gaya bicaramu, tapi jangan ubah angka evaluasi atau framing split.

### Slide 2, Why This Problem Matters, sekitar 1 menit

"Di bagian modeling, masalah yang ingin kami tangkap bukan cuma nilai sensor yang melewati batas min atau max.

Pada pompa distribusi air, anomali bisa muncul sebagai pola lintas sensor. Contohnya, satu sensor masih terlihat normal kalau dilihat sendiri, tetapi kombinasinya dengan vibration, pressure, current, dan flow sudah tidak wajar.

Karena itu threshold konvensional lemah untuk tiga kasus: contextual anomaly, sensor drift, dan collective anomaly. Contextual anomaly berarti angkanya tampak aman, tetapi tidak cocok dengan konteks operasi. Sensor drift berarti pembacaan bergeser pelan. Collective anomaly berarti beberapa sensor berubah bersama-sama dan baru terlihat kalau dibaca sebagai pola.

Jadi kontribusi modeling kami adalah membaca pola multivariate window sensor, bukan hanya membuat alarm satu sensor. Dari sini baru masuk ke pilihan model champion."

### Slide 3, Scope and Honest Data Framing, sekitar 1 menit

"Sebelum masuk ke hasil model, saya ingin mengunci framing data dulu.

Dataset kami adalah SKAB, Skoltech Anomaly Benchmark. Ini public water-circulation testbed dengan fault injection terkontrol. Sensor yang dipakai mencakup vibration, current, voltage, pressure, temperature, thermocouple, dan volume flow rate.

Yang penting, SKAB bukan data operasional PDAM nyata. Jadi klaim kami bukan bahwa model ini sudah terbukti di lapangan PDAM. Klaim yang benar adalah: SKAB kami pakai sebagai surrogate untuk membuktikan metodologi, pipeline, dan evaluasi modeling secara akademik.

Untuk MVP, kami juga tidak mengklaim sensor ESP32 asli atau deployment Kubernetes. Fokus modeling saya adalah bagaimana data time-series dipotong menjadi window, dibuat fitur raw dan spectral, lalu dideteksi dengan PCA T2/Q sebagai champion."

### Slide 7, Why PCA-Spectral as Day-1 Champion, sekitar 1 menit 15 detik

"Pertanyaan yang biasanya muncul adalah: kenapa champion-nya PCA-spectral, bukan XGBoost atau model supervised yang angkanya terlihat lebih tinggi?

Jawabannya ada di kondisi Day-1. Di PDAM real, saat sistem pertama dipasang, kita biasanya belum punya inventori fault berlabel yang lengkap. Kita punya data operasi normal dulu. Karena itu pendekatan yang masuk akal adalah normal-baseline-first.

PCA-spectral cocok untuk kondisi itu. Model dilatih pada window normal, lalu membaca apakah window baru masih mirip pola normal atau tidak.

Di PCA T2/Q, ada dua alarm. T2 membaca jarak sepanjang pola normal yang sudah dipelajari. Kalau window jauh di arah principal component, T2 naik. Q atau SPE membaca residual, yaitu sisa yang tidak bisa direkonstruksi oleh PCA. Kalau hubungan antar fitur rusak, Q naik.

Keputusan anomali dibuat dengan OR: kalau T2 melewati threshold atau Q melewati threshold, window dianggap anomali.

Alasan kami memilih PCA-spectral sebagai champion adalah karena dia deployable dari Day-1, cepat, bisa diurai per sensor atau fitur, dan lebih mudah dijelaskan ke operator. Tapi kami tidak berhenti di PCA. LSTM Autoencoder tetap disiapkan sebagai challenger, dan family lain seperti Isolation Forest, XGBoost, serta LightGBM juga ada di jalur eksperimen."

### Slide 8, Honest Evaluation Spectrum, sekitar 1 menit 10 detik

"Slide ini adalah slide yang paling penting untuk menjaga klaim tetap jujur.

Angka F1 tidak boleh dicampur, karena setiap split menjawab pertanyaan berbeda.

Untuk PCA-spectral champion, setting-nya train normal-only dan test di valve2 plus other. F1-nya sekitar 0.58. Angka ini tidak terlihat paling tinggi, tetapi ini setting yang deployable dan menguji novel-fault generalization.

Supervised cross-group punya labeled anomalies di train. F1-nya sekitar 0.60 dengan AUC 0.937. Ini tetap lebih jujur daripada random window, karena masih bicara generalisasi ke kelompok fault baru.

Supervised file-level stratified ada di sekitar F1 0.70.

Lalu ada supervised in-distribution, chrono 80/20, yang Kaggle-comparable. Di sini XGB sekitar 0.909 dan LGBM sekitar 0.905. Tapi angka ini butuh contoh berlabel dari setiap fault type di train. Jadi angka ini tidak boleh dijual sebagai readiness untuk novel fault Day-1.

Terakhir, random-window bisa sampai sekitar 0.985, tetapi itu max leakage. Itu bukan klaim generalisasi yang valid.

Karena itu kami juga tidak memakai point-adjustment, karena bisa melebih-lebihkan hasil. Jadi framing kami: PCA-spectral adalah champion deployment awal, supervised adalah jalur naik kelas setelah label matang."

### Slide 13, Limitations and Roadmap, modeling part, sekitar 45 detik

"Untuk modeling, ada beberapa batasan yang sengaja kami buka.

Pertama, live PCA inference saat ini mendukung raw dan spectral. Mode enriched masih offline. Jadi saya tidak akan mengklaim enriched sebagai live serving.

Kedua, scheduled retraining saat ini masih PCA-family. Supervised promotion dengan label operator adalah roadmap setelah data label cukup matang.

Ketiga, hasil supervised yang tinggi perlu dibaca sesuai split. XGB dan LightGBM bagus untuk in-distribution ketika fault type sudah ada di train, tetapi bukan jawaban Day-1 saat fault label belum tersedia.

Roadmap modeling kami jelas: kumpulkan operator labels dari triage, pakai label itu untuk supervised challenger, lalu promote hanya kalau gate lolos: F1 naik lebih dari 0.02 dan FAR masih dalam guard 1.05. Setelah ini saya serahkan ke bagian observability untuk menunjukkan bagaimana pipeline dan bukti operasionalnya dipantau."

## 3. Antisipasi Q&A

### Q1. Kenapa tidak langsung pakai XGBoost atau LightGBM kalau angkanya bisa sekitar 0.90?

Jawaban: angka sekitar 0.90 itu berasal dari supervised in-distribution chrono 80/20, dengan fault type yang sudah muncul di train. Itu bagus sebagai upper bound, tetapi bukan kondisi Day-1 PDAM. Untuk Day-1, kita belum punya labeled fault inventory. PCA-spectral dipilih karena bisa belajar dari normal baseline dulu.

### Q2. Apakah F1 0.58 untuk PCA-spectral tidak terlalu rendah?

Jawaban: angka itu harus dibaca sesuai split: train normal-only, test valve2 plus other. Ini setting yang lebih jujur untuk novel-fault generalization. Kami tidak menyembunyikan bahwa supervised bisa lebih tinggi jika label tersedia. Justru slide 8 memisahkan semua angka supaya tidak overclaim.

### Q3. Apa beda T2 dan Q/SPE dalam satu kalimat?

Jawaban: T2 mengukur seberapa jauh window bergerak di sepanjang pola normal PCA. Q/SPE mengukur sisa yang tidak bisa dijelaskan setelah PCA merekonstruksi window. T2 itu jauh di jalan normal, Q itu keluar dari jalan normal.

### Q4. Kenapa keputusan memakai T2 atau Q, bukan salah satu saja?

Jawaban: karena anomali bisa muncul dalam dua bentuk. Ada window yang ekstrem tetapi masih mengikuti pola normal, itu ditangkap T2. Ada window yang tidak terlalu ekstrem tetapi merusak hubungan antar fitur, itu ditangkap Q. Aturan OR membuat detector peka terhadap dua bentuk itu.

### Q5. Threshold PCA diambil dari mana?

Jawaban: threshold harus dikalibrasi dari training normal atau validation normal. Jangan ambil dari test, karena test harus tetap menjadi ujian akhir. Saat inference, window baru hanya dibandingkan dengan threshold yang sudah disimpan.

### Q6. Apa risiko kalau scaler di-fit pada semua data?

Jawaban: itu data leakage. Jika scaler melihat test atau window inference, ukuran normal ikut berubah. Model terlihat lebih baik dari kondisi nyata. Karena itu RobustScaler fit hanya pada data normal training.

### Q7. Apa fungsi FFT dalam model ini?

Jawaban: FFT mengubah window sensor dari domain waktu menjadi ringkasan frekuensi. Dari situ kita ambil fitur seperti energy, dominant frequency, spectral centroid, dan band energy. Fitur ini membantu model membaca perubahan ritme pompa, bukan hanya level sensor mentah.

### Q8. Apakah SKAB bisa mewakili PDAM?

Jawaban: SKAB bisa dipakai sebagai surrogate water-circulation testbed untuk demo akademik dan metodologi. Tetapi SKAB bukan data operasional PDAM nyata. Jadi klaim kami adalah readiness metodologi dan pipeline, bukan performa lapangan PDAM.

### Q9. Apakah PCA bisa menjelaskan sensor mana yang bermasalah?

Jawaban: PCA lebih mudah diurai dibanding banyak model supervised. Kita bisa melihat kontribusi fitur, residual per fitur, dan apakah alarm utama datang dari T2 atau Q. Itu membantu operator memahami sensor atau fitur mana yang paling menyumbang skor.

### Q10. Kapan challenger boleh mengganti champion?

Jawaban: saat gate `should_promote` lolos. Syaratnya, challenger F1 harus lebih besar dari champion + 0.02, dan FAR challenger harus <= champion_far * 1.05. Jadi model baru tidak cukup hanya naik F1, false alarm-nya juga harus tetap terkendali.

### Q11. Apa posisi LSTM Autoencoder di project ini?

Jawaban: LSTM Autoencoder adalah challenger utama. Ia juga bisa belajar normal-only melalui reconstruction error, tetapi lebih berat dan lebih sulit dijelaskan dibanding PCA. Karena itu PCA menjadi champion awal, sedangkan LSTM-AE menjadi kandidat pembanding dan promosi.

### Q12. Kenapa tidak pakai point-adjustment agar F1 lebih tinggi?

Jawaban: point-adjustment bisa membuat hasil terlihat terlalu bagus karena satu deteksi dalam rentang anomali bisa mengangkat banyak titik. Untuk deck ini, kami tidak memakainya supaya evaluasi tidak melebih-lebihkan generalisasi.

### Q13. Apa yang dimaksud champion bukan model terbaik absolut?

Jawaban: champion adalah model yang paling layak dipakai untuk kondisi saat ini. Untuk Day-1, syaratnya bukan hanya akurasi, tetapi juga bisa dilatih tanpa label fault, cepat, dapat dijelaskan, dan aman dijadikan baseline. Jika label sudah matang dan challenger lolos gate, champion bisa diganti.

### Q14. Apa jawaban jika reviewer bilang data normal-only tidak cukup?

Jawaban: benar untuk tahap jangka panjang. Normal-only cukup untuk cold start dan deteksi deviasi awal, tetapi bukan akhir roadmap. Karena itu kami menyiapkan operator labeling, supervised challenger, dan champion-challenger gate sebagai jalur setelah label cukup matang.

### Q15. Apa satu kalimat penutup bagian modeling?

Jawaban: "Modeling kami sengaja dimulai dari PCA-spectral yang bisa jalan dari normal baseline Day-1, lalu disiapkan untuk naik ke supervised challenger ketika label operator sudah cukup dan gate promosi membuktikan model baru lebih baik tanpa menaikkan false alarm secara berlebihan."

## Sumber di repo

- `docs/research/belajar-pca-t2-q-spectral-manual.md`
- `docs/plans/design.md`
- `docs/presentation/labeling-strategy-notes.md`
- `docs/plans/sprint-remaining.md`
- `docs/presentation/deck-outline.md`
