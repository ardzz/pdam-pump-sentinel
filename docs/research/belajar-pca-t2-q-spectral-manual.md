---
title: Belajar Menghitung PCA T2-Q Manual dengan Fitur Spektral
tags:
  - pca
  - anomaly-detection
  - spectral-features
  - pump-sentinel
aliases:
  - Belajar Menghitung PCA T2-Q Manual dengan Fitur Spektral
created: 2026-06-24
project: PDAM Pump Sentinel
model: PCA T2/Q spectral
---

# Belajar Menghitung PCA T2-Q Manual dengan Fitur Spektral

Catatan ini adalah panduan pelan-pelan untuk menghitung **PCA Hotelling T2 dan Q atau SPE** dengan fitur spektral. Tujuannya bukan menulis program. Tujuannya adalah bisa duduk dengan kertas, kalkulator biasa, dan tabel kecil, lalu paham angka model berasal dari mana.

Kita akan mulai dari nol. Benar-benar nol. Mulai dari jumlah, rata-rata, selisih, kuadrat, varians, standar deviasi, lalu naik sedikit demi sedikit sampai PCA, T2, Q/SPE, threshold, dan keputusan anomali.

Konteksnya tetap **PDAM Pump Sentinel**. Sistem ini memantau kondisi pompa distribusi air. Data sensor pompa dipotong menjadi window, window diubah menjadi fitur spektral, lalu PCA T2/Q membaca apakah window itu masih mirip pola normal atau sudah mencurigakan.

Catatan penting proyek: **SKAB adalah surrogate data dari water circulation testbed, bukan data operasional PDAM nyata**. Jadi angka dan contoh di catatan ini dipakai untuk belajar dan demo akademik. Jangan membaca angka contoh sebagai klaim performa lapangan PDAM.

## Cara membaca catatan ini

Jangan lompat dulu ke rumus PCA. PCA kelihatan menakutkan karena banyak simbol. Padahal isi kecilnya hanya beberapa gerakan sederhana:

1. kurangi angka dengan rata-rata,
2. bagi dengan standar deviasi,
3. kalikan angka lalu jumlahkan,
4. buat garis utama pola normal,
5. lihat titik baru jauh atau tidak dari pola itu,
6. lihat titik baru masih bisa digambar ulang oleh pola itu atau tidak.

Kalau kamu lupa matematika, itu tidak apa-apa. Catatan ini memang ditulis untuk keadaan itu.

Yang perlu disiapkan:

| Alat | Dipakai untuk |
|---|---|
| Kertas | menulis tabel kecil |
| Kalkulator | menghitung kuadrat, akar, pembagian |
| Pensil | menggambar garis PC1 secara kasar |
| Kesabaran | karena PCA lebih mudah kalau tidak buru-buru |

Yang tidak akan dilakukan di sini:

| Tidak dilakukan | Kenapa |
|---|---|
| Tutorial Python | fokusnya hitung manual |
| Model keluarga lain | fokusnya PCA T2/Q spectral saja |
| Turunan control limit lanjutan | terlalu berat untuk jalur utama belajar manual |
| FFT besar lengkap | tidak enak dihitung tangan, jadi kita sederhanakan |

## Gambaran besar dalam satu cerita

Bayangkan pompa normal punya kebiasaan. Saat energi getaran naik sedikit, frekuensi dominan juga ikut naik sedikit. Saat energi turun, frekuensi dominan juga turun. Polanya seperti titik-titik yang berbaris dekat satu garis miring.

PCA belajar garis miring itu. Garis itu disebut **principal component pertama**, atau **PC1**.

Setelah itu, window baru datang. PCA bertanya dua hal:

| Pertanyaan | Nama skor | Bahasa sederhana |
|---|---|---|
| Apakah titik baru terlalu jauh di sepanjang garis normal? | Hotelling T2 | jauh tetapi masih di jalan yang benar |
| Apakah titik baru jauh dari garis normal? | Q atau SPE | keluar dari jalan yang biasa |

Keputusan dasarnya:

```text
anomali jika T2 melewati ambang T2 atau Q melewati ambang Q
```

Dalam bentuk pendek:

$$
\mathrm{is\_anomaly}=(T^2>\text{threshold}_{T^2})\ \mathrm{OR}\ (Q>\text{threshold}_Q)
$$

Di sini $\mathrm{OR}$ berarti cukup salah satu benar. Kalau T2 tinggi, anomali. Kalau Q tinggi, anomali. Kalau dua-duanya tinggi, jelas anomali.

![Ilustrasi alur PCA T2 dan Q](assets/pca-t2-q-manual/pipeline-overview.png)

Gambar di atas adalah peta besar. Jangan hafalkan dulu semua kotaknya. Cukup lihat bahwa perjalanan model selalu dari kiri ke kanan: window sensor dibuat menjadi fitur, fitur dibuat adil dengan scaling, PCA membaca pola normal, lalu T2 dan Q dipakai untuk keputusan.

Kalau sudah paham bagian ini, kamu harus bisa menjawab:

| Pertanyaan | Jawaban pendek |
|---|---|
| T2 melihat apa? | jauh di arah pola normal PCA |
| Q/SPE melihat apa? | sisa error yang tidak bisa dijelaskan PCA |
| Kenapa pakai dua skor? | karena dua jenis keanehan bisa berbeda |

## Peta perjalanan belajar

Kita akan berjalan seperti tangga kecil.

![Peta hitung manual PCA T2-Q](assets/pca-t2-q-manual/manual-calculation-map.png)

Gambar ini adalah versi visual dari urutan hitung di kertas. Kalau nanti bingung sedang berada di langkah mana, kembali ke gambar ini dan cari kotak yang sedang kamu kerjakan.

| Tangga | Isi | Tujuan |
|---:|---|---|
| 1 | aritmetika dasar | tidak takut angka |
| 2 | scaling | semua fitur dibuat sebanding |
| 3 | vektor dan dot product | paham proyeksi PCA |
| 4 | matrix dan transpose | paham bentuk tabel PCA |
| 5 | covariance dan correlation | paham fitur bergerak bersama |
| 6 | fitur spektral | paham asal fitur dari window sensor |
| 7 | eigenvector dan eigenvalue | paham arah PC dan besar variasi |
| 8 | PCA projection | menghitung score $t$ |
| 9 | reconstruction dan residual | menghitung bagian yang bisa dan tidak bisa dijelaskan |
| 10 | T2 | menghitung ekstrem di subspace PCA |
| 11 | Q/SPE | menghitung error residual |
| 12 | threshold | membuat batas normal |
| 13 | inference | memutuskan window baru normal atau anomali |

Kita pakai contoh utama yang kecil:

| Isi contoh | Nilai |
|---|---:|
| Window normal training | 5 window |
| Fitur spektral | 2 fitur |
| Principal component yang disimpan | 1 PC |
| Window baru untuk inferensi | 1 window |

## Bagian 1. Aritmetika dasar yang dibutuhkan

Bagian ini mungkin terasa sangat dasar. Tetap penting, karena PCA hanya menyusun ulang operasi kecil ini.

### 1.1 Angka dan data kecil

Misalkan kita punya lima angka energi getaran dari lima window normal:

```text
8, 9, 10, 11, 12
```

Setiap angka adalah satu nilai fitur. Kita belum bicara PCA. Kita hanya bertanya, angka-angka ini pusatnya di mana dan sebarannya seberapa lebar.

### 1.2 Sum atau jumlah

**Sum** artinya jumlahkan semua angka.

Kita akan pakai simbol $\mathrm{sum}$. Simbol $\mathrm{sum}$ berarti penjumlahan semua nilai dalam daftar.

Untuk data:

```text
8, 9, 10, 11, 12
```

Jumlahnya:

$$
\mathrm{sum}=8+9+10+11+12=50
$$

Bahasa bayinya: semua angka dimasukkan ke satu keranjang, lalu dihitung total isi keranjangnya.

Cara membacanya di kertas:

| Langkah kecil | Maksud |
|---|---|
| tulis semua angka | jangan ada yang hilang |
| beri tanda `+` di antara angka | artinya semua digabung |
| hitung totalnya | total itu disebut $\mathrm{sum}$ |

Jadi $\mathrm{sum}$ bukan operasi aneh. $\mathrm{sum}$ hanya nama pendek untuk "jumlahkan semua isi daftar".

### 1.3 Banyak data, simbol $n$

Kita akan pakai simbol $n$ untuk jumlah data.

Di contoh ini ada lima angka:

$$
n=5
$$

Jadi $n$ bukan nilai sensor. $n$ hanya jumlah baris atau jumlah window.

Bahasa bayinya: kalau data adalah antrean anak, $n$ adalah berapa anak yang berdiri di antrean. Bukan tinggi badan anak. Bukan berat anak. Hanya banyaknya anak.

Hati-hati membedakan dua huruf ini:

| Simbol | Dipakai untuk | Contoh |
|---|---|---|
| $n$ kecil | jumlah window atau jumlah baris data training | `n = 5` window normal |
| $N$ besar | jumlah sample di dalam satu window sensor | `N = 60` sample dalam window |

Di Bagian 1 sampai Bagian 4, kita lebih sering memakai $n$. Di bagian spektral, kita akan memakai $N$ untuk panjang window.

### 1.4 Mean atau rata-rata

**Mean** adalah pusat sederhana dari data.

Kita akan pakai simbol $\mu$ untuk mean. Huruf ini sering ditulis sebagai huruf Yunani, tetapi di catatan ini kita tulis $\mu$ agar mudah diketik dan dibaca.

Rumus mean:

$$
\mu = \frac{\mathrm{sum}}{n}
$$

Untuk data energi tadi:

$$
\mu = \frac{50}{5}=10
$$

Artinya pusat data `8, 9, 10, 11, 12` adalah `10`.

Bahasa bayinya: kalau semua angka diratakan supaya sama tinggi, semuanya menjadi `10`.

Mean juga bisa dibayangkan sebagai titik tengah keseimbangan. Angka `8` dan `9` menarik ke kiri atau ke bawah. Angka `11` dan `12` menarik ke kanan atau ke atas. Di tengahnya ada `10`.

Kalau data sensor pompa normal punya mean energi `10`, artinya energi normal biasanya berkumpul di sekitar `10`, walaupun tiap window boleh sedikit naik atau turun.

### 1.5 Deviation atau selisih dari rata-rata

**Deviation** adalah jarak satu angka dari mean.

Kita akan pakai simbol $d$ untuk deviation.

Rumusnya:

$$
d = x - \mu
$$

Di sini:

| Simbol | Arti |
|---|---|
| $x$ | satu angka data |
| $\mu$ | mean data |
| $d$ | selisih angka dari mean |

Untuk data energi:

| $x$ | $\mu$ | $d=x-\mu$ |
|---:|---:|---:|
| 8 | 10 | -2 |
| 9 | 10 | -1 |
| 10 | 10 | 0 |
| 11 | 10 | 1 |
| 12 | 10 | 2 |

Kalau $d$ negatif, angka berada di bawah rata-rata. Kalau $d$ positif, angka berada di atas rata-rata. Kalau $d$ nol, angka tepat di rata-rata.

Tanda deviation memberi arah:

| Nilai $d$ | Bahasa bayi | Contoh |
|---:|---|---|
| negatif | angka ada di bawah pusat | `8 - 10 = -2` |
| nol | angka pas di pusat | `10 - 10 = 0` |
| positif | angka ada di atas pusat | `12 - 10 = 2` |

Di PCA, deviation penting karena model tidak hanya melihat angka mentah. Model melihat angka itu berbeda seberapa jauh dari kebiasaan normal.

### 1.6 Square atau kuadrat

**Square** artinya angka dikalikan dengan dirinya sendiri.

Kita tulis $d^2$ untuk kuadrat dari $d$.

Contoh:

$$
(-2)^2=(-2)(-2)=4
$$

$$
(-1)^2=(-1)(-1)=1
$$

$$
0^2=0\cdot0=0
$$

$$
1^2=1\cdot1=1
$$

$$
2^2=2\cdot2=4
$$

Kenapa deviation dikuadratkan? Karena kalau langsung dijumlahkan, nilai negatif dan positif bisa saling menghapus.

Lihat ini:

$$
-2+(-1)+0+1+2=0
$$

Padahal data jelas menyebar. Supaya jarak tidak saling menghapus, kita kuadratkan.

Bahasa bayinya: kuadrat membuat jarak berubah menjadi angka positif. Jarak ke bawah dan jarak ke atas sama-sama dihitung sebagai besar jarak, bukan saling membatalkan.

Kuadrat juga memberi hukuman lebih besar untuk jarak yang besar:

| Jarak | Kuadrat | Rasa |
|---:|---:|---|
| 1 | 1 | kecil |
| 2 | 4 | bukan dua kali, tetapi empat kali |
| 3 | 9 | jauh makin terasa |

Karena itu T2 dan Q nanti juga memakai kuadrat. Model ingin jarak besar kelihatan jelas.

### 1.7 Variance atau varians

**Variance** adalah ukuran seberapa menyebar data dari mean.

Kita akan pakai simbol $s^2$ untuk sample variance.

Rumus sample variance:

$$
s^2 = \frac{\sum_{i=1}^{n} d_i^2}{n-1}
$$

Di sini:

| Simbol | Arti |
|---|---|
| $s^2$ | sample variance |
| $d$ | deviation dari mean |
| $d^2$ | deviation kuadrat |
| $n$ | jumlah data |
| $n-1$ | pembagi untuk sample variance |

Kenapa pakai $n-1$, bukan $n$? Untuk latihan ini, cukup ingat aturan praktisnya: saat kita memakai beberapa data normal sebagai contoh dari proses yang lebih besar, kita pakai pembagi $n-1$. Ini disebut sample variance.

Bahasa bayinya begini. Kita tidak punya semua kemungkinan kondisi pompa normal di dunia. Kita hanya punya beberapa contoh window normal. Karena contoh itu kecil, sebarannya mudah terlihat terlalu kecil. Pembagi $n-1$ sedikit membesarkan perkiraan sebaran agar lebih adil.

Untuk contoh lima window:

```text
n = 5
n - 1 = 4
```

Jadi kita membagi dengan `4`, bukan `5`.

Aturan pegangan:

| Situasi | Pembagi yang dipakai di catatan ini |
|---|---:|
| menghitung varians dari contoh training normal | $n-1$ |
| menghitung rata-rata biasa | $n$ |

Kalau kamu hanya ingin lulus hitungan manual catatan ini, jangan pusing dulu dengan bukti statistiknya. Pegang saja: varians sample memakai $n-1$.

Untuk energi:

| $x$ | $d$ | $d^2$ |
|---:|---:|---:|
| 8 | -2 | 4 |
| 9 | -1 | 1 |
| 10 | 0 | 0 |
| 11 | 1 | 1 |
| 12 | 2 | 4 |

Jumlah kuadrat deviation:

$$
\sum d^2 = 4 + 1 + 0 + 1 + 4 = 10
$$

Karena `n = 5`, maka `n - 1 = 4`.

$$
s^2 = \frac{10}{4}=2.5
$$

Jadi varians energi adalah `2.5`.

### 1.8 Standard deviation atau standar deviasi

**Standard deviation** adalah akar dari varians.

Kita akan pakai simbol $s$ untuk sample standard deviation.

Rumus:

$$
s = \sqrt{s^2}
$$

Di sini `sqrt` berarti akar kuadrat.

Untuk energi:

$$
s = \sqrt{2.5}=1.581
$$

Cara meraba akarnya:

```text
cari angka yang kalau dikuadratkan mendekati 2.5:

1.5^2   = 2.250  (kurang)
1.6^2   = 2.560  (lebih)
1.58^2  = 2.496  (kurang dikit)
1.581^2 = 2.500  (pas)

jadi sqrt(2.5) = 1.581
```

Kenapa standar deviasi lebih enak dari varians? Karena satuannya kembali mirip satuan data asli. Kalau data energi satuannya unit energi, standar deviasi juga terasa seperti unit energi. Varians terasa seperti unit energi kuadrat, jadi kurang nyaman dibaca.

Bahasa bayinya: standar deviasi adalah ukuran satu langkah normal dari pusat. Kalau `s = 1.581`, maka beda sekitar `1.581` dari mean terasa seperti satu langkah sebaran normal.

Nanti z-score memakai standar deviasi sebagai penggaris:

```text
z = 1 berarti 1 langkah standar deviasi di atas mean
z = -1 berarti 1 langkah standar deviasi di bawah mean
z = 2 berarti 2 langkah standar deviasi di atas mean
```

### 1.9 Ringkasan aritmetika dasar

Untuk data `8, 9, 10, 11, 12`:

| Langkah | Hasil |
|---|---:|
| sum | 50 |
| n | 5 |
| mean $\mu$ | 10 |
| deviation | -2, -1, 0, 1, 2 |
| squared deviation | 4, 1, 0, 1, 4 |
| sample variance $s^2$ | 2.5 |
| sample standard deviation $s$ | 1.581 |
 
Kalau sudah paham bagian ini, kamu harus bisa:

| Kamu harus bisa | Contoh |
|---|---|
| menghitung jumlah | `8 + 9 + 10 + 11 + 12 = 50` |
| menghitung mean | `50 / 5 = 10` |
| menghitung deviation | `8 - 10 = -2` |
| menghitung kuadrat deviation | `(-2)^2 = 4` |
| menghitung standar deviasi | `sqrt(2.5) = 1.581` |

## Bagian 2. Kenapa scaling dibutuhkan sebelum PCA

PCA melihat angka. PCA tidak tahu arti satuan.

Misalkan kita punya dua fitur:

| Fitur | Contoh nilai | Satuan |
|---|---:|---|
| Energi getaran | 8 sampai 12 | unit energi |
| Frekuensi dominan | 29 sampai 32 | Hz |

Rentangnya tidak sama. Satu fitur bisa punya angka besar karena satuannya memang besar, bukan karena lebih penting.

Contoh yang lebih ekstrem:

| Fitur | Rentang |
|---|---:|
| Suhu | 25 sampai 35 |
| Energi FFT | 1000 sampai 9000 |

Kalau langsung PCA tanpa scaling, fitur energi FFT bisa mendominasi hanya karena angkanya besar. PCA bisa mengira energi FFT paling penting, padahal mungkin hanya beda satuan.

### 2.1 Ide scaling

Scaling membuat tiap fitur punya bahasa yang sama.

Kita akan memakai **z-score**.

Untuk satu nilai mentah $x$, mean training $\mu$, dan standar deviasi training $s$, z-score ditulis:

$$
z_i = \frac{x_i - \mu}{s}
$$

Di sini:

| Simbol | Arti |
|---|---|
| $x$ | nilai fitur mentah |
| $\mu$ | mean fitur dari data training normal |
| $s$ | standar deviasi fitur dari data training normal |
| $z$ | nilai fitur setelah scaling |

Bahasa bayinya:

1. $x-\mu$ bertanya, angka ini beda berapa dari rata-rata?
2. dibagi $s$ bertanya, beda itu berapa kali ukuran sebaran normal?

Kalau `z = 0`, nilai tepat di rata-rata training. Kalau `z = 1`, nilai berada satu standar deviasi di atas rata-rata. Kalau `z = -2`, nilai berada dua standar deviasi di bawah rata-rata.

Anggap $\mu$ dan $s$ dari training normal sebagai penggaris tetap.

```text
mean training = titik nol penggaris
std training  = jarak satu garis pada penggaris
z-score       = posisi angka pada penggaris itu
```

Penggaris ini tidak boleh berubah saat inference. Kalau penggaris ikut berubah untuk setiap window baru, kita tidak lagi membandingkan window baru dengan normal lama. Kita hanya membuat ukuran baru yang selalu menyesuaikan diri.

Contoh rasa:

| z-score | Arti |
|---:|---|
| `0` | tepat di pusat normal |
| `0.5` | setengah langkah standar deviasi di atas pusat |
| `1` | satu langkah standar deviasi di atas pusat |
| `-1` | satu langkah standar deviasi di bawah pusat |
| `3` | tiga langkah standar deviasi, mulai terasa jauh |

### 2.2 Statistik training tidak boleh dicampur inference

Ini aturan penting.

Mean dan standar deviasi harus dihitung dari **training normal**. Saat window baru datang, kita tidak menghitung mean dan standar deviasi baru dari window itu. Kita memakai $\mu$ dan $s$ dari training.

Kenapa? Karena kita ingin bertanya:

```text
window baru ini aneh atau tidak jika dibandingkan dengan kebiasaan normal yang sudah dipelajari?
```

Kalau mean dan standar deviasi dihitung ulang dari window baru, ukuran normalnya ikut berubah. Itu seperti memindahkan gawang saat bola ditendang.

Kalau sudah paham bagian ini, kamu harus bisa:

| Kamu harus bisa | Penjelasan |
|---|---|
| menjelaskan scaling | membuat fitur beda satuan jadi sebanding |
| menghitung z-score | kurangi mean, lalu bagi standar deviasi |
| menghindari leakage | mean dan standar deviasi inference tetap dari training normal |

## Bagian 3. Vektor, dot product, matrix, dan transpose

PCA memakai bahasa vektor dan matrix. Jangan panik. Untuk contoh dua fitur, bentuknya kecil sekali.

### 3.1 Vektor

**Vektor** adalah daftar angka yang punya urutan.

Kalau satu window punya dua fitur setelah scaling, misalnya:

```text
z_E = 2.530
z_F = -0.351
```

Kita bisa menulisnya sebagai vektor:

```text
z = [2.530, -0.351]
```

Di sini:

| Simbol | Arti |
|---|---|
| $z$ | vektor fitur yang sudah di-scaling |
| $z_E$ | komponen z-score untuk energi |
| $z_F$ | komponen z-score untuk frekuensi |

Bahasa bayinya: vektor adalah kotak kecil berisi angka berurutan. Kotak pertama untuk energi, kotak kedua untuk frekuensi.

Urutan kotak tidak boleh tertukar.

| Posisi kotak | Isi di contoh ini |
|---:|---|
| kotak 1 | energi $E$ |
| kotak 2 | frekuensi $F$ |

Kalau kita menulis `[2.530, -0.351]`, artinya bukan sekadar dua angka. Artinya:

```text
kotak energi    = 2.530
kotak frekuensi = -0.351
```

Kalau urutannya tertukar menjadi `[-0.351, 2.530]`, artinya berubah total. PCA akan membaca energi sebagai frekuensi dan frekuensi sebagai energi.

### 3.2 Panjang vektor

Kadang kita butuh memastikan vektor arah punya panjang 1. Untuk vektor dua angka:

```text
p = [a, b]
```

Panjang vektor:

$$
\mathrm{length}=\sqrt{a^2+b^2}
$$

Contoh:

```text
p = [0.707, 0.707]

length = sqrt(0.707^2 + 0.707^2)
       = sqrt(0.500 + 0.500)
       = sqrt(1.000)
       = 1.000
```

Vektor dengan panjang 1 disebut unit vector. PCA loading biasanya memakai unit vector agar proyeksi rapi.

### 3.3 Dot product

**Dot product** adalah cara mengalikan dua vektor dengan panjang sama, lalu menjumlahkan hasilnya.

Bahasa bayinya: pasangkan kotak pertama dengan kotak pertama, kotak kedua dengan kotak kedua, kalikan tiap pasangan, lalu jumlahkan.

Misalkan:

```text
a = [a1, a2]
b = [b1, b2]
```

Dot product ditulis:

$$
a \cdot b = a_1b_1 + a_2b_2
$$

Contoh:

$$
[2,3]\cdot[4,5] = 2\cdot4 + 3\cdot5 = 8 + 15 = 23
$$

Tabelnya seperti ini:

| Kotak | Vektor pertama | Vektor kedua | Kali pasangan |
|---:|---:|---:|---:|
| 1 | 2 | 4 | 8 |
| 2 | 3 | 5 | 15 |
| Jumlah |  |  | 23 |

Jadi dot product bukan perkalian biasa. Ia adalah "kali pasangan, lalu jumlah".

Di PCA, dot product dipakai untuk menghitung seberapa besar bayangan titik pada arah PC.

Contoh nanti:

$$
t = z p_1
$$

Di sini:

| Simbol | Arti |
|---|---|
| $t$ | score PCA, yaitu posisi window pada PC |
| $z$ | vektor fitur setelah scaling |
| $p_1$ | arah PC1 |

### 3.4 Matrix

**Matrix** adalah tabel angka.

Bahasa bayinya: matrix adalah tabel yang punya baris dan kolom. Kalau vektor adalah satu kotak berderet, matrix adalah banyak kotak yang disusun seperti tabel nilai sekolah.

Kalau kita punya 5 window dan 2 fitur, matrix data bisa seperti ini:

```text
Z = [ z_E(W1)  z_F(W1) ]
    [ z_E(W2)  z_F(W2) ]
    [ z_E(W3)  z_F(W3) ]
    [ z_E(W4)  z_F(W4) ]
    [ z_E(W5)  z_F(W5) ]
```

Baris berarti window. Kolom berarti fitur.

Cara membacanya:

| Bagian matrix | Arti di PCA |
|---|---|
| satu baris | satu window pompa |
| satu kolom | satu jenis fitur |
| satu sel | nilai satu fitur pada satu window |

Kalau matrix punya 5 baris dan 2 kolom, artinya ada 5 window dan 2 fitur.

Kita akan pakai simbol:

| Simbol | Arti |
|---|---|
| $Z$ | matrix semua z-score training |
| baris $Z$ | satu window |
| kolom $Z$ | satu fitur |

### 3.5 Transpose

**Transpose** artinya menukar baris menjadi kolom.

Bahasa bayinya: transpose seperti memutar tabel sehingga yang tadinya turun ke bawah menjadi jalan ke samping. Bukan mengubah angkanya, hanya mengubah posisi bacanya.

Kalau matrix kecil:

```text
A = [ 1  2 ]
    [ 3  4 ]
    [ 5  6 ]
```

Maka transpose-nya, ditulis $A^\top$, adalah:

```text
A^T = [ 1  3  5 ]
      [ 2  4  6 ]
```

Simbol $^\top$ berarti transpose.

Contoh membaca posisi:

| Angka | Posisi di $A$ | Posisi di $A^\top$ |
|---:|---|---|
| 1 | baris 1 kolom 1 | baris 1 kolom 1 |
| 2 | baris 1 kolom 2 | baris 2 kolom 1 |
| 5 | baris 3 kolom 1 | baris 1 kolom 3 |

Kenapa transpose dipakai? Karena covariance matrix bisa dihitung dengan mengalikan $Z^\top$ dan $Z$. Tetapi untuk contoh dua fitur, kita bisa menghitungnya pakai tabel kecil saja.

Kalau sudah paham bagian ini, kamu harus bisa:

| Kamu harus bisa | Contoh |
|---|---|
| membaca vektor | `[2.530, -0.351]` punya dua komponen |
| menghitung dot product | `[2,3] dot [4,5] = 23` |
| membaca matrix | baris adalah window, kolom adalah fitur |
| menjelaskan transpose | baris dan kolom ditukar |

## Bagian 4. Covariance dan correlation dengan bahasa sederhana

PCA belajar arah variasi dari hubungan antar fitur. Hubungan itu sering dibaca lewat covariance atau correlation.

### 4.1 Covariance

**Covariance** melihat apakah dua fitur bergerak bersama.

Misalkan ada dua fitur:

| Fitur | Arti |
|---|---|
| $E$ | energi band getaran |
| $F$ | frekuensi dominan |

Kalau $E$ naik dan $F$ juga naik, covariance cenderung positif.

Kalau $E$ naik tetapi $F$ turun, covariance cenderung negatif.

Kalau $E$ dan $F$ tidak punya pola bersama, covariance mendekati nol.

Rumus covariance sample untuk dua fitur $A$ dan `B`:

$$
\mathrm{cov}(A,B)=\frac{\sum_{i=1}^{n}(A_i-\mu_A)(B_i-\mu_B)}{n-1}
$$

Sebelum rumus itu dipakai, arti simbolnya:

| Simbol | Arti |
|---|---|
| `A_i` | nilai fitur A pada window ke-i |
| `B_i` | nilai fitur B pada window ke-i |
| `mu_A` | mean fitur A |
| `mu_B` | mean fitur B |
| $n$ | jumlah window |
| `cov(A, B)` | covariance antara A dan B |

Bahasa bayinya: lihat selisih A dari rata-ratanya, lihat selisih B dari rata-ratanya, kalikan. Kalau sering sama-sama positif atau sama-sama negatif, hasilnya positif.

Tabel tanda covariance:

| Selisih A | Selisih B | Hasil kali | Makna |
|---:|---:|---:|---|
| positif | positif | positif | A tinggi, B juga tinggi |
| negatif | negatif | positif | A rendah, B juga rendah |
| positif | negatif | negatif | A tinggi, B rendah |
| negatif | positif | negatif | A rendah, B tinggi |

Kalau hasil positif lebih sering menang, covariance positif. Kalau hasil negatif lebih sering menang, covariance negatif. Kalau campur aduk seimbang, covariance dekat nol.

### 4.2 Correlation

**Correlation** mirip covariance, tetapi sudah dinormalisasi. Nilainya biasanya berada antara `-1` dan `1`.

| Nilai correlation | Arti |
|---:|---|
| mendekati 1 | dua fitur naik turun bersama |
| mendekati -1 | satu naik saat yang lain turun |
| mendekati 0 | tidak ada hubungan linear yang jelas |

Tabel tanda yang lebih pelan:

| Correlation | Cerita titik | Rasa pompa |
|---:|---|---|
| `r > 0` | titik cenderung dari kiri bawah ke kanan atas | energi naik, frekuensi ikut naik |
| `r < 0` | titik cenderung dari kiri atas ke kanan bawah | energi naik, frekuensi turun |
| `r sekitar 0` | titik menyebar tanpa arah garis jelas | dua fitur tidak bergerak bersama secara rapi |

Kalau data sudah diubah menjadi z-score, covariance antar kolom z-score sama dengan correlation. Ini enak, karena matrix PCA jadi lebih mudah dibaca.

### 4.3 Matrix covariance atau correlation

Untuk dua fitur yang sudah di-z-score, matrix correlation sering berbentuk:

$$
S=\begin{bmatrix}1 & r \\ r & 1\end{bmatrix}
$$

Di sini:

| Simbol | Arti |
|---|---|
| $S$ | matrix covariance atau correlation |
| `1` | correlation fitur dengan dirinya sendiri |
| $r$ | correlation antara dua fitur berbeda |

Kalau `r = 0.971`, artinya dua fitur sangat kuat naik turun bersama.

Cara membaca matrix $S$:

| Bagian $S$ | Nilai | Arti |
|---|---:|---|
| kiri atas | 1 | fitur E dibandingkan dengan E sendiri |
| kanan bawah | 1 | fitur F dibandingkan dengan F sendiri |
| kanan atas | $r$ | hubungan E dengan F |
| kiri bawah | $r$ | hubungan F dengan E, nilainya sama |

Jadi $S$ adalah tabel hubungan antar fitur. PCA membaca tabel ini untuk mencari arah garis normal.

Kalau sudah paham bagian ini, kamu harus bisa:

| Kamu harus bisa | Penjelasan |
|---|---|
| menjelaskan covariance positif | dua fitur cenderung bergerak bersama |
| menjelaskan covariance negatif | satu fitur naik saat fitur lain turun |
| membaca $r$ | $r$ besar positif berarti hubungan normal kuat |
| membaca matrix $S$ | diagonal 1, luar diagonal correlation antar fitur |

## Bagian 5. Dasar fitur spektral dari nol

PCA pada proyek ini tidak langsung membaca deret sensor mentah. PCA membaca ringkasan dari deret sensor. Ringkasan itu kita sebut **fitur spektral**.

Kita mulai dari cerita pompa.

Pompa yang sehat punya irama. Getarannya naik turun dengan pola tertentu. Arus listriknya juga punya kebiasaan. Tekanan dan flow ikut bergerak sesuai ritme kerja pompa. Kalau ada masalah, ritme itu bisa berubah. Ada getaran yang menjadi lebih kuat. Ada frekuensi yang muncul lebih tajam. Ada energi yang pindah ke band tertentu.

Fitur spektral adalah cara mengubah irama itu menjadi angka kecil yang bisa masuk tabel PCA.

Alur besarnya:

```text
sensor time-series
potong menjadi window
lihat ritme frekuensi di window itu
ringkas menjadi fitur spektral
satu window menjadi satu baris fitur untuk PCA
```

![Ilustrasi window sensor menjadi fitur spektral](assets/pca-t2-q-manual/spectral-features.png)

Gambar di atas menunjukkan ide utamanya. Di kiri ada sinyal dalam waktu. Di kanan ada batang-batang frekuensi. Dari batang frekuensi itu kita mengambil angka ringkasan seperti energy, frekuensi dominan, centroid, atau band energy.

### 5.1 Time-series

**Time-series** adalah data yang datang berurutan terhadap waktu.

Contoh getaran pompa:

| Waktu | Nilai getaran |
|---:|---:|
| 0 detik | 1.0 |
| 1 detik | 1.2 |
| 2 detik | 0.8 |
| 3 detik | 1.1 |

Urutan penting. Nilai pada detik 1 datang setelah detik 0. Nilai pada detik 2 datang setelah detik 1.

Bahasa bayinya: time-series seperti rekaman suara pompa. Kalau urutannya diacak, iramanya rusak.

### 5.2 Sample

**Sample** adalah satu titik pengukuran.

Kalau sensor membaca getaran setiap detik, maka tiap bacaan adalah satu sample.

Kita pakai simbol $N$ untuk jumlah sample dalam satu window.

Contoh window kecil:

```text
x[0], x[1], x[2], x[3]
```

Di sini:

| Simbol | Arti |
|---|---|
| $x$ | sinyal dalam satu window |
| `x[0]` | sample pertama |
| `x[1]` | sample kedua |
| $N$ | jumlah sample dalam window |

Kalau ada 4 sample, maka `N = 4`.

Jangan tertukar:

| Simbol | Arti |
|---|---|
| $n$ kecil | jumlah window training |
| $N$ besar | jumlah sample di dalam satu window |

### 5.3 Sampling rate $f_s$

**Sampling rate** adalah berapa kali sensor membaca data per detik.

Kita pakai simbol $f_s$ untuk sampling rate.

Contoh:

| $f_s$ | Arti |
|---:|---|
| 1 Hz | 1 sample per detik |
| 10 Hz | 10 sample per detik |
| 100 Hz | 100 sample per detik |

`Hz` berarti per detik.

Kalau `fs = 10 Hz`, sensor mengambil 10 sample dalam 1 detik.

Bahasa bayinya: $f_s$ adalah seberapa rajin sensor memotret kondisi pompa. `fs = 10 Hz` berarti sensor memotret 10 kali setiap detik.

### 5.4 Window dan durasi window

**Window** adalah potongan pendek dari time-series.

Misalnya sistem membaca 60 sample terakhir dari sensor pompa, lalu menganggapnya sebagai satu paket kecil untuk dianalisis.

Bahasa bayinya: window adalah satu potong roti dari roti panjang time-series.

Contoh:

```text
time-series panjang:  x0, x1, x2, x3, x4, x5, x6, x7, ...
window pertama:       x0, x1, x2, x3
window kedua:             x1, x2, x3, x4
window ketiga:                x2, x3, x4, x5
```

Durasi window dihitung dari jumlah sample dan sampling rate:

$$
\text{durasi window}=\frac{N}{f_s}
$$

Contoh:

$$
N=60\ \text{sample},\quad f_s=1\ \mathrm{Hz}
$$

$$
\text{durasi window}=\frac{60}{1}=60\ \text{detik}
$$

Contoh lain:

$$
N=100\ \text{sample},\quad f_s=10\ \mathrm{Hz}
$$

$$
\text{durasi window}=\frac{100}{10}=10\ \text{detik}
$$

Di proyek PDAM Pump Sentinel, window dipakai karena fault pompa sering terlihat dari pola beberapa detik atau beberapa puluh detik, bukan dari satu titik sensor saja.

### 5.5 Frekuensi sebagai irama pompa

**Frekuensi** berarti seberapa cepat sinyal berulang.

Kalau getaran naik turun satu kali per detik, frekuensinya `1 Hz`. Kalau naik turun lima kali per detik, frekuensinya `5 Hz`.

Bayangkan kamu mendengar suara pompa:

| Rasa suara atau getaran | Rasa frekuensi |
|---|---|
| pelan naik turun | frekuensi rendah |
| cepat bergetar | frekuensi tinggi |
| ada dengung kuat di satu nada | ada frekuensi dominan |

Pompa punya ritme normal. Fault seperti imbalance, cavitation, atau gangguan aliran bisa mengubah energi pada frekuensi tertentu. Karena itu fitur spektral berguna.

### 5.6 FFT sebagai ide, bukan hitungan besar

FFT adalah cara cepat untuk melihat isi frekuensi dari satu window sinyal.

Untuk belajar manual, kita tidak perlu menghitung FFT besar dari nol. Yang perlu dipahami adalah pertanyaan yang dijawab FFT.

```text
window sinyal waktu -> FFT -> daftar kekuatan pada beberapa frekuensi
```

FFT menjawab:

```text
di frekuensi mana irama pompa paling kuat?
```

Analogi pompa:

| Di dunia pompa | Di dunia FFT |
|---|---|
| suara pompa punya beberapa nada | sinyal punya beberapa frekuensi |
| satu nada terdengar paling keras | satu bin punya magnitude terbesar |
| nada rendah dan tinggi bisa berubah | energi band rendah dan tinggi bisa berubah |

Hasil FFT bisa berupa bilangan kompleks. Untuk fitur sederhana, kita sering memakai **magnitude**. Magnitude adalah besar kekuatan, tanpa arah kompleksnya.

### 5.7 Magnitude FFT

Kita pakai simbol $M[k]$ untuk magnitude FFT pada bin ke-$k$.

Sebelum memakainya, arti simbolnya:

| Simbol | Arti |
|---|---|
| $k$ | nomor bin frekuensi |
| $M[k]$ | besar atau kekuatan sinyal pada bin ke-k |
| $N$ | jumlah sample dalam window |
| $f_s$ | sampling rate |

Magnitude adalah angka non-negatif. Kalau $M[2]$ besar, berarti frekuensi pada bin 2 kuat.

Bahasa bayinya: bayangkan spektrum sebagai deretan gelas. Setiap gelas adalah frekuensi. Isi gelas adalah magnitude. Gelas yang paling penuh berarti frekuensi itu paling kuat.

### 5.8 Frequency bin

**Frequency bin** adalah kotak frekuensi hasil FFT.

Rumus frekuensi untuk bin ke-$k$:

$$
f_k = \frac{k f_s}{N}
$$

Sebelum rumus dipakai:

| Simbol | Arti |
|---|---|
| $f_k$ | frekuensi untuk bin ke-k |
| $k$ | nomor bin |
| $f_s$ | sampling rate |
| $N$ | jumlah sample dalam window |

Contoh kecil:

$$
f_s=8\ \mathrm{Hz},\quad N=8\ \text{sample}
$$

Maka jarak antar bin:

$$
\frac{f_s}{N}=\frac{8}{8}=1\ \mathrm{Hz}
$$

Frekuensi bin:

| $k$ | $f_k=kf_s/N$ |
|---:|---:|
| 0 | 0 Hz |
| 1 | 1 Hz |
| 2 | 2 Hz |
| 3 | 3 Hz |
| 4 | 4 Hz |

Bin $k=0$ disebut **DC bin**. DC bin sering menggambarkan rata-rata atau offset sinyal. Kalau kita ingin fokus pada osilasi getaran, DC bin sering diabaikan.

Bahasa bayinya: DC bin bukan irama naik turun. Ia lebih mirip posisi dasar sinyal. Untuk getaran, kita sering lebih tertarik pada irama, jadi DC bisa disisihkan.

### 5.9 Energy

**Energy** dalam fitur spektral biasanya dihitung dari magnitude yang dikuadratkan lalu dijumlahkan.

Untuk satu band frekuensi:

$$
E_{\text{band}}=\sum_{k\in\text{band}}M[k]^2
$$

Sebelum rumus dipakai:

| Simbol | Arti |
|---|---|
| $E_{\text{band}}$ | energi pada band frekuensi tertentu |
| $M[k]$ | magnitude FFT pada bin ke-k |
| `M[k]^2` | magnitude kuadrat |
| `k di dalam band` | bin yang dipilih untuk band itu |

Contoh:

| $k$ | Frekuensi | $M[k]$ |
|---:|---:|---:|
| 1 | 1 Hz | 2 |
| 2 | 2 Hz | 3 |
| 3 | 3 Hz | 1 |

Jika band yang kita pilih adalah 1 Hz sampai 3 Hz, energinya:

$$
E_{\text{band}}=2^2+3^2+1^2=4+9+1=14
$$

Bahasa pompa: energi band memberi tahu seberapa kuat getaran pompa pada rentang irama tertentu.

### 5.10 Dominant frequency

**Dominant frequency** adalah frekuensi dengan magnitude terbesar.

Kita tulis $f_{\text{dom}}$ untuk dominant frequency.

Contoh:

| $k$ | Frekuensi | $M[k]$ |
|---:|---:|---:|
| 1 | 1 Hz | 2 |
| 2 | 2 Hz | 3 |
| 3 | 3 Hz | 1 |

Magnitude terbesar adalah `3` pada $k=2$, frekuensinya `2 Hz`.

Jadi:

$$
f_{\text{dom}}=2\ \mathrm{Hz}
$$

Bahasa bayinya: cari batang paling tinggi pada grafik spektrum, lalu baca frekuensinya. Pada pompa, ini seperti nada yang paling keras terdengar dalam window itu.

### 5.11 Spectral centroid

**Spectral centroid** adalah titik tengah berbobot dari spektrum. Ia seperti pusat massa frekuensi.

Kita tulis $f_{\text{centroid}}$.

Rumus:

$$
f_{\text{centroid}}=\frac{\sum_k f_k M[k]}{\sum_k M[k]}
$$

Sebelum rumus dipakai:

| Simbol | Arti |
|---|---|
| $f_{\text{centroid}}$ | pusat frekuensi berbobot magnitude |
| $f_k$ | frekuensi bin ke-k |
| $M[k]$ | magnitude pada bin ke-k |
| `sum(f_k * M[k])` | jumlah frekuensi dikali kekuatannya |
| `sum(M[k])` | total magnitude |

Contoh:

| $k$ | $f_k$ | $M[k]$ | $f_kM[k]$ |
|---:|---:|---:|---:|
| 1 | 1 | 2 | 2 |
| 2 | 2 | 3 | 6 |
| 3 | 3 | 1 | 3 |

Hitung:

$$
\sum_k f_kM[k]=2+6+3=11
$$

$$
\sum_k M[k]=2+3+1=6
$$

$$
f_{\text{centroid}}=\frac{11}{6}=1.833\ \mathrm{Hz}
$$

Centroid `1.833 Hz` berarti pusat massa spektrum berada dekat 2 Hz. Kalau energi banyak pindah ke frekuensi tinggi, centroid bisa ikut naik.

### 5.12 Band energy

**Band energy** adalah energi pada rentang frekuensi tertentu.

Misalnya kita punya bin:

| $k$ | $f_k$ | $M[k]$ |
|---:|---:|---:|
| 1 | 1 Hz | 2 |
| 2 | 2 Hz | 3 |
| 3 | 3 Hz | 1 |
| 4 | 4 Hz | 4 |

Kita ingin energi band rendah 1 sampai 2 Hz:

$$
E_{\text{low}}=M[1]^2+M[2]^2=2^2+3^2=4+9=13
$$

Energi band tinggi 3 sampai 4 Hz:

$$
E_{\text{high}}=M[3]^2+M[4]^2=1^2+4^2=1+16=17
$$

Fitur seperti ini membantu model melihat apakah energi berpindah ke band tertentu.

Bahasa pompa: jika kondisi sehat biasanya kuat di band rendah, lalu window baru tiba-tiba kuat di band tinggi, model perlu melihat perubahan itu.

### 5.13 Kenapa FFT disederhanakan untuk hitung tangan

FFT penuh untuk window nyata bisa punya 60, 128, 256, atau lebih banyak sample. Menghitung FFT penuh dengan tangan akan sangat panjang dan tidak membantu tujuan utama catatan ini.

Karena itu, untuk belajar manual kita menyederhanakan:

| Bagian nyata | Versi latihan tangan |
|---|---|
| FFT banyak sample | magnitude FFT sudah diberikan |
| banyak bin frekuensi | 3 sampai 5 bin saja |
| banyak fitur spektral | 2 fitur saja |
| banyak window | 5 window normal |

Yang penting bukan hafal hitung FFT besar. Yang penting paham bahwa window sensor diubah menjadi angka ringkas seperti energi band dan frekuensi dominan. Angka ringkas itulah yang masuk PCA.

Kalau sudah paham bagian ini, kamu harus bisa:

| Kamu harus bisa | Penjelasan |
|---|---|
| menjelaskan window | potongan pendek time-series |
| menjelaskan sampling rate | jumlah sample per detik |
| menghitung durasi window | `N / fs` |
| menghitung frequency bin | `f_k = k * fs / N` |
| menjelaskan DC bin | bin 0 Hz, sering berarti offset |
| menjelaskan magnitude FFT | kekuatan sinyal pada bin frekuensi |
| menghitung energy | jumlah magnitude kuadrat |
| mencari dominant frequency | frekuensi dengan magnitude terbesar |
| menghitung centroid kecil | jumlah $fM$ dibagi jumlah $M$ |

## Bagian 6. Dari fitur spektral ke data PCA

Setelah satu window sensor diubah menjadi fitur spektral, PCA tidak lagi melihat deret sample mentah. PCA melihat tabel fitur.

Satu baris PCA berarti satu window.

```text
satu window sensor -> beberapa fitur spektral -> satu baris di tabel PCA
```

Contoh fitur yang akan dipakai dalam latihan utama:

| Nama fitur | Simbol | Arti sederhana |
|---|---|---|
| Energi band getaran | $E$ | seberapa kuat energi pada band frekuensi yang dipilih |
| Frekuensi dominan | $F$ | frekuensi yang magnitude-nya paling besar |

Kita sengaja hanya memakai dua fitur agar bisa menggambar dan menghitung dengan tangan.

Dalam proyek nyata, fitur bisa lebih banyak. Tetapi logika PCA T2/Q tetap sama. PCA selalu menerima tabel: baris adalah window, kolom adalah fitur.

### 6.1 Contoh ekstraksi fitur kecil

Misalkan satu window punya hasil magnitude FFT seperti ini:

| $k$ | $f_k$ | $M[k]$ |
|---:|---:|---:|
| 0 | 0 Hz | 10 |
| 1 | 10 Hz | 2 |
| 2 | 20 Hz | 3 |
| 3 | 30 Hz | 1 |

Kita abaikan $k=0$ karena itu DC bin.

Energi band 10 sampai 30 Hz:

$$
E=2^2+3^2+1^2=4+9+1=14
$$

Frekuensi dominan:

$$
\text{magnitude terbesar di } k=2
$$

$$
f_2=20\ \mathrm{Hz}
$$

$$
F=20\ \mathrm{Hz}
$$

Jadi satu window berubah dari deret sensor menjadi dua angka:

$$
x=[E,F]=[14,20]
$$

Di sini $x$ berarti vektor fitur mentah satu window.

### 6.2 Dari banyak window menjadi tabel PCA

Kalau kita punya lima window normal, tiap window diubah dengan cara yang sama.

| Window | Ringkasan dari spektrum | Baris fitur untuk PCA |
|---:|---|---|
| W1 | hitung $E$ dan $F$ dari FFT W1 | `[E(W1), F(W1)]` |
| W2 | hitung $E$ dan $F$ dari FFT W2 | `[E(W2), F(W2)]` |
| W3 | hitung $E$ dan $F$ dari FFT W3 | `[E(W3), F(W3)]` |
| W4 | hitung $E$ dan $F$ dari FFT W4 | `[E(W4), F(W4)]` |
| W5 | hitung $E$ dan $F$ dari FFT W5 | `[E(W5), F(W5)]` |

Tabel PCA mentahnya menjadi:

```text
X = [ E(W1)  F(W1) ]
    [ E(W2)  F(W2) ]
    [ E(W3)  F(W3) ]
    [ E(W4)  F(W4) ]
    [ E(W5)  F(W5) ]
```

Bahasa bayinya: FFT mengubah satu potong irama pompa menjadi beberapa angka. PCA mengumpulkan angka-angka itu dari banyak potong irama, lalu belajar pola normalnya.

### 6.3 Kenapa satu window menjadi satu baris

PCA membandingkan window dengan window lain. Karena itu satu window harus menjadi satu benda yang utuh.

| Bentuk | Arti |
|---|---|
| satu window sensor | satu kejadian pendek pada pompa |
| satu vektor fitur | ringkasan kejadian pendek itu |
| satu baris PCA | posisi kejadian itu di ruang fitur |

Jika W1 punya `E = 8` dan `F = 29`, maka baris W1 adalah:

```text
W1 = [8, 29]
```

Jika W5 punya `E = 12` dan `F = 32`, maka baris W5 adalah:

```text
W5 = [12, 32]
```

Nanti PCA melihat bahwa ketika $E$ naik dari 8 ke 12, $F$ juga cenderung naik dari 29 ke 32. Itulah pola normal yang akan menjadi garis PC1.

Kalau sudah paham bagian ini, kamu harus bisa menjelaskan bahwa PCA tidak menerima FFT mentah dalam latihan ini. PCA menerima fitur ringkas seperti $E$ dan $F$, satu window per baris.

## Bagian 7. Eigenvector dan eigenvalue dengan cara lembut

Bagian ini sering membuat orang berhenti. Kita buat sederhana.

### 7.1 Gagasan PCA sebagai mencari arah garis

Bayangkan titik normal dua fitur digambar di kertas:

```text
sumbu kiri-kanan: energi E
sumbu bawah-atas: frekuensi F
```

Kalau energi naik dan frekuensi juga naik, titik-titik normal akan membentuk awan miring dari kiri bawah ke kanan atas.

Setiap window adalah satu titik.

| Window | Setelah jadi fitur | Setelah digambar |
|---|---|---|
| W1 | `[E, F]` | satu titik |
| W2 | `[E, F]` | satu titik lain |
| W3 | `[E, F]` | satu titik lain lagi |

Bahasa bayinya: window bukan lagi deret panjang. Window sudah menjadi titik kecil di kertas. Kalau ada 5 window normal, ada 5 titik normal.

PCA bertanya:

```text
arah garis mana yang paling menjelaskan sebaran titik normal?
```

Arah garis itu adalah **eigenvector** utama. Dalam PCA, eigenvector utama disebut **loading PC1**.

Rumus pendek eigen yang sering muncul adalah:

$$
S p = \lambda p
$$

Bahasa bayinya: matrix hubungan $S$ bertemu arah $p$, lalu hasilnya tetap searah, hanya panjangnya berubah sebesar $\lambda$.

Cerita geometri PCA:

| Istilah | Bahasa gambar | Bahasa bayi |
|---|---|---|
| titik window | satu titik di kertas | satu kejadian pompa |
| PC1 | jalan utama atau garis utama | arah normal paling ramai |
| eigenvector | arah jalan | panah yang menunjukkan jalan menghadap ke mana |
| eigenvalue | sebaran di jalan itu | seberapa panjang titik normal menyebar di jalan |
| projection | bayangan titik ke jalan | jatuhkan titik ke jalan normal |
| reconstruction | gambar ulang dari bayangan | titik dibuat ulang dari posisi di jalan |
| residual | sisa jarak dari gambar ulang | bagian yang tidak cocok dengan jalan |

Analogi jalan:

```text
titik normal = mobil yang biasanya lewat dekat jalan utama
PC1          = jalan utama normal
T2           = mobil terlalu jauh tidak di sepanjang jalan itu?
Q            = mobil keluar tidak dari jalan itu?
```

Pertanyaan T2 dan Q harus diingat seperti ini:

| Skor | Pertanyaan bayi |
|---|---|
| T2 | jauh tidak di jalan normal? |
| Q/SPE | keluar tidak dari jalan normal? |

### 7.2 Eigenvector

**Eigenvector** adalah arah khusus yang tetap menjadi arah setelah matrix hubungan data bekerja padanya.

Untuk belajar manual, cukup pegang arti praktisnya:

```text
eigenvector PCA = arah pola utama data normal
```

Kalau PC1 adalah jalan utama, eigenvector adalah arah jalan itu. Ia memberi tahu: untuk bergerak di jalan normal, fitur mana naik dan fitur mana turun.

Pada contoh dua fitur:

| Bentuk arah | Arti |
|---|---|
| `[0.707, 0.707]` | bergerak ke energi tinggi dan frekuensi tinggi bersama |
| `[0.707, -0.707]` | bergerak ke energi tinggi tetapi frekuensi rendah |

Tanda plus atau minus penting. Dua plus berarti dua fitur berjalan searah. Satu plus dan satu minus berarti dua fitur berlawanan arah.

Kita akan pakai simbol $p_1$ untuk eigenvector PC1 atau loading PC1.

Di contoh dua fitur yang bergerak naik bersama, arah PC1 kira-kira:

$$
p_1=[0.707,0.707]
$$

Artinya PC1 memberi bobot sama ke energi dan frekuensi setelah scaling.

Kenapa `0.707`? Karena:

$$
0.707\ \text{dibulatkan dari}\ \frac{1}{\sqrt{2}}
$$

$$
\frac{1}{\sqrt{2}}=0.707106\ldots
$$

Vektor `[0.707, 0.707]` panjangnya 1:

$$
\sqrt{0.707^2+0.707^2}=\sqrt{0.500+0.500}=\sqrt{1.000}=1.000
$$

Di hitungan tangan kita menulis `0.707` agar tabel tidak terlalu panjang. Di kode proyek, simpan angka lebih presisi seperti `0.70710678`, bukan hanya tiga desimal.

### 7.3 Eigenvalue

**Eigenvalue** adalah angka yang memberi tahu seberapa besar variasi data pada arah eigenvector.

Kalau eigenvector adalah arah jalan, eigenvalue adalah ukuran ramainya penyebaran titik di jalan itu.

| Eigenvalue | Bahasa bayi |
|---:|---|
| besar | banyak variasi normal terjadi di arah ini |
| kecil | sedikit variasi normal terjadi di arah ini |

PCA biasanya menyimpan PC dengan eigenvalue besar karena arah itu menjelaskan kebiasaan normal paling banyak.

Kita akan pakai simbol $\lambda_1$ untuk eigenvalue PC1.

Arti praktisnya:

```text
lambda_1 besar berarti PC1 menjelaskan variasi normal yang besar
lambda_1 kecil berarti PC1 menjelaskan variasi normal yang kecil
```

Untuk matrix correlation dua fitur:

$$
S=\begin{bmatrix}1 & r \\ r & 1\end{bmatrix}
$$

Jika dua fitur berkorelasi positif kuat, PC1 punya:

$$
p_1=[0.707,0.707]
$$

$$
\lambda_1=1+r
$$

PC2, yang tegak lurus PC1, punya:

$$
p_2=[0.707,-0.707]
$$

$$
\lambda_2=1-r
$$

Di latihan utama, kita hanya menyimpan PC1. PC2 tidak dipakai untuk rekonstruksi. Justru karena PC2 tidak disimpan, Q/SPE bisa melihat sisa error.

Untuk dua fitur z-score, total variance kira-kira `2`, karena tiap fitur punya variance `1`.

Jika nanti `r = 0.971`, maka:

$$
\lambda_1=1+0.971=1.971
$$

$$
\text{bagian variance PC1}=\frac{\lambda_1}{2}=\frac{1.971}{2}=0.9855
$$

Jadi bagian PC1 sekitar 98.5 persen.

Artinya, dalam contoh dua fitur ini, PC1 menangkap hampir semua variasi normal. Sisa kecilnya berada di arah PC2.

### 7.4 Kasus mudah dua fitur

Ada tiga kasus mudah:

| Pola dua fitur | Arah PC1 kira-kira | Arti |
|---|---|---|
| dua fitur naik bersama | `[0.707, 0.707]` | garis kiri bawah ke kanan atas |
| satu naik, satu turun | `[0.707, -0.707]` | garis kiri atas ke kanan bawah |
| tidak berkorelasi | arah tidak dominan kuat | PCA tidak punya satu garis jelas |

Untuk contoh kita, energi dan frekuensi naik bersama. Jadi PC1 kita pakai:

$$
p_1=[0.707,0.707]
$$

### 7.5 Projection, reconstruction, dan residual dalam satu cerita

Nanti kita akan menghitung tiga benda ini:

```text
t      = projection atau bayangan titik ke PC1
z_hat  = reconstruction atau gambar ulang dari bayangan
e      = residual atau sisa yang tidak tergambar
```

Bayangkan titik window adalah bola kecil di dekat jalan PC1.

1. Projection menjatuhkan bayangan bola ke jalan.
2. Score $t$ memberi tahu posisi bayangan itu di jalan.
3. Reconstruction menggambar ulang bola dengan hanya memakai posisi bayangan di jalan.
4. Residual adalah jarak dari bola asli ke gambar ulangnya.

Kalau residual kecil, titik asli dekat jalan. Kalau residual besar, titik asli keluar dari jalan.

Inilah kenapa T2 dan Q berbeda:

| Situasi | T2 | Q |
|---|---|---|
| titik jauh di ujung jalan tetapi masih di jalan | tinggi | rendah |
| titik dekat tengah tetapi melenceng keluar jalan | bisa rendah | tinggi |
| titik jauh dan melenceng | tinggi | tinggi |

Untuk contoh $X_{\text{new}}$, energi tinggi tetapi frekuensi tidak ikut tinggi. Titiknya tidak mengikuti jalan energi dan frekuensi naik bersama. Jadi Q yang akan berteriak keras.

![Ilustrasi geometri PCA, projection, reconstruction, dan residual](assets/pca-t2-q-manual/pca-geometry.png)

Pada gambar ini, titik biru adalah data normal. Garis miring adalah PC1, yaitu jalan normal. Titik merah adalah window baru. Bayangannya jatuh ke garis PC1, lalu PCA menggambar ulang titik dari bayangan itu. Jarak dari gambar ulang ke titik asli adalah residual. Residual besar membuat Q besar.

Kalau sudah paham bagian ini, kamu harus bisa:

| Kamu harus bisa | Penjelasan |
|---|---|
| menjelaskan eigenvector | arah pola utama data |
| menjelaskan eigenvalue | besar variasi pada arah itu |
| menjelaskan projection | bayangan titik ke garis PC1 |
| menjelaskan reconstruction | menggambar ulang titik dari bayangan PC1 |
| menjelaskan residual | sisa yang tidak bisa digambar ulang |
| mengenali kasus dua fitur naik bersama | PC1 kira-kira `[0.707, 0.707]` |
| menjelaskan kenapa 1 PC disimpan | agar latihan sederhana dan Q/SPE masih terlihat |

## Bagian 8. Notasi lengkap sebelum contoh utama

Sebelum masuk contoh angka panjang, kita rapikan simbol.

| Simbol | Dibaca | Arti |
|---|---|---|
| $n$ | en | jumlah window training normal |
| $p$ | pe | jumlah fitur |
| $N$ | en besar | jumlah sample dalam satu window sensor |
| $f_s$ | ef es | sampling rate, sample per detik |
| $k$ | ka | nomor frequency bin FFT |
| $f_k$ | ef ka | frekuensi pada bin ke-k |
| $M[k]$ | em ka | magnitude FFT pada bin ke-k |
| $E$ | e | fitur energi band |
| $F$ | ef | fitur frekuensi dominan |
| $x$ | eks | fitur mentah satu window |
| $\mu$ | miu | mean fitur dari training normal |
| $s$ | es | standar deviasi fitur dari training normal |
| $z$ | zet | fitur setelah z-score |
| $Z$ | zet besar | matrix z-score training |
| $S$ | es besar | matrix covariance atau correlation |
| $r$ | er | correlation antara dua fitur |
| $p_1$ | pe satu | loading vector PC1 |
| $\lambda_1$ | lambda satu | eigenvalue PC1 |
| $t$ | te | score PCA pada PC1 |
| $\hat z$ | zet topi | rekonstruksi z dari PC1 |
| $e$ | e kecil | residual, yaitu `z - z_hat` |
| $T^2$ | te dua | Hotelling T2 |
| $Q$ | kiu | Q statistic atau SPE |
| $\mathrm{SPE}$ | es pe e | squared prediction error |
| $\text{threshold}_{T^2}$ | ambang T2 | batas T2 normal |
| $\text{threshold}_Q$ | ambang Q | batas Q normal |

Catatan kecil: $Q$ dan $\mathrm{SPE}$ di catatan ini berarti hal yang sama, yaitu jumlah kuadrat residual.

Ringkasan makna simbol yang sering tertukar:

| Simbol | Jangan tertukar dengan | Cara mengingat |
|---|---|---|
| $n$ | $N$ | $n$ kecil untuk banyak window training |
| $N$ | $n$ | $N$ besar untuk banyak sample dalam window |
| $x$ | $z$ | $x$ masih mentah, $z$ sudah di-scaling |
| $t$ | $T^2$ | $t$ posisi di PC, $T^2$ skor jarak dari $t$ |
| $e$ | $E$ | $e$ kecil residual, $E$ besar fitur energi |
| $Q$ | `q` lain | $Q$ di sini sama dengan SPE |

Kalau sudah paham bagian ini, kamu harus bisa melihat rumus tanpa panik karena setiap simbol sudah punya nama.

## Bagian 9. Data latihan utama

Kita punya 5 window normal untuk training. Setiap window sudah diubah menjadi 2 fitur spektral.

| Window normal | $E$, energi band | $F$, frekuensi dominan |
|---:|---:|---:|
| W1 | 8 | 29 |
| W2 | 9 | 30 |
| W3 | 10 | 30 |
| W4 | 11 | 31 |
| W5 | 12 | 32 |

Kita lihat pola normalnya:

| Dari | Ke | Pola |
|---|---|---|
| W1 ke W5 | $E$ naik dari 8 ke 12 | energi makin besar |
| W1 ke W5 | $F$ naik dari 29 ke 32 | frekuensi dominan ikut naik |

Jadi kebiasaan normalnya kira-kira:

```text
kalau E naik, F juga cenderung naik
```

PCA akan belajar pola itu sebagai PC1.

## Bagian 10. Hitung mean dan standar deviasi training

Kita hitung statistik training untuk setiap fitur secara terpisah.

### 10.0 Kebijakan pembulatan angka di contoh ini

Supaya hitungan tangan tidak terlalu penuh, catatan ini memakai pembulatan.

| Jenis angka | Cara tulis di catatan |
|---|---|
| mean sederhana | tulis sesuai hasil, misalnya `10` atau `30.4` |
| standar deviasi | biasanya 3 desimal |
| z-score | 3 desimal |
| loading $p_1$ | 3 desimal, `0.707` |
| score, T2, Q | 3 desimal untuk tabel utama |
| residual kuadrat yang sangat kecil | kadang perlu 6 desimal agar tidak terlihat hilang |

Peringatan penting: pembulatan ini hanya untuk belajar manual. Kode proyek harus menyimpan presisi lebih banyak. Kalau kode hanya menyimpan tiga desimal, T2 dan Q bisa sedikit bergeser, terutama saat skor dekat threshold.

Jadi kalau kamu menghitung ulang dan mendapat angka beda tipis, jangan langsung panik. Cek dulu apakah bedanya berasal dari pembulatan.

### 10.1 Hitung mean $E$

Data $E$:

```text
8, 9, 10, 11, 12
```

Jumlah:

$$
\mathrm{sum}_E=8+9+10+11+12=50
$$

Cara raw jumlahnya:

```text
8 + 9 = 17
17 + 10 = 27
27 + 11 = 38
38 + 12 = 50

jadi sum_E = 50
```

Jumlah window:

$$
n=5
$$

Mean:

$$
\mu_E=\frac{\mathrm{sum}_E}{n}=\frac{50}{5}=10
$$

Cara raw pembagiannya:

```text
cari angka x 5 yang jadi 50:

  9  x 5 = 45  (kurang)
  11 x 5 = 55  (lebih)
  10 x 5 = 50  (pas)

jadi 50 / 5 = 10
```

### 10.2 Hitung standar deviasi $E$

Tabel deviation:

| Window | $E$ | $\mu_E$ | $d_E=E-\mu_E$ | $d_E^2$ |
|---:|---:|---:|---:|---:|
| W1 | 8 | 10 | -2 | 4 |
| W2 | 9 | 10 | -1 | 1 |
| W3 | 10 | 10 | 0 | 0 |
| W4 | 11 | 10 | 1 | 1 |
| W5 | 12 | 10 | 2 | 4 |

Jumlah kuadrat deviation:

$$
\sum d_E^2=4+1+0+1+4=10
$$

Cara raw jumlahnya:

```text
4 + 1 = 5
5 + 0 = 5
5 + 1 = 6
6 + 4 = 10

jadi sum d_E^2 = 10
```

Sample variance:

$$
s_E^2=\frac{\sum d_E^2}{n-1}=\frac{10}{4}=2.5
$$

Cara raw pembagiannya:

```text
cari angka x 4 yang jadi 10:

  2   x 4 = 8   (kurang)
  3   x 4 = 12  (lebih)
  2.5 x 4 = 10  (pas)

jadi 10 / 4 = 2.5
```

Standar deviasi:

$$
s_E=\sqrt{2.5}=1.581
$$

Cara meraba akarnya:

```text
cari angka yang kalau dikuadratkan mendekati 2.5:

1.5^2   = 2.250  (kurang)
1.6^2   = 2.560  (lebih)
1.58^2  = 2.496  (kurang dikit)
1.581^2 = 2.500  (pas)

jadi sqrt(2.5) = 1.581
```

### 10.3 Hitung mean $F$

Data $F$:

```text
29, 30, 30, 31, 32
```

Jumlah:

$$
\mathrm{sum}_F=29+30+30+31+32=152
$$

Cara raw jumlahnya:

```text
29 + 30 = 59
59 + 30 = 89
89 + 31 = 120
120 + 32 = 152

jadi sum_F = 152
```

Mean:

$$
\mu_F=\frac{\mathrm{sum}_F}{n}=\frac{152}{5}=30.4
$$

Cara raw pembagiannya:

```text
cari angka x 5 yang jadi 152:

  30   x 5 = 150  (kurang)
  31   x 5 = 155  (lebih)
  30.4 x 5 = 152  (pas)

jadi 152 / 5 = 30.4
```

### 10.4 Hitung standar deviasi $F$

Tabel deviation:

| Window | $F$ | $\mu_F$ | $d_F=F-\mu_F$ | $d_F^2$ |
|---:|---:|---:|---:|---:|
| W1 | 29 | 30.4 | -1.4 | 1.96 |
| W2 | 30 | 30.4 | -0.4 | 0.16 |
| W3 | 30 | 30.4 | -0.4 | 0.16 |
| W4 | 31 | 30.4 | 0.6 | 0.36 |
| W5 | 32 | 30.4 | 1.6 | 2.56 |

Cara raw kuadratnya:

```text
(-1.4)^2 = 1.4 x 1.4
          = 1.4 x 1 + 1.4 x 0.4
          = 1.400 + 0.560
          = 1.960 -> 1.96

(-0.4)^2 = 0.4 x 0.4
          = 0.160 -> 0.16

(-0.4)^2 = 0.4 x 0.4
          = 0.160 -> 0.16

0.6^2    = 0.6 x 0.6
          = 0.360 -> 0.36

1.6^2    = 1.6 x 1.6
          = 1.6 x 1 + 1.6 x 0.6
          = 1.600 + 0.960
          = 2.560 -> 2.56
```

Jumlah kuadrat deviation:

$$
\sum d_F^2=1.96+0.16+0.16+0.36+2.56=5.20
$$

Cara raw jumlahnya:

```text
1.96 + 0.16 = 2.12
2.12 + 0.16 = 2.28
2.28 + 0.36 = 2.64
2.64 + 2.56 = 5.20

jadi sum d_F^2 = 5.20
```

Sample variance:

$$
s_F^2=\frac{\sum d_F^2}{n-1}=\frac{5.20}{4}=1.30
$$

Cara raw pembagiannya:

```text
cari angka x 4 yang jadi 5.20:

  1.2 x 4 = 4.800  (kurang)
  1.4 x 4 = 5.600  (lebih)
  1.3 x 4 = 5.200  (pas)

jadi 5.20 / 4 = 1.30
```

Standar deviasi:

$$
s_F=\sqrt{1.30}=1.140
$$

Cara meraba akarnya:

```text
cari angka yang kalau dikuadratkan mendekati 1.30:

1.1^2   = 1.210     (kurang)
1.2^2   = 1.440     (lebih)
1.14^2  = 1.300     (pas setelah dibulatkan)
1.140^2 = 1.299600  (pas untuk catatan 3 desimal)

jadi sqrt(1.30) = 1.140
```

### 10.5 Simpan statistik training

Statistik training yang harus disimpan:

| Fitur | Mean | Standar deviasi |
|---|---:|---:|
| E | 10 | 1.581 |
| F | 30.4 | 1.140 |

Angka ini nanti dipakai juga untuk window baru. Jangan dihitung ulang saat inference.

Kalau sudah paham bagian ini, kamu harus bisa:

| Kamu harus bisa | Hasil contoh |
|---|---:|
| mean E | 10 |
| standar deviasi E | 1.581 |
| mean F | 30.4 |
| standar deviasi F | 1.140 |

## Bagian 11. Ubah fitur mentah menjadi z-score

Rumus z-score untuk satu fitur:

$$
z_i = \frac{x_i - \mu}{s}
$$

Arti simbol:

| Simbol | Arti |
|---|---|
| $x$ | nilai fitur mentah |
| $\mu$ | mean fitur dari training normal |
| $s$ | standar deviasi fitur dari training normal |
| $z$ | nilai setelah scaling |

### 11.1 Z-score untuk E

Pakai:

$$
\mu_E=10,\quad s_E=1.581
$$

Hitung setiap window:

$$
z_E(W1)=\frac{8-10}{1.581}=\frac{-2}{1.581}=-1.265
$$

Cara raw W1:

```text
8 - 10 = -2

cari angka x 1.581 mendekati 2:
  1.2   x 1.581 = 1.897  (kurang)
  1.3   x 1.581 = 2.055  (lebih)
  1.26  x 1.581 = 1.992  (kurang sedikit)
  1.265 x 1.581 = 2.000  (pas)

jadi -2 / 1.581 = -1.265
```

$$
z_E(W2)=\frac{9-10}{1.581}=\frac{-1}{1.581}=-0.632
$$

Cara raw W2:

```text
9 - 10 = -1

cari angka x 1.581 mendekati 1:
  0.6    x 1.581 = 0.949  (kurang)
  0.7    x 1.581 = 1.107  (lebih)
  0.63   x 1.581 = 0.996  (kurang sedikit)
  0.6324 x 1.581 = 1.000  (pas untuk catatan)

0.6324 dibulatkan di tabel menjadi 0.632
jadi -1 / 1.581 = -0.632
```

$$
z_E(W3)=\frac{10-10}{1.581}=\frac{0}{1.581}=0.000
$$

Cara raw W3:

```text
10 - 10 = 0

0 dibagi angka apa pun tetap 0
jadi 0 / 1.581 = 0.000
```

$$
z_E(W4)=\frac{11-10}{1.581}=\frac{1}{1.581}=0.632
$$

Cara raw W4:

```text
11 - 10 = 1

cari angka x 1.581 mendekati 1:
  0.6    x 1.581 = 0.949  (kurang)
  0.7    x 1.581 = 1.107  (lebih)
  0.63   x 1.581 = 0.996  (kurang sedikit)
  0.6324 x 1.581 = 1.000  (pas untuk catatan)

0.6324 dibulatkan di tabel menjadi 0.632
jadi 1 / 1.581 = 0.632
```

$$
z_E(W5)=\frac{12-10}{1.581}=\frac{2}{1.581}=1.265
$$

Cara raw W5:

```text
12 - 10 = 2

cari angka x 1.581 mendekati 2:
  1.2   x 1.581 = 1.897  (kurang)
  1.3   x 1.581 = 2.055  (lebih)
  1.26  x 1.581 = 1.992  (kurang sedikit)
  1.265 x 1.581 = 2.000  (pas)

jadi 2 / 1.581 = 1.265
```

### 11.2 Z-score untuk F

Pakai:

$$
\mu_F=30.4,\quad s_F=1.140
$$

Hitung setiap window:

$$
z_F(W1)=\frac{29-30.4}{1.140}=\frac{-1.4}{1.140}=-1.228
$$

Cara raw W1:

```text
29 - 30.4 = -1.4

cari angka x 1.140 mendekati 1.4:
  1.2   x 1.140 = 1.368  (kurang)
  1.3   x 1.140 = 1.482  (lebih)
  1.23  x 1.140 = 1.402  (lebih sedikit)
  1.228 x 1.140 = 1.400  (pas)

jadi -1.4 / 1.140 = -1.228
```

$$
z_F(W2)=\frac{30-30.4}{1.140}=\frac{-0.4}{1.140}=-0.351
$$

Cara raw W2:

```text
30 - 30.4 = -0.4

cari angka x 1.140 mendekati 0.4:
  0.3   x 1.140 = 0.342  (kurang)
  0.4   x 1.140 = 0.456  (lebih)
  0.35  x 1.140 = 0.399  (kurang sedikit)
  0.351 x 1.140 = 0.400  (pas)

jadi -0.4 / 1.140 = -0.351
```

$$
z_F(W3)=\frac{30-30.4}{1.140}=\frac{-0.4}{1.140}=-0.351
$$

Cara raw W3:

```text
30 - 30.4 = -0.4

cari angka x 1.140 mendekati 0.4:
  0.3   x 1.140 = 0.342  (kurang)
  0.4   x 1.140 = 0.456  (lebih)
  0.35  x 1.140 = 0.399  (kurang sedikit)
  0.351 x 1.140 = 0.400  (pas)

jadi -0.4 / 1.140 = -0.351
```

$$
z_F(W4)=\frac{31-30.4}{1.140}=\frac{0.6}{1.140}=0.526
$$

Cara raw W4:

```text
31 - 30.4 = 0.6

cari angka x 1.140 mendekati 0.6:
  0.5   x 1.140 = 0.570  (kurang)
  0.6   x 1.140 = 0.684  (lebih)
  0.53  x 1.140 = 0.604  (lebih sedikit)
  0.526 x 1.140 = 0.600  (pas)

jadi 0.6 / 1.140 = 0.526
```

$$
z_F(W5)=\frac{32-30.4}{1.140}=\frac{1.6}{1.140}=1.403
$$

Cara raw W5:

```text
32 - 30.4 = 1.6

cari angka x 1.140 mendekati 1.6:
  1.3    x 1.140 = 1.482  (kurang)
  1.5    x 1.140 = 1.710  (lebih)
  1.4    x 1.140 = 1.596  (kurang sedikit)
  1.4034 x 1.140 = 1.600  (pas untuk catatan)

1.4034 dibulatkan di tabel menjadi 1.403
jadi 1.6 / 1.140 = 1.403
```

### 11.3 Tabel z-score training

| Window | $z_E$ | $z_F$ |
|---:|---:|---:|
| W1 | -1.265 | -1.228 |
| W2 | -0.632 | -0.351 |
| W3 | 0.000 | -0.351 |
| W4 | 0.632 | 0.526 |
| W5 | 1.265 | 1.403 |

Cara cek cepat:

| Cek | Harapan |
|---|---|
| rata-rata kolom z_E | mendekati 0 |
| rata-rata kolom z_F | mendekati 0 |
| sample variance z_E | mendekati 1 |
| sample variance z_F | mendekati 1 |

Kita pakai pembulatan tiga desimal. Karena dibulatkan, hasil cek tidak akan sempurna persis.

Contoh efek pembulatan:

```text
s_E asli kira-kira 1.581138...
di tabel ditulis 1.581
```

Jika memakai angka asli, z-score sedikit berbeda di digit keempat dan seterusnya. Untuk catatan tangan, itu boleh.

Kalau sudah paham bagian ini, kamu harus bisa menghitung z-score satu baris. Misalnya W1:

$$
E=8\ \text{menjadi}\ z_E=-1.265
$$

$$
F=29\ \text{menjadi}\ z_F=-1.228
$$

## Bagian 12. Hitung correlation matrix dari z-score

Karena data sudah di-z-score, kita bisa memakai correlation matrix.

Untuk dua fitur, bentuknya:

$$
S=\begin{bmatrix}1 & r \\ r & 1\end{bmatrix}
$$

Arti simbol:

| Simbol | Arti |
|---|---|
| $S$ | matrix correlation |
| $r$ | correlation antara z_E dan z_F |
| `1` | correlation fitur dengan dirinya sendiri |

Rumus $r$:

$$
r=\frac{\sum_{i=1}^{n}z_{E,i}z_{F,i}}{n-1}
$$

Arti simbol:

| Simbol | Arti |
|---|---|
| `z_E_i` | z-score E pada window ke-i |
| `z_F_i` | z-score F pada window ke-i |
| $n$ | jumlah window training |
| $r$ | correlation antara dua fitur |

### 12.1 Hitung perkalian tiap baris

| Window | $z_E$ | $z_F$ | z_E * z_F |
|---:|---:|---:|---:|
| W1 | -1.265 | -1.228 | 1.553 |
| W2 | -0.632 | -0.351 | 0.222 |
| W3 | 0.000 | -0.351 | 0.000 |
| W4 | 0.632 | 0.526 | 0.333 |
| W5 | 1.265 | 1.403 | 1.775 |

Cara meraba perkalian tiap baris:

```text
W1:
-1.265 x -1.228

abaikan dulu tanda minusnya:
1.265 x 1.2   = 1.518000
1.265 x 0.028 = 0.035420
jumlah        = 1.553420
dibulatkan    = 1.553

minus x minus -> positif
jadi W1 = 1.553
```

```text
W2:
-0.632 x -0.351

abaikan dulu tanda minusnya:
0.632 x 0.35  = 0.221200
0.632 x 0.001 = 0.000632
jumlah        = 0.221832
dibulatkan    = 0.222

minus x minus -> positif
jadi W2 = 0.222
```

```text
W3:
0.000 x -0.351

0 dikali apa saja = 0
jadi W3 = 0.000
```

```text
W4:
0.632 x 0.526

Catatan bayi:
di tabel z-score kita tulis 3 desimal.
Digit belakang aslinya masih ada.
Supaya hasilnya sama dengan catatan, pakai angka yang belum dipotong habis.

0.632456 x 0.526235

0.632456 x 0.5      = 0.316228
0.632456 x 0.02     = 0.012649
0.632456 x 0.006    = 0.003795
0.632456 x 0.0002   = 0.000126
0.632456 x 0.00003  = 0.000019
0.632456 x 0.000005 = 0.000003
jumlah              = 0.332820
dibulatkan          = 0.333

positif x positif -> positif
jadi W4 = 0.333
```

```text
W5:
1.265 x 1.403

1.265 x 1.4   = 1.771000
1.265 x 0.003 = 0.003795
jumlah        = 1.774795
dibulatkan    = 1.775

positif x positif -> positif
jadi W5 = 1.775
```

Jumlahkan kolom terakhir:

```text
mulai dari 1.553

1.553 + 0.222 = 1.775
1.775 + 0.000 = 1.775
1.775 + 0.333 = 2.108
2.108 + 1.775 = 3.883

jadi jumlahnya 3.883
```

$$
\sum z_Ez_F=1.553+0.222+0.000+0.333+1.775=3.883
$$

Karena `n = 5`, maka `n - 1 = 4`.

Sekarang bagi pelan-pelan:

```text
3.883 / 4

4 x 0.9   = 3.6    (kurang)
4 x 0.97  = 3.88   (kurang dikit)
4 x 0.971 = 3.884  (pas, dibulatkan)

jadi 3.883 / 4 = 0.971
```

$$
r=\frac{3.883}{4}=0.971
$$

Catatan pembulatan: jika memakai angka z-score yang lebih presisi dari standar deviasi asli, correlation yang lebih tepat sekitar:

$$
r=0.9707\ldots
$$

Di catatan ini kita tulis `0.971` agar hitungan tangan pendek.

Jadi matrix correlation:

$$
S=\begin{bmatrix}1 & 0.971 \\ 0.971 & 1\end{bmatrix}
$$

Interpretasi:

```text
E dan F sangat kuat naik turun bersama pada data normal
```

Kalau sudah paham bagian ini, kamu harus bisa:

| Kamu harus bisa | Hasil contoh |
|---|---:|
| menghitung `z_E * z_F` tiap window | lihat tabel |
| menjumlahkan hasil perkalian | 3.883 |
| membagi dengan $n-1$ | 0.971 |
| membaca arti $r$ | korelasi positif kuat |

## Bagian 13. Tentukan PC1 dan eigenvalue

Kita sudah punya:

$$
S=\begin{bmatrix}1 & 0.971 \\ 0.971 & 1\end{bmatrix}
$$

Untuk matrix dua fitur seperti ini, saat correlation $r$ positif kuat, PC1 adalah arah dua fitur naik bersama.

Kita pakai:

$$
p_1=[0.707,0.707]
$$

Catatan pembulatan:

$$
0.707\ \text{adalah pembulatan dari}\ \frac{1}{\sqrt{2}}
$$

Cara meraba $\sqrt{2}$ dulu:

```text
cari angka yang kalau dikuadratkan jadi 2

1 x 1 = 1       (kurang)
2 x 2 = 4       (kelebihan)
1.4 x 1.4 = 1.96      (kurang)
1.41 x 1.41 = 1.9881  (kurang)
1.414 x 1.414 = 1.999396  (pas dekat 2)

jadi sqrt(2) kira-kira 1.414
```

Sekarang raba `1 / 1.414`:

```text
1.414 x 0.7   = 0.989800  (kurang dari 1)
1.414 x 0.707 = 0.999698  (pas dekat 1)

jadi 1 / 1.414 = 0.707
```

$$
\frac{1}{\sqrt{2}}=0.707106\ldots
$$

Di kertas, `0.707` cukup. Di kode, gunakan angka loading yang keluar dari PCA dengan presisi penuh.

Arti $p_1$:

| Komponen | Arti |
|---|---|
| angka pertama `0.707` | bobot untuk z_E |
| angka kedua `0.707` | bobot untuk z_F |

Karena bobotnya sama-sama positif, PC1 membaca pola:

```text
E tinggi bersama F tinggi
E rendah bersama F rendah
```

Eigenvalue PC1:

Cara meraba eigenvalue PC1:

```text
lambda_1 = 1 + r
r = 0.971

1 + 0.971 = 1.971

jadi lambda_1 = 1.971
```

$$
\lambda_1 = 1+r = 1+0.971 = 1.971
$$

Arti `lambda_1 = 1.971`: PC1 menjelaskan variasi besar pada data normal. Karena total variance pada dua fitur z-score adalah kira-kira `2`, maka PC1 menjelaskan hampir semuanya.

Dalam contoh dua fitur ini:

Sekarang raba pembagian `1.971 / 2`:

```text
2 x 0.9    = 1.8    (kurang)
2 x 0.98   = 1.96   (kurang dikit)
2 x 0.985  = 1.970  (kurang tipis)
2 x 0.9855 = 1.971  (pas)

jadi 1.971 / 2 = 0.9855
```

$$
\text{persentase variance PC1}=\frac{\lambda_1}{2}=\frac{1.971}{2}=0.9855
$$

Jadi persentasenya 98.55 persen.

Jadi kalimat pendeknya: PC1 menjelaskan sekitar 98.5 persen variasi dua fitur. Sisa sekitar 1.5 persen berada di arah PC2.

PC2 tidak kita simpan dalam contoh ini. Kalau PC2 disimpan juga, rekonstruksi dengan dua fitur dan dua PC bisa menjadi sempurna, lalu Q menjadi nol. Untuk belajar Q/SPE, kita sengaja menyimpan 1 PC saja.

Kalau sudah paham bagian ini, kamu harus bisa:

| Kamu harus bisa | Hasil contoh |
|---|---|
| menulis loading PC1 | `[0.707, 0.707]` |
| menjelaskan arah PC1 | dua fitur naik bersama |
| menghitung eigenvalue PC1 | `1 + 0.971 = 1.971` |

## Bagian 14. Proyeksi PCA, menghitung score $t$

Sekarang setiap window akan diproyeksikan ke PC1.

**Projection** berarti kita mencari bayangan titik pada arah PC1.

Score PCA pada PC1 kita tulis $t$.

Rumus:

$$
t = z p_1
$$

Arti simbol:

| Simbol | Arti |
|---|---|
| $t$ | score PCA pada PC1 |
| $z$ | vektor z-score satu window, bentuk `[z_E, z_F]` |
| $p_1$ | loading PC1, bentuk `[0.707, 0.707]` |
| `dot` | kalikan komponen sepasang lalu jumlahkan |

Karena ada dua fitur:

$$
t=z_E\cdot0.707+z_F\cdot0.707
$$

### 14.1 Hitung score W1

Untuk W1:

$$
z(W1)=[-1.265,-1.228]
$$

Hitung:

Cara meraba dot product W1:

```text
t_W1 = (-1.265 x 0.707) + (-1.228 x 0.707)
```

Perkalian pertama:

```text
1.265 x 0.707

1.265 x 0.7   = 0.885500
1.265 x 0.007 = 0.008855
jumlah        = 0.894355
dibulatkan    = 0.894

karena -1.265 tandanya negatif:
-1.265 x 0.707 = -0.894
```

Perkalian kedua:

```text
1.228 x 0.707

1.228 x 0.7   = 0.859600
1.228 x 0.007 = 0.008596
jumlah        = 0.868196
dibulatkan    = 0.868

karena -1.228 tandanya negatif:
-1.228 x 0.707 = -0.868
```

Jumlahkan dua hasilnya:

```text
-0.894 + (-0.868) = -1.762

jadi t_W1 = -1.762
```

$$
t_{W1}=(-1.265)(0.707)+(-1.228)(0.707)=-0.894+(-0.868)=-1.762
$$

Arti `t_W1` negatif: W1 berada di sisi rendah dari pola normal, energi rendah dan frekuensi rendah.

### 14.2 Hitung score semua window

Untuk semua window, caranya sama:

```text
t = z_E x 0.707 + z_F x 0.707
```

Kita pakai hasil perkalian yang dibulatkan 3 desimal, seperti gaya tabel catatan.

```text
W1:

z_E x 0.707:
1.265 x 0.7   = 0.885500
1.265 x 0.007 = 0.008855
jumlah        = 0.894355
dibulatkan    = 0.894
tandanya negatif -> -0.894

z_F x 0.707:
1.228 x 0.7   = 0.859600
1.228 x 0.007 = 0.008596
jumlah        = 0.868196
dibulatkan    = 0.868
tandanya negatif -> -0.868

jumlah t:
-0.894 + (-0.868) = -1.762

jadi t_W1 = -1.762
```

```text
W2:

z_E x 0.707:
0.632 x 0.7   = 0.442400
0.632 x 0.007 = 0.004424
jumlah        = 0.446824
dibulatkan    = 0.447
tandanya negatif -> -0.447

z_F x 0.707:
0.351 x 0.7   = 0.245700
0.351 x 0.007 = 0.002457
jumlah        = 0.248157
dibulatkan    = 0.248
tandanya negatif -> -0.248

jumlah t:
-0.447 + (-0.248) = -0.695

jadi t_W2 = -0.695
```

```text
W3:

z_E x 0.707:
0.000 x 0.707 = 0.000

z_F x 0.707:
0.351 x 0.7   = 0.245700
0.351 x 0.007 = 0.002457
jumlah        = 0.248157
dibulatkan    = 0.248
tandanya negatif -> -0.248

jumlah t:
0.000 + (-0.248) = -0.248

jadi t_W3 = -0.248
```

```text
W4:

z_E x 0.707:
0.632 x 0.7   = 0.442400
0.632 x 0.007 = 0.004424
jumlah        = 0.446824
dibulatkan    = 0.447
tandanya positif -> 0.447

z_F x 0.707:
0.526 x 0.7   = 0.368200
0.526 x 0.007 = 0.003682
jumlah        = 0.371882
dibulatkan    = 0.372
tandanya positif -> 0.372

jumlah t:
0.447 + 0.372 = 0.819

jadi t_W4 = 0.819
```

```text
W5:

z_E x 0.707:
1.265 x 0.7   = 0.885500
1.265 x 0.007 = 0.008855
jumlah        = 0.894355
dibulatkan    = 0.894
tandanya positif -> 0.894

z_F x 0.707:
1.403 x 0.7   = 0.982100
1.403 x 0.007 = 0.009821
jumlah        = 0.991921
dibulatkan    = 0.992
tandanya positif -> 0.992

jumlah t:
0.894 + 0.992 = 1.886

jadi t_W5 = 1.886
```

| Window | $z_E$ | $z_F$ | Perhitungan t | t |
|---:|---:|---:|---|---:|
| W1 | -1.265 | -1.228 | `-1.265*0.707 + -1.228*0.707` | -1.762 |
| W2 | -0.632 | -0.351 | `-0.632*0.707 + -0.351*0.707` | -0.695 |
| W3 | 0.000 | -0.351 | `0.000*0.707 + -0.351*0.707` | -0.248 |
| W4 | 0.632 | 0.526 | `0.632*0.707 + 0.526*0.707` | 0.819 |
| W5 | 1.265 | 1.403 | `1.265*0.707 + 1.403*0.707` | 1.886 |

Interpretasi score:

| Nilai t | Arti |
|---|---|
| negatif | sisi energi dan frekuensi rendah |
| mendekati nol | dekat pusat normal |
| positif | sisi energi dan frekuensi tinggi |
| sangat besar absolutnya | jauh di sepanjang pola PC1 |

Kalau sudah paham bagian ini, kamu harus bisa menghitung satu $t$ dengan dot product.

## Bagian 15. Rekonstruksi dari PC1

Setelah kita tahu score $t$, kita bisa menggambar ulang atau merekonstruksi posisi window memakai PC1 saja.

Kita tulis hasil rekonstruksi sebagai $\hat z$.

Rumus untuk 1 PC:

$$
\hat z = t p_1^\top
$$

Arti simbol:

| Simbol | Arti |
|---|---|
| $\hat z$ | z-score hasil rekonstruksi PCA |
| $t$ | score PCA pada PC1 |
| $p_1$ | loading PC1 |

Karena `p1 = [0.707, 0.707]`, maka:

$$
\hat z_E=t\cdot0.707
$$

$$
\hat z_F=t\cdot0.707
$$

### 15.1 Rekonstruksi W1

Untuk W1:

$$
t_{W1}=-1.762,\quad p_1=[0.707,0.707]
$$

Hitung pelan. Karena dua loading sama-sama `0.707`, hitung satu kali dulu.

```text
1.762 x 0.7   = 1.2334
1.762 x 0.007 = 0.012334
jumlah        = 1.245734
dibulatkan    = 1.246
```

Karena score-nya negatif, hasilnya ikut negatif:

$$
\hat z_{W1}=-1.762[0.707,0.707]=[-1.246,-1.246]
$$

PCA menggambar ulang W1 sebagai titik yang tepat berada di garis PC1.

### 15.2 Residual

**Residual** adalah sisa yang tidak bisa dijelaskan oleh PC1.

Kita tulis residual sebagai $e$.

Rumus:

$$
e=z-\hat z
$$

Arti simbol:

| Simbol | Arti |
|---|---|
| $e$ | residual |
| $z$ | z-score asli window |
| $\hat z$ | z-score rekonstruksi dari PC1 |

Untuk W1:

$$
z_{W1}=[-1.265,-1.228],\quad \hat z_{W1}=[-1.246,-1.246]
$$

Hitung residual satu-satu, jangan lompat.

```text
e_E = z_E - z_hat_E
    = -1.265 - (-1.246)
    = -1.265 + 1.246
    = -0.019

e_F = z_F - z_hat_F
    = -1.228 - (-1.246)
    = -1.228 + 1.246
    = 0.018
```

Jadi:

$$
e_{W1}=z_{W1}-\hat z_{W1}=[-1.265,-1.228]-[-1.246,-1.246]=[-0.019,0.018]
$$

Residual W1 kecil. Artinya W1 cocok dengan garis PC1.

### 15.3 Rekonstruksi dan residual semua window

Cara ngisi tabelnya sama: hitung dulu $\hat z=t\cdot0.707$, lalu residual $e=z-\hat z$.

```text
W1 z_hat:
1.762 x 0.7   = 1.2334
1.762 x 0.007 = 0.012334
jumlah        = 1.245734 -> 1.246
karena t negatif, z_hat = -1.246

W1 residual:
e_E = -1.265 - (-1.246) = -0.019
e_F = -1.228 - (-1.246) =  0.018
```

```text
W2 z_hat:
0.695 x 0.7   = 0.4865
0.695 x 0.007 = 0.004865
jumlah        = 0.491365 -> 0.491
karena t negatif, z_hat = -0.491

W2 residual:
e_E = -0.632 - (-0.491) = -0.141
e_F = -0.351 - (-0.491) =  0.140
```

```text
W3 z_hat:
0.248 x 0.7   = 0.1736
0.248 x 0.007 = 0.001736
jumlah        = 0.175336 -> 0.175
karena t negatif, z_hat = -0.175

W3 residual:
e_E =  0.000 - (-0.175) =  0.175
e_F = -0.351 - (-0.175) = -0.176
```

```text
W4 z_hat:
0.819 x 0.7   = 0.5733
0.819 x 0.007 = 0.005733
jumlah        = 0.579033 -> 0.579

W4 residual:
e_E = 0.632 - 0.579 =  0.053
e_F = 0.526 - 0.579 = -0.053
```

```text
W5 z_hat:
1.886 x 0.7   = 1.3202
1.886 x 0.007 = 0.013202
jumlah        = 1.333402 -> 1.333

W5 residual:
e_E = 1.265 - 1.333 = -0.068
e_F = 1.403 - 1.333 =  0.070
```

| Window | t | z_hat_E | z_hat_F | e_E = z_E - z_hat_E | e_F = z_F - z_hat_F |
|---:|---:|---:|---:|---:|---:|
| W1 | -1.762 | -1.246 | -1.246 | -0.019 | 0.018 |
| W2 | -0.695 | -0.491 | -0.491 | -0.141 | 0.140 |
| W3 | -0.248 | -0.175 | -0.175 | 0.175 | -0.176 |
| W4 | 0.819 | 0.579 | 0.579 | 0.053 | -0.053 |
| W5 | 1.886 | 1.333 | 1.333 | -0.068 | 0.070 |

Kalau sudah paham bagian ini, kamu harus bisa:

| Kamu harus bisa | Contoh |
|---|---|
| menghitung $\hat z$ | `-1.762 * [0.707,0.707]` |
| menghitung residual | `z - z_hat` |
| menjelaskan residual kecil | window cocok dengan pola PC1 |

## Bagian 16. Hotelling T2

### Visual atlas: T2 sebagai jarak sepanjang pola normal

Sebelum masuk rumus, pakai satu gambar dulu. Pola normal pompa kita gambarkan sebagai sebuah **jalan**. T2 cuma satu pertanyaan sederhana: titik ini sudah melaju sejauh apa **di sepanjang jalan**, tetapi masih di atas aspal?

#### 1. Lihat petanya

![Peta skor PCA: jalan PC1 di dalam elips toleransi, panah oranye t mengukur jarak sepanjang jalan, rumus T2 sama dengan t kuadrat dibagi lambda 1](assets/pca-t2-q-manual/t2-formula-score-ellipse.png)

Jalan hijau adalah **PC1**, arah pola normal. Bulatan gelap di tengah adalah **pusat normal**, kebiasaan rata-rata pompa. Lima titik W1 sampai W5 adalah window training yang semuanya duduk rapi di atas jalan.

#### 2. Elips toleransi

Elips tipis yang membungkus jalan itu adalah **toleransi**. Lebarnya diatur oleh $\lambda_1$, yaitu eigenvalue PC1. Kalau arah ini memang biasa berubah banyak, elipsnya lebar, jadi jarak jauh di arah itu belum langsung dianggap aneh.

#### 3. T2 = melaju di jalan

Panah oranye berjalan **searah jalan**, dari pusat normal ke titik window. Panjang langkah itulah yang diukur T2. Makin jauh titik meluncur ke ujung jalan, makin besar T2. Ingat: T2 besar belum tentu rusak, titiknya masih di atas aspal.

#### 4. Bukan keluar jalan

Tanda silang merah kecil mengingatkan: T2 tidak mengukur seberapa jauh titik **keluar** dari jalan. Itu tugas Q nanti. T2 murni jarak sepanjang jalan.

#### 5. Jembatan ke rumus

Cerita jalan ini persis rumus di bawah:

$$
T^2=\frac{t^2}{\lambda_1}
$$

$t$ adalah seberapa jauh titik di jalan (score PC1), dikuadratkan supaya arah maju atau mundur sama saja, lalu dibagi $\lambda_1$ sebagai lebar toleransi.

#### 6. Hubungkan ke angka window

Lihat tabel di bawah. W5 punya $t=1.886$, paling jauh dari pusat, jadi $T^2=1.805$ paling besar. W3 hampir di pusat ($t=-0.248$), jadi $T^2=0.031$ paling kecil. Semua tetap di atas jalan, hanya beda jauhnya melaju.

#### 7. Self check

Tutup gambar, jawab cepat:

1. Jalan hijau melambangkan apa?
2. Panah oranye mengukur jarak ke arah mana?
3. Kenapa kita bagi dengan $\lambda_1$?
4. T2 besar selalu berarti rusak? Kenapa tidak?

Kalau keempatnya lancar, lanjut ke hitungan angka di bawah.

Hotelling T2 mengukur seberapa jauh window berada di subspace PCA yang disimpan.

Karena kita hanya menyimpan 1 PC, rumusnya sederhana:

$$
T^2=\frac{t^2}{\lambda_1}
$$

Arti simbol:

| Simbol | Arti |
|---|---|
| $T^2$ | skor Hotelling T2 |
| $t$ | score PCA pada PC1 |
| `t^2` | score dikuadratkan |
| $\lambda_1$ | eigenvalue PC1 |

Kenapa dibagi $\lambda_1$? Karena PC yang variasi normalnya besar diberi toleransi lebih besar. Kalau arah itu memang biasa berubah banyak, jarak di arah itu tidak langsung dianggap aneh.

### 16.1 T2 untuk W1

Untuk W1:

$$
t_{W1}=-1.762,\quad \lambda_1=1.971
$$

Hitung kuadrat score dulu.

```text
1.762 x 1.762

1.762 x 1     = 1.762
1.762 x 0.7   = 1.2334
1.762 x 0.06  = 0.10572
1.762 x 0.002 = 0.003524
jumlah        = 3.104644
dibulatkan    = 3.105
```

Jadi:

$$
(-1.762)^2=3.105
$$

Sekarang bagi dengan $\lambda_1=1.971$. Kita meraba hasil bagi.

```text
1.971 x 1.5   = 2.9565 (kurang)
1.971 x 1.6   = 3.1536 (lebih)
1.971 x 1.57  = 3.09447 (kurang dikit)
1.971 x 1.575 = 3.104325 (pas, dibulatkan)
jadi 3.105 / 1.971 = 1.575
```

Maka:

$$
T^2_{W1}=\frac{(-1.762)^2}{1.971}=\frac{3.105}{1.971}=1.575
$$

### 16.2 T2 semua window training

Cara ngisi tabel: kuadratkan $t$, lalu bagi dengan `1.971`.

```text
W1:
t^2 = 1.762 x 1.762 = 3.104644 -> 3.105

1.971 x 1.5   = 2.9565 (kurang)
1.971 x 1.6   = 3.1536 (lebih)
1.971 x 1.575 = 3.104325 (pas, dibulatkan)
T2 = 3.105 / 1.971 = 1.575
```

```text
W2:
0.695 x 0.695

0.695 x 0.6   = 0.417
0.695 x 0.09  = 0.06255
0.695 x 0.005 = 0.003475
jumlah        = 0.483025 -> 0.483

1.971 x 0.2   = 0.3942 (kurang)
1.971 x 0.3   = 0.5913 (lebih)
1.971 x 0.24  = 0.47304 (kurang)
1.971 x 0.245 = 0.482895 (pas, dibulatkan)
T2 = 0.483 / 1.971 = 0.245
```

```text
W3:
0.248 x 0.248

0.248 x 0.2   = 0.0496
0.248 x 0.04  = 0.00992
0.248 x 0.008 = 0.001984
jumlah        = 0.061504 -> 0.062

1.971 x 0.03  = 0.05913 (kurang)
1.971 x 0.032 = 0.063072 (lebih)
1.971 x 0.031 = 0.061101 (pas, dibulatkan)
T2 = 0.062 / 1.971 = 0.031
```

```text
W4:
0.819 x 0.819

0.819 x 0.8   = 0.6552
0.819 x 0.019 = 0.015561
jumlah        = 0.670761 -> 0.671

1.971 x 0.3  = 0.5913 (kurang)
1.971 x 0.4  = 0.7884 (lebih)
1.971 x 0.34 = 0.67014 (pas, dibulatkan)
T2 = 0.671 / 1.971 = 0.340
```

```text
W5:
1.886 x 1.886

1.886 x 1     = 1.886
1.886 x 0.8   = 1.5088
1.886 x 0.08  = 0.15088
1.886 x 0.006 = 0.011316
jumlah        = 3.556996 -> 3.557

1.971 x 1.8   = 3.5478 (kurang)
1.971 x 1.81  = 3.56751 (lebih)
1.971 x 1.805 = 3.557655 (pas, dibulatkan)
T2 = 3.557 / 1.971 = 1.805
```

| Window | $t$ | $t^2$ | $T^2=t^2/1.971$ |
|---:|---:|---:|---:|
| W1 | -1.762 | 3.105 | 1.575 |
| W2 | -0.695 | 0.483 | 0.245 |
| W3 | -0.248 | 0.062 | 0.031 |
| W4 | 0.819 | 0.671 | 0.340 |
| W5 | 1.886 | 3.557 | 1.805 |

Interpretasi:

| $T^2$ | Arti sederhana |
|---|---|
| kecil | dekat pusat pola normal |
| besar | jauh di sepanjang arah normal |

W5 punya T2 paling besar karena ia paling jauh di sisi energi dan frekuensi tinggi, tetapi masih mengikuti pola normal naik bersama.

Kalau sudah paham bagian ini, kamu harus bisa menghitung T2 dari satu nilai $t$.

## Bagian 17. Q atau SPE

### Visual atlas: Q/SPE sebagai sisa gambar ulang PCA

Masih di jalan yang sama. Kalau T2 mengukur maju di jalan, Q mengukur hal lain: seberapa jauh titik **melenceng keluar** dari jalan. Q adalah sisa yang tidak bisa digambar ulang oleh PC1.

#### 1. Lihat petanya

![Peta residual PCA: titik asli z di luar jalan, proyeksi z topi di jalan, panah merah residual e tegak lurus keluar jalan, rumus Q sama dengan e_E kuadrat plus e_F kuadrat](assets/pca-t2-q-manual/q-spe-formula-residual-vector.png)

Titik kosong $z$ adalah window asli setelah distandardisasi. Titik di jalan, $\hat z$, adalah usaha PCA menggambar ulang $z$ hanya dengan PC1.

#### 2. PCA mencoba menggambar ulang

PCA tidak menyimpan titik asli. Ia hanya boleh menaruh titik **di jalan**. Maka ia memilih titik terdekat di jalan, yaitu $\hat z$, sebagai tebakan terbaiknya untuk $z$.

#### 3. Q = sisa yang gagal digambar

Panah merah dari $\hat z$ ke $z$ adalah **residual** $e=z-\hat z$. Itu bagian yang tidak tertangkap PC1. Sudut siku kecil menandai bahwa residual diukur **tegak lurus keluar** jalan, bukan sepanjang jalan.

#### 4. Dua komponen residual

Panah merah punya dua bagian, $e_E$ dan $e_F$, yaitu sisa di fitur E dan fitur F. Keduanya dikuadratkan supaya tidak saling menghapus, lalu dijumlahkan.

#### 5. Jembatan ke rumus

Gambar ini persis rumus di bawah:

$$
Q=e_E^2+e_F^2
$$

Q besar berarti panah merah panjang, titik jauh dari jalan, hubungan antar fitur sudah tidak cocok dengan pola normal.

#### 6. Hubungkan ke angka window

Untuk window training, panah merahnya pendek. Contoh W1 punya $e_E=-0.019$ dan $e_F=0.018$, jadi $Q=0.001$, hampir menempel jalan. Itu sebabnya semua $Q$ training kecil di tabel bawah.

#### 7. Self check

Tutup gambar, jawab cepat:

1. Apa beda $z$ dan $\hat z$?
2. Panah merah mengukur jarak ke arah mana?
3. Kenapa residual dikuadratkan sebelum dijumlah?
4. Q besar artinya apa tentang hubungan E dan F?

Kalau lancar, lanjut ke hitungan residual di bawah.

Q atau SPE mengukur sisa error rekonstruksi.

Kita pakai rumus:

$$
Q=\mathrm{SPE}=e_E^2+e_F^2
$$

Arti simbol:

| Simbol | Arti |
|---|---|
| $Q$ | Q statistic, sama dengan SPE di catatan ini |
| $\mathrm{SPE}$ | squared prediction error |
| $e_E$ | residual fitur E |
| $e_F$ | residual fitur F |
| $e_E^2$ | residual E dikuadratkan |
| $e_F^2$ | residual F dikuadratkan |

Bahasa bayinya: lihat sisa error di setiap fitur, kuadratkan supaya tidak saling hapus, lalu jumlahkan.

### 17.1 Q untuk W1

Residual W1:

$$
e_E=-0.019,\quad e_F=0.018
$$

Hitung kuadrat residual satu-satu.

```text
0.019 x 0.019

0.019 x 0.01  = 0.00019
0.019 x 0.009 = 0.000171
jumlah        = 0.000361
```

```text
0.018 x 0.018

0.018 x 0.01  = 0.00018
0.018 x 0.008 = 0.000144
jumlah        = 0.000324
```

Jumlahkan:

```text
0.000361
0.000324
--------
0.000685
```

Jadi:

$$
Q_{W1}=(-0.019)^2+(0.018)^2=0.000361+0.000324=0.000685
$$

Dibulatkan:

$$
Q_{W1}=0.001
$$

### 17.2 Q semua window training

Cara ngisi tabel: kuadratkan residual E, kuadratkan residual F, lalu jumlahkan.

```text
W1:
0.019^2 = 0.000361
0.018^2 = 0.000324
Q       = 0.000361 + 0.000324
        = 0.000685 -> 0.001
```

```text
W2:
0.141 x 0.141

0.141 x 0.1   = 0.0141
0.141 x 0.04  = 0.00564
0.141 x 0.001 = 0.000141
jumlah        = 0.019881 -> 0.020

0.140 x 0.140

0.140 x 0.1  = 0.014
0.140 x 0.04 = 0.0056
jumlah       = 0.019600 -> 0.020

Q = 0.019881 + 0.019600
  = 0.039481
di tabel bayi, komponen dibulatkan jadi 0.020 + 0.020 = 0.040
jadi Q yang dipakai note = 0.040 (dibulatkan)
```

```text
W3:
0.175 x 0.175

0.175 x 0.1   = 0.0175
0.175 x 0.07  = 0.01225
0.175 x 0.005 = 0.000875
jumlah        = 0.030625 -> 0.031

0.176 x 0.176

0.176 x 0.1   = 0.0176
0.176 x 0.07  = 0.01232
0.176 x 0.006 = 0.001056
jumlah        = 0.030976 -> 0.031

Q = 0.030625 + 0.030976
  = 0.061601 -> 0.062
```

```text
W4:
0.053 x 0.053

0.053 x 0.05  = 0.00265
0.053 x 0.003 = 0.000159
jumlah        = 0.002809 -> 0.003

0.053 x 0.053 = 0.002809 -> 0.003

Q = 0.002809 + 0.002809
  = 0.005618 -> 0.006
```

```text
W5:
0.068 x 0.068

0.068 x 0.06  = 0.00408
0.068 x 0.008 = 0.000544
jumlah        = 0.004624 -> 0.005

0.070 x 0.070

0.070 x 0.07 = 0.004900 -> 0.005

Q = 0.004624 + 0.004900
  = 0.009524 -> 0.010
```

| Window | $e_E$ | $e_F$ | $e_E^2$ | $e_F^2$ | $Q/\mathrm{SPE}$ |
|---:|---:|---:|---:|---:|---:|
| W1 | -0.019 | 0.018 | 0.000 | 0.000 | 0.001 |
| W2 | -0.141 | 0.140 | 0.020 | 0.020 | 0.040 |
| W3 | 0.175 | -0.176 | 0.031 | 0.031 | 0.062 |
| W4 | 0.053 | -0.053 | 0.003 | 0.003 | 0.006 |
| W5 | -0.068 | 0.070 | 0.005 | 0.005 | 0.010 |

Tabel di atas membulatkan $e_E^2$ dan $e_F^2$ ke tiga desimal. Untuk angka sangat kecil, ini bisa membingungkan karena terlihat seperti `0.000`, padahal bukan nol.

Contoh W1 dengan enam desimal:

| Komponen | Nilai residual | Kuadrat enam desimal |
|---|---:|---:|
| $e_E$ | -0.019 | 0.000361 |
| $e_F$ | 0.018 | 0.000324 |
| jumlah |  | 0.000685 |

Jadi $Q_{W1}$ kecil, bukan hilang. Saat dibulatkan tiga desimal, `0.000685` menjadi kira-kira `0.001`.

Interpretasi:

| $Q$ | Arti sederhana |
|---|---|
| kecil | window bisa digambar ulang oleh PC1 dengan baik |
| besar | ada pola yang tidak cocok dengan PC1 |

Jika energi naik tetapi frekuensi tidak ikut naik, titik bisa jauh dari garis PC1. Q akan besar.

Kalau sudah paham bagian ini, kamu harus bisa menghitung Q dari residual.

## Bagian 18. Threshold atau ambang batas

### Visual atlas: threshold sebagai pagar normal

Skor T2 dan Q sudah selesai dihitung, tetapi angka mentah belum bisa memutuskan apa-apa. Kita butuh **pagar**. Threshold adalah pagar di sekeliling kebiasaan normal pompa. Window yang masih di dalam pagar dianggap normal. Window yang melompati pagar baru dicurigai.

#### 1. Lihat petanya

![Peta threshold PCA: diagram batang T2 dan Q training normal dengan garis max normal, metafora pagar normal, catatan p95 validation, aturan anti-bocor, dan angka threshold_T2 1.805 serta threshold_Q 0.062](assets/pca-t2-q-manual/threshold-empirical-rules.png)

Di kiri ada dua diagram batang. Batang oranye adalah skor T2 lima window normal W1 sampai W5. Batang merah adalah skor Q lima window normal yang sama. Di kanan ada halaman bergaris pagar: titik normal duduk tenang di dalam, satu titik merah melompat keluar dan diberi label anomali.

#### 2. Tinggi pagar sama dengan skor normal tertinggi

Untuk latihan tangan, pagar dipasang setinggi batang paling tinggi. Garis putus-putus `threshold = max(normal)` menempel di puncak batang tertinggi. Artinya semua window normal masih tertutup pagar, dan apa pun yang lebih tinggi dari itu dianggap melewati batas.

#### 3. Pagar versi proyek nyata

Kartu `Proyek nyata` mengingatkan bahwa `max` itu rapuh. Satu skor normal yang kebetulan besar bisa membuat pagar terlalu tinggi. Maka proyek nyata sering memakai `threshold = p95(validation normal)`, yaitu batas yang masih meloloskan sekitar 95 persen skor normal.

#### 4. Aturan anti-bocor

Kartu bergembok adalah aturan paling penting: ambang hanya boleh dihitung dari training atau validation normal. Test tetap menjadi ujian akhir. Kalau threshold diintip dari test, evaluasi bocor dan model terlihat lebih pintar dari aslinya.

#### 5. Garis keputusan dan kasus tepi

Strip keputusan menulis `anomali jika skor > threshold`. Perhatikan tandanya: kita pakai `>`, bukan `>=`. Jadi kalau skor persis sama dengan threshold, di catatan ini ia belum dianggap melewati pagar. Pilih satu aturan sejak awal lalu pakai konsisten.

#### 6. Jembatan ke rumus dan angka

Cerita pagar ini persis rumus di bawah:

$$
\text{threshold}_{T^2}=\max(T^2\ \text{training normal})=1.805
$$

$$
\text{threshold}_Q=\max(Q\ \text{training normal})=0.062
$$

Angka `1.805` datang dari T2 terbesar, yaitu W5. Angka `0.062` datang dari Q terbesar, yaitu W3. Keduanya bisa kamu cek pada tabel training di bawah.

#### 7. Self check

Tutup gambar, jawab cepat:

1. Kenapa kita butuh pagar sebelum memutuskan anomali?
2. Tinggi pagar manual diambil dari nilai apa?
3. Kenapa proyek nyata lebih suka p95 validation daripada max?
4. Kalau skor persis sama dengan threshold, lewat atau tidak di catatan ini?

Kalau keempatnya lancar, lanjut ke hitungan ambang di bawah.

Skor T2 dan Q baru berguna kalau kita punya batas.

Kita akan pakai threshold empiris sederhana dari data normal training. Untuk latihan tangan, cara paling mudah adalah mengambil nilai maksimum dari skor normal training.

$$
\text{threshold}_{T^2}=\max(T^2\ \text{training normal})
$$

$$
\text{threshold}_Q=\max(Q\ \text{training normal})
$$

Arti simbol:

| Simbol | Arti |
|---|---|
| $\text{threshold}_{T^2}$ | batas T2 yang dianggap masih normal |
| $\text{threshold}_Q$ | batas Q yang dianggap masih normal |
| `max` | nilai terbesar |

### 18.1 Ambang T2

T2 training:

| Window | T2 |
|---:|---:|
| W1 | 1.575 |
| W2 | 0.245 |
| W3 | 0.031 |
| W4 | 0.340 |
| W5 | 1.805 |

Nilai terbesar:

Kita tidak hitung berat. Kita cuma bandingkan satu-satu, lalu ambil yang paling besar.

```text
mulai dari W1 = 1.575

bandingkan W2:
0.245 lebih kecil dari 1.575
jadi pegangan tetap 1.575

bandingkan W3:
0.031 lebih kecil dari 1.575
jadi pegangan tetap 1.575

bandingkan W4:
0.340 lebih kecil dari 1.575
jadi pegangan tetap 1.575

bandingkan W5:
1.805 lebih besar dari 1.575
jadi pegangan ganti ke 1.805

nilai terbesar = 1.805
```

$$
\text{threshold}_{T^2}=\max(1.575,0.245,0.031,0.340,1.805)=1.805
$$

### 18.2 Ambang Q

Q training:

| Window | Q |
|---:|---:|
| W1 | 0.001 |
| W2 | 0.040 |
| W3 | 0.062 |
| W4 | 0.006 |
| W5 | 0.010 |

Nilai terbesar:

Ini juga cuma bandingkan lalu ambil terbesar.

```text
mulai dari W1 = 0.001

bandingkan W2:
0.040 lebih besar dari 0.001
jadi pegangan ganti ke 0.040

bandingkan W3:
0.062 lebih besar dari 0.040
jadi pegangan ganti ke 0.062

bandingkan W4:
0.006 lebih kecil dari 0.062
jadi pegangan tetap 0.062

bandingkan W5:
0.010 lebih kecil dari 0.062
jadi pegangan tetap 0.062

nilai terbesar = 0.062
```

$$
\text{threshold}_Q=\max(0.001,0.040,0.062,0.006,0.010)=0.062
$$

### 18.3 Catatan threshold untuk proyek nyata

Untuk belajar manual, `max(training normal)` cukup mudah.

Untuk proyek nyata, threshold lebih baik dikalibrasi dari validation normal, misalnya:

$$
\text{threshold}_{T^2}=p95(T^2\ \text{validation normal})
$$

$$
\text{threshold}_Q=p95(Q\ \text{validation normal})
$$

Di sini $p95$ berarti nilai yang kira-kira lebih besar dari 95 persen data normal. Kalau ada 100 skor normal, sekitar 95 skor berada di bawah p95.

Aturan anti-bocor:

| Aturan | Kenapa |
|---|---|
| jangan fit scaler pada validation/test | normalisasi harus berasal dari training normal |
| jangan pilih threshold dari test | test harus tetap menjadi ujian akhir |
| jangan pakai label anomaly sebagai fitur | model akan curang |
| simpan threshold bersama model | inference harus memakai batas yang sama |

Catatan operator perbandingan: di catatan ini keputusan memakai tanda $>$.

$$
\text{anomali jika skor}>\text{threshold}
$$

Bukan:

$$
\text{anomali jika skor}\ge\text{threshold}
$$

Artinya, kalau skor persis sama dengan threshold, ia belum dianggap melewati batas pada aturan catatan ini. Dalam proyek nyata, pilih satu aturan sejak awal, tulis di metodologi, lalu pakai konsisten di training, evaluasi, dan inference.

Kalau sudah paham bagian ini, kamu harus bisa:

| Kamu harus bisa | Hasil contoh |
|---|---:|
| memilih threshold_T2 manual | 1.805 |
| memilih threshold_Q manual | 0.062 |
| menjelaskan p95 | batas yang meloloskan sekitar 95 persen normal |

## Bagian 19. Window inference baru

### Visual atlas: keputusan inference dari T2 dan Q

Bagian ini menghitung satu window baru dari awal sampai vonis. Sebelum tenggelam di angka, lihat alur keputusannya sebagai satu jalur dengan dua cabang alarm.

#### 1. Lihat alurnya

![Diagram alur keputusan inference X new: standardisasi, proyeksi PC1, hitung T2 dan Q, bandingkan threshold, cabang T2 salah dan cabang Q benar, kesimpulan anomali karena Q](assets/pca-t2-q-manual/t2-q-inference-decision-flow.png)

Mulai dari atas: window baru `X_new` dengan E=14 dan F=30 turun melewati empat langkah, lalu pecah jadi dua cabang.

#### 2. Empat langkah persiapan

Urutannya selalu sama: standardisasi jadi $z_{\text{new}}$, proyeksi ke PC1 jadi $t_{\text{new}}$, hitung T2 dan Q, baru bandingkan dengan threshold. Jangan menghitung mean atau std baru, pakai milik training.

#### 3. Cabang T2 (kiri)

Cabang kiri tenang. $T^2_{\text{new}}=1.205$ dibandingkan $\text{threshold}_{T^2}=1.805$. Karena $1.205>1.805$ adalah **SALAH**, alarm T2 tidak bunyi. Titik belum terlalu jauh melaju di jalan.

#### 4. Cabang Q (kanan)

Cabang kanan merah. $Q_{\text{new}}=4.150$ dibandingkan $\text{threshold}_Q=0.062$. Karena $4.150>0.062$ adalah **BENAR**, alarm Q bunyi keras. Titik melenceng jauh keluar jalan.

#### 5. Jembatan ke rumus

Aturannya satu OR:

$$
\mathrm{is\_anomaly}=(T^2_{\text{new}}>\text{threshold}_{T^2})\ \mathrm{OR}\ (Q_{\text{new}}>\text{threshold}_Q)
$$

Cukup satu cabang benar untuk memicu anomali. Di sini Q yang menangkapnya.

#### 6. Vonis akhir

Kotak merah bawah: **anomali karena Q melewati ambang**. T2 sendirian akan meloloskan window ini. Q yang menyelamatkan deteksi. Inilah kenapa kita butuh dua skor, bukan satu.

#### 7. Self check

Tutup gambar, jawab cepat:

1. Empat langkah sebelum membandingkan threshold, apa saja urutannya?
2. Cabang mana yang SALAH, cabang mana yang BENAR?
3. Kenapa aturan OR cukup satu cabang benar?
4. Kalau hanya pakai T2, apa yang terjadi pada window ini?

Kalau lancar, ikuti hitungan langkah demi langkah di bawah.

Sekarang model sudah punya semua bekal:

| Benda | Nilai |
|---|---:|
| `mu_E` | 10 |
| `s_E` | 1.581 |
| `mu_F` | 30.4 |
| `s_F` | 1.140 |
| $p_1$ | `[0.707, 0.707]` |
| $\lambda_1$ | 1.971 |
| $\text{threshold}_{T^2}$ | 1.805 |
| $\text{threshold}_Q$ | 0.062 |

Window baru punya fitur spektral:

| Window baru | E | F |
|---|---:|---:|
| X_new | 14 | 30 |

Secara rasa, ini mencurigakan. Energi band tinggi, tetapi frekuensi dominan tidak ikut naik. Pada data normal, energi dan frekuensi cenderung naik bersama.

Sekarang kita hitung dengan prosedur PCA T2/Q.

### 19.1 Standardisasi window baru

Pakai mean dan standar deviasi training. Jangan hitung baru.

Untuk E:

Kurangi dulu.

```text
14 - 10 = 4
```

Lalu raba pembagian `4 / 1.581`.

```text
1.581 x 2.5  = 3.953 (kurang)
1.581 x 2.53 = 4.000 (pas)
jadi 4 / 1.581 = 2.530
```

$$
z_E=\frac{E_{\text{new}}-\mu_E}{s_E}=\frac{14-10}{1.581}=\frac{4}{1.581}=2.530
$$

Untuk F:

Kurangi dulu.

```text
30 - 30.4 = -0.4
```

Raba pembagian `0.4 / 1.140` dulu, tanda minus dipasang belakangan.

```text
1.140 x 0.35  = 0.399 (kurang sedikit)
1.140 x 0.351 = 0.400 (pas)
jadi 0.4 / 1.140 = 0.351

karena atasnya -0.4,
maka -0.4 / 1.140 = -0.351
```

$$
z_F=\frac{F_{\text{new}}-\mu_F}{s_F}=\frac{30-30.4}{1.140}=\frac{-0.4}{1.140}=-0.351
$$

Jadi:

$$
z_{\text{new}}=[2.530,-0.351]
$$

Arti z-score ini:

| Fitur | z-score | Arti |
|---|---:|---|
| E | 2.530 | energi jauh di atas rata-rata normal |
| F | -0.351 | frekuensi sedikit di bawah rata-rata normal |

Ini sudah memberi tanda: E tinggi, F tidak ikut tinggi.

### 19.2 Proyeksi window baru ke PC1

Rumus:

$$
t_{\text{new}}=z_{\text{new}}p_1
$$

Dengan angka:

Pecah perkalian pertama.

```text
2.530 x 0.707
= 2.530 x (0.700 + 0.007)

2.530 x 0.700 = 1.771
2.530 x 0.007 = 0.018

running:
1.771 + 0.018 = 1.789
```

Pecah perkalian kedua.

```text
-0.351 x 0.707
= -(0.351 x 0.707)
= -(0.351 x (0.700 + 0.007))

0.351 x 0.700 = 0.246
0.351 x 0.007 = 0.002

running:
0.246 + 0.002 = 0.248

karena tadi negatif:
-0.351 x 0.707 = -0.248
```

Jumlahkan pelan-pelan.

```text
t_new = 1.789 + (-0.248)
t_new = 1.541
```

$$
t_{\text{new}}=(2.530)(0.707)+(-0.351)(0.707)=1.789+(-0.248)=1.541
$$

Arti `t_new = 1.541`: window baru berada di sisi tinggi PC1, tetapi belum tentu terlalu ekstrem.

### 19.3 Hitung T2 window baru

Rumus:

$$
T^2_{\text{new}}=\frac{t_{\text{new}}^2}{\lambda_1}
$$

Hitung:

Kuadratkan dulu `1.541`.

```text
1.541^2 = 1.541 x 1.541
1.541 x 1.500 = 2.312
1.541 x 0.041 = 0.063

running:
2.312 + 0.063 = 2.375
```

Lalu raba pembagian `2.375 / 1.971`.

```text
1.971 x 1.2   = 2.365 (kurang)
1.971 x 1.205 = 2.375 (pas)
jadi 2.375 / 1.971 = 1.205
```

$$
T^2_{\text{new}}=\frac{(1.541)^2}{1.971}=\frac{2.375}{1.971}=1.205
$$

Bandingkan dengan threshold:

```text
T2 baru      = 1.205
threshold T2 = 1.805

1.205 lebih kecil dari 1.805
jadi 1.205 > 1.805 adalah False
```

$$
T^2_{\text{new}}=1.205,\quad \text{threshold}_{T^2}=1.805
$$

$$
1.205>1.805\ \text{adalah False}
$$

Dari T2 saja, window baru belum melewati batas ekstrem di sepanjang PC1.

### 19.4 Rekonstruksi window baru

Rumus:

$$
\hat z_{\text{new}}=t_{\text{new}}p_1^\top
$$

Hitung:

Kedua komponen dikali angka PC1 yang sama.

```text
1.541 x 0.707
= 1.541 x (0.700 + 0.007)

1.541 x 0.700 = 1.079
1.541 x 0.007 = 0.011

running:
1.079 + 0.011 = 1.090 (dibulatkan)
```

Jadi komponen pertama `1.090`, komponen kedua juga `1.090`.

$$
\hat z_{\text{new}}=1.541[0.707,0.707]=[1.090,1.090]
$$

PCA mencoba menggambar ulang window baru sebagai titik di garis normal PC1. Karena PC1 menganggap E dan F naik bersama, rekonstruksi membuat dua komponen sama-sama tinggi.

### 19.5 Residual window baru

Rumus:

$$
e_{\text{new}}=z_{\text{new}}-\hat z_{\text{new}}
$$

Hitung:

Kurangi komponen E.

```text
e_E = z_E - z_hat_E
e_E = 2.530 - 1.090
e_E = 1.440
```

Kurangi komponen F.

```text
e_F = z_F - z_hat_F
e_F = -0.351 - 1.090
e_F = -1.441
```

Jadi:

$$
e_{\text{new}}=[2.530,-0.351]-[1.090,1.090]=[2.530-1.090,-0.351-1.090]=[1.440,-1.441]
$$

Residual besar. PCA ingin menggambar ulang E dan F sama-sama tinggi, tetapi data asli punya E tinggi dan F rendah. Selisihnya besar.

### 19.6 Hitung Q/SPE window baru

Rumus:

$$
Q_{\text{new}}=e_E^2+e_F^2
$$

Hitung:

Kuadratkan residual E.

```text
1.440^2 = 1.440 x 1.440
1.440 x 1.000 = 1.440
1.440 x 0.400 = 0.576
1.440 x 0.040 = 0.058

running:
1.440 + 0.576 = 2.016
2.016 + 0.058 = 2.074
```

Kuadratkan residual F. Tanda minus hilang kalau dikuadratkan.

```text
(-1.441)^2 = 1.441 x 1.441
1.441 x 1.000 = 1.441
1.441 x 0.400 = 0.576
1.441 x 0.040 = 0.058
1.441 x 0.001 = 0.001

running:
1.441 + 0.576 = 2.017
2.017 + 0.058 = 2.075
2.075 + 0.001 = 2.076
```

Jumlahkan Q pelan-pelan.

```text
Q_new = 2.074 + 2.076
Q_new = 4.150
```

$$
Q_{\text{new}}=(1.440)^2+(-1.441)^2=2.074+2.076=4.150
$$

Bandingkan dengan threshold:

```text
Q baru      = 4.150
threshold Q = 0.062

4.150 lebih besar dari 0.062
jadi 4.150 > 0.062 adalah True
```

$$
Q_{\text{new}}=4.150,\quad \text{threshold}_Q=0.062
$$

$$
4.150>0.062\ \text{adalah True}
$$

Q melewati ambang dengan sangat besar.

### 19.7 Keputusan akhir

Aturan:

$$
\mathrm{is\_anomaly}=(T^2_{\text{new}}>\text{threshold}_{T^2})\ \mathrm{OR}\ (Q_{\text{new}}>\text{threshold}_Q)
$$

Kita memakai $>$, bukan $\ge$. Jadi kata "melewati threshold" berarti lebih besar dari threshold, bukan sama dengan threshold.

Masukkan angka:

```text
cabang T2:
1.205 > 1.805
ini False

cabang Q:
4.150 > 0.062
ini True

aturan OR:
False OR True = True
```

$$
\mathrm{is\_anomaly}=(1.205>1.805)\ \mathrm{OR}\ (4.150>0.062)=\mathrm{False}\ \mathrm{OR}\ \mathrm{True}=\mathrm{True}
$$

Jadi window baru diklasifikasikan sebagai anomali.

### 19.8 Skor gabungan sederhana

Kadang kita ingin satu angka ringkas. Kita bisa pakai:

$$
\text{score}=\max\left(\frac{T^2_{\text{new}}}{\text{threshold}_{T^2}},\frac{Q_{\text{new}}}{\text{threshold}_Q}\right)
$$

Arti simbol:

| Simbol | Arti |
|---|---|
| $\text{score}$ | skor gabungan relatif terhadap threshold |
| `max` | ambil nilai terbesar |
| `T2_new / threshold_T2` | seberapa dekat T2 ke ambang |
| `Q_new / threshold_Q` | seberapa dekat Q ke ambang |

Hitung:

Raba dulu skor T2, yaitu `1.205 / 1.805`.

```text
1.805 x 0.6    = 1.083 (kurang)
1.805 x 0.66   = 1.191 (kurang)
1.805 x 0.6676 = 1.205 (pas)

jadi 1.205 / 1.805 = 0.6676
dibulatkan tiga desimal = 0.668
```

$$
\frac{T^2_{\text{new}}}{\text{threshold}_{T^2}}=\frac{1.205}{1.805}=0.668
$$

Raba skor Q, yaitu `4.150 / 0.062`.

```text
0.062 x 66     = 4.092 (kurang)
0.062 x 67     = 4.154 (lebih)
0.062 x 66.9   = 4.148 (kurang sedikit)
0.062 x 66.935 = 4.150 (pas)

jadi 4.150 / 0.062 = 66.935
```

$$
\frac{Q_{\text{new}}}{\text{threshold}_Q}=\frac{4.150}{0.062}=66.935
$$

Sekarang ambil yang paling besar.

```text
score_T2 = 0.668
score_Q  = 66.935

66.935 lebih besar dari 0.668
jadi score = 66.935
```

$$
\text{score}=\max(0.668,66.935)=66.935
$$

Karena `score > 1`, window anomali. Penyebab utamanya Q/SPE, bukan T2.

Bahasa operatornya:

```text
Window ini tidak terlalu ekstrem di sepanjang pola normal utama, tetapi hubungan antar fitur rusak. Energi band tinggi, frekuensi dominan tidak ikut naik seperti pola normal.
```

Kalau sudah paham bagian ini, kamu harus bisa menghitung inference dari awal sampai keputusan akhir.

## Bagian 20. Membaca kombinasi T2 dan Q

### Visual atlas: T2 dan Q sebagai peta jalan PC1

Sebelum membaca angka, lihat dulu gambarnya. Bagian ini memakai satu cerita sederhana: pola normal adalah sebuah **jalan**. T2 dan Q cuma dua cara mengukur posisi satu titik terhadap jalan itu.

#### 1. Road scene

![Peta jalan PC1: T2 jarak sepanjang jalan, Q jarak keluar dari jalan](assets/pca-t2-q-manual/pc1-road-t2-q-map.png)

Lihat jalan hijau di gambar. Jalan itu adalah **PC1**, yaitu pola normal pompa. Titik-titik kecil di pangkal jalan adalah window-window normal. Bulatan gelap bernama **pusat normal** adalah titik tengah kebiasaan pompa. Semua window sehat berkumpul dekat jalan ini.

#### 2. T2 = jarak sepanjang jalan

Panah oranye berjalan **searah jalan**. Itulah **T2**. T2 menjawab pertanyaan bayi: titik ini sudah berjalan sejauh apa dari pusat normal, tetapi masih di atas jalan? Kalau titik meluncur jauh ke ujung jalan, T2 membesar. Catatan penting: T2 besar belum tentu rusak. Titiknya masih di atas aspal, hanya posisinya jauh.

#### 3. Q/SPE = jarak keluar dari jalan

Panah merah keluar **tegak lurus dari jalan**, menuju titik window baru. Itulah **Q** atau **SPE**. Q menjawab pertanyaan bayi: titik ini melenceng berapa jauh keluar dari jalan? Sudut siku kecil di kaki panah merah mengingatkan bahwa Q diukur lurus keluar jalan, bukan sepanjang jalan. Q besar berarti hubungan antar fitur sudah tidak cocok lagi dengan pola normal.

#### 4. Dua arah yang berbeda

Tahan dua gambar mental ini: T2 = **maju di jalan**, Q = **keluar dari jalan**. Dua arah ini tegak lurus, jadi satu titik bisa punya T2 kecil tetapi Q besar, atau sebaliknya. Karena arahnya beda, satu skor saja tidak cukup untuk membaca window.

#### 5. Empat kasus di peta

![Empat kasus kombinasi T2 dan Q sebagai posisi titik di peta jalan PC1](assets/pca-t2-q-manual/t2-q-four-cases-road.png)

Gambar kedua memecah cerita menjadi empat panel. Jalannya sama di semua panel, yang berubah hanya posisi dan warna titik:

1. **T2 rendah, Q rendah** (titik hijau di jalan, dekat pusat): mirip normal, window sehat.
2. **T2 tinggi, Q rendah** (titik oranye jauh tetapi tetap di jalan): jauh tetapi masih mengikuti pola normal, cek apakah beban operasi memang tinggi.
3. **T2 rendah, Q tinggi** (titik merah keluar dari jalan dekat pusat): tidak cocok dengan pola normal, energi naik sendiri, cek hubungan antar fitur atau fault awal.
4. **T2 tinggi, Q tinggi** (titik merah tua jauh dan keluar jalan): ekstrem dan pola rusak, dua alarm sepakat, cek lebih serius.

Empat panel ini sama persis dengan tabel kombinasi di bawah. Pakai gambar dulu, baru baca tabel.

#### 6. Hubungkan ke contoh $X_{\text{new}}$

Contoh window baru kita berada di panel ketiga: **T2 rendah, Q tinggi**. Titiknya tidak jauh meluncur di jalan ($T^2$ belum lewat threshold), tetapi melenceng jauh keluar jalan ($Q$ jauh di atas threshold). Maka keputusannya anomali **karena Q, bukan karena T2**. Aturan $T^2$ OR $Q$ menangkapnya justru lewat Q.

#### 7. Self check

Tutup gambar, lalu jawab cepat:

1. Jalan hijau di gambar melambangkan apa? (pola normal / PC1)
2. Panah oranye dan panah merah, mana T2 dan mana Q?
3. Titik dengan T2 kecil tetapi Q besar ada di panel mana, dan apa artinya?
4. Kenapa kita tidak boleh memakai T2 saja untuk contoh $X_{\text{new}}$?

Kalau keempatnya bisa kamu jawab tanpa melihat gambar, lanjut ke detail angka di bawah.

T2 dan Q memberi dua sudut pandang. Jangan memilih salah satu saja saat membaca PCA T2/Q.

Ingat cerita jalan:

| Skor | Pertanyaan bayi | Gambar mental |
|---|---|---|
| T2 | jauh tidak di jalan normal? | posisi jauh di sepanjang jalan PC1 |
| Q/SPE | keluar tidak dari jalan normal? | posisi melenceng dari jalan PC1 |

Tabel kombinasi:

| Kondisi | Arti praktis | Contoh rasa pada pompa | Tindakan membaca |
|---|---|---|---|
| T2 rendah, Q rendah | mirip normal | window sehat | tidak ada tanda kuat |
| T2 tinggi, Q rendah | jauh tetapi masih mengikuti pola normal | energi dan frekuensi sama-sama naik | cek apakah beban operasi memang tinggi |
| T2 rendah, Q tinggi | tidak cocok dengan pola normal | energi naik sendiri, frekuensi tidak ikut | cek hubungan antar fitur, sensor, atau fault awal |
| T2 tinggi, Q tinggi | ekstrem dan pola rusak | fault besar mengubah level dan struktur spektral | cek lebih serius karena dua alarm sepakat |

![Ilustrasi perbedaan T2 tinggi dan Q tinggi](assets/pca-t2-q-manual/t2-vs-q.png)

Pakai gambar ini sebagai pengingat cepat: T2 membesar saat titik berjalan jauh di rel normal. Q membesar saat titik keluar dari rel normal.

Mengapa dua skor perlu dipakai?

Jika hanya memakai T2, window baru pada contoh bisa lolos karena T2 tidak melewati threshold. Padahal Q sangat besar. Jika hanya memakai Q, kejadian yang sangat ekstrem tetapi masih berada di garis normal bisa kurang terlihat.

Jadi aturan $T^2$ OR $Q$ membuat detector peka terhadap dua bentuk keanehan:

1. jauh di arah normal,
2. keluar dari arah normal.

Untuk $X_{\text{new}}$, ceritanya begini:

| Bacaan | Nilai | Arti |
|---|---:|---|
| $T^2_{\text{new}}$ | 1.205 | belum lebih besar dari 1.805 |
| $Q_{\text{new}}$ | 4.150 | jauh lebih besar dari 0.062 |
| keputusan | anomali | Q melewati threshold |

Bahasa operatornya:

```text
Window tidak terlalu jauh di sepanjang jalan normal utama.
Tetapi window keluar jauh dari jalan normal itu.
Energi band tinggi, frekuensi dominan tidak ikut tinggi.
```

Kalau sudah paham bagian ini, kamu harus bisa menjelaskan kenapa contoh $X_{\text{new}}$ anomali karena Q, bukan karena T2.

## Bagian 21. Bentuk umum kalau PC lebih dari satu

### Visual atlas: banyak PC sebagai beberapa jalan normal

Sejauh ini kita hanya punya satu jalan normal, yaitu PC1. Tetapi proyek nyata bisa menyimpan beberapa jalan sekaligus. Bayangkan sebuah persimpangan: dari satu pusat normal memancar beberapa jalan, PC1, PC2, sampai PCA. Window punya bayangan di tiap jalan.

#### 1. Lihat petanya

![Peta banyak PC: beberapa jalan normal PC1 PC2 sampai PCA dari pusat normal, vektor skor t, rumus T2 jumlah t_a kuadrat dibagi lambda_a, rekonstruksi z topi, dan awan residual untuk Q](assets/pca-t2-q-manual/multi-pc-t2-q-generalization.png)

Dari bulatan pusat normal memancar beberapa jalan teal: PC1, PC2, sampai PCA. Titik window di samping menjatuhkan bayangan oranye ke tiap jalan, menghasilkan skor `t1`, `t2`, sampai `tA`.

#### 2. Beberapa jalan, beberapa bayangan

Kalau 1 PC berarti satu jalan, banyak PC berarti banyak jalan normal yang disimpan. Karena itu skor bukan lagi satu angka, tetapi satu vektor:

$$
t=[t_1,t_2,\ldots,t_A]
$$

Tiap $t_a$ adalah posisi bayangan window pada jalan ke-a.

#### 3. T2 sebagai jumlah kontribusi tiap jalan

Kartu `Hotelling T2 umum` menjumlahkan kontribusi semua jalan:

$$
T^2=\sum_{a=1}^{A}\frac{t_a^2}{\lambda_a}
$$

Tiap skor dikuadratkan, dibagi eigenvalue jalannya sendiri, lalu semua dijumlahkan. Untuk 1 PC, penjumlahan ini hanya punya satu suku, yaitu $t^2/\lambda_1$ yang sudah kita pakai.

#### 4. Rekonstruksi dari semua jalan

Window digambar ulang memakai semua PC yang disimpan:

$$
\hat z=tP_A^\top
$$

Di sini $P_A$ adalah kumpulan loading dari PC yang disimpan. Makin banyak jalan disimpan, makin mirip gambar ulang dengan titik asli.

#### 5. Q sebagai sisa yang tetap keluar

Awan merah pada gambar adalah residual, yaitu sisa yang tetap tidak tergambar walau semua jalan dipakai. Q tetap bertanya hal yang sama:

$$
Q=\sum_{j=1}^{p}(z_j-\hat z_j)^2
$$

#### 6. Hubungkan ke jalur utama

Pita bawah gambar berbunyi: pola sama seperti 1 PC, hanya diulang untuk tiap PC. Jadi jalur utama kita tetap aman. Kuasai dulu 2 fitur dan 1 PC, lalu multi-PC hanya pengulangan pola yang sama.

#### 7. Self check

Tutup gambar, jawab cepat:

1. Kalau ada A buah PC, skor window berbentuk apa?
2. Bagaimana T2 umum menjumlahkan kontribusi tiap PC?
3. Kenapa 1 PC hanyalah kasus khusus dari rumus umum?
4. Q multi-PC masih menanyakan hal yang sama atau berbeda?

Kalau lancar, lanjut ke penjelasan bentuk umum di bawah.

Bagian ini hanya pengantar. Jalur utama kita tetap 2 fitur dan 1 PC.

Tetapi di proyek nyata, PCA bisa menyimpan lebih dari satu PC. Misalnya ada banyak fitur spektral dan model menyimpan 3 PC.

Kalau PCA menyimpan $A$ principal component, score menjadi beberapa angka:

$$
t=[t_1,t_2,\ldots,t_A]
$$

Arti simbol:

| Simbol | Arti |
|---|---|
| $A$ | jumlah PC yang disimpan |
| $t_1$ | score pada PC1 |
| $t_2$ | score pada PC2 |
| $t_A$ | score pada PC ke-A |

Bahasa bayinya: kalau 1 PC berarti satu jalan utama, banyak PC berarti beberapa jalan arah normal yang disimpan. Window punya bayangan di tiap jalan.

Hotelling T2 umum:

$$
T^2=\sum_{a=1}^{A}\frac{t_a^2}{\lambda_a}
$$

Arti praktisnya: setiap score dikuadratkan, lalu dibagi eigenvalue PC masing-masing, lalu dijumlahkan.

Tabel langkah T2 multi-PC:

| PC | Score | Eigenvalue | Bagian T2 |
|---:|---:|---:|---:|
| 1 | $t_1$ | $\lambda_1$ | `t1^2 / lambda_1` |
| 2 | $t_2$ | $\lambda_2$ | `t2^2 / lambda_2` |
| A | $t_A$ | $\lambda_A$ | `tA^2 / lambda_A` |

Rekonstruksi umum:

$$
\hat z=tP_A^\top
$$

Arti simbol:

| Simbol | Arti |
|---|---|
| $P_A$ | matrix loading dari PC yang disimpan |
| $P_A^\top$ | transpose dari matrix loading |
| $\hat z$ | rekonstruksi dari PC yang disimpan |

Q/SPE umum:

$$
Q=\mathrm{SPE}=\sum_{j=1}^{p}(z_j-\hat z_j)^2
$$

Arti simbol:

| Simbol | Arti |
|---|---|
| $j$ | nomor fitur |
| $z_j$ | z-score asli fitur ke-j |
| $\hat z_j$ | rekonstruksi fitur ke-j |

Bahasa bayinya: walaupun PC lebih dari satu, Q tetap bertanya hal yang sama. Setelah window digambar ulang dari semua PC yang disimpan, masih ada sisa tidak?

Untuk belajar kertas, jangan mulai dari multi-PC. Kuasai 2 fitur dan 1 PC dulu. Setelah itu, multi-PC hanya pengulangan pola yang sama.

Kalau sudah paham bagian ini, kamu harus bisa melihat bahwa rumus multi-PC hanya mengulang pola 1 PC beberapa kali.

## Bagian 22. Worksheet lengkap training normal

### Visual atlas: worksheet training sebagai jalur kerja

Bagian ini panjang, jadi gampang tersesat. Pakai satu gambar dulu sebagai peta. Worksheet training adalah sebuah **jalur kerja**, seperti ban berjalan di pabrik: data normal masuk di ujung depan, lalu melewati stasiun demi stasiun sampai keluar menjadi threshold.

#### 1. Lihat jalurnya

![Diagram jalur kerja worksheet training normal: urut dari nama fitur, data normal, mean std, z-score, correlation, PC1, score t, rekonstruksi, residual, T2 dan Q, sampai threshold, dengan peringatan kunci urutan fitur E F](assets/pca-t2-q-manual/training-normal-worksheet-flow.png)

Ada sebelas stasiun bernomor, mengular dari kiri ke kanan. Pita teal di tengah menegaskan: ini worksheet training normal, jadi semua window harus normal.

#### 2. Bagian depan jalur

Stasiun 1 dan 2 menyiapkan bahan: beri nama fitur `[E, F]` dengan jelas, lalu tulis data normal mentah. Tanpa nama dan urutan fitur yang rapi, stasiun berikutnya bisa salah baca.

#### 3. Bagian adil dan hubungan

Stasiun 3 sampai 5 membuat fitur adil lalu membaca hubungannya: hitung mean dan standar deviasi, ubah menjadi z-score, lalu hitung correlation. Di sinilah fitur beda satuan dibuat sebanding.

#### 4. Bagian PCA dan skor

Stasiun 6 dan 7 mencari jalan normal: tentukan PC1 dan eigenvalue, lalu proyeksikan tiap window menjadi score `t`. Jalan normal lahir di sini.

#### 5. Bagian sisa dan skor akhir

Stasiun 8 sampai 10 menghitung sisa dan skor: gambar ulang `z_hat`, hitung residual `e`, lalu hitung T2 dan Q untuk tiap window normal.

#### 6. Jembatan ke rumus dan pagar

Stasiun 11 memasang pagar dari skor normal. Rumus inti sepanjang jalur:

$$
z=\frac{x-\mu}{s},\quad t=zp_1,\quad T^2=\frac{t^2}{\lambda_1},\quad Q=\sum_j e_j^2
$$

Pita oranye di bawah adalah peringatan keras: kunci urutan fitur `[E, F]`. Kalau urutan berubah, semua stasiun setelahnya ikut salah.

#### 7. Self check

Tutup gambar, jawab cepat:

1. Kenapa worksheet ini hanya boleh diisi window normal?
2. Stasiun mana yang membuat fitur beda satuan jadi sebanding?
3. Di stasiun mana jalan normal PC1 lahir?
4. Kenapa urutan fitur `[E, F]` wajib dikunci?

Kalau lancar, lanjut mengisi worksheet di bawah.

Salin bagian ini ke kertas kalau ingin latihan dari data lain. Worksheet ini sengaja panjang agar kamu tidak perlu menebak kolom apa yang harus ditulis.

### 22.1 Template penamaan fitur

Sebelum hitung angka, beri nama fitur dengan jelas.

| Nomor fitur | Nama fitur manusia | Simbol pendek | Satuan | Asal dari spektrum | DC dipakai? |
|---:|---|---|---|---|---|
| 1 |  |  |  |  | ya atau tidak |
| 2 |  |  |  |  | ya atau tidak |

Contoh untuk catatan ini:

| Nomor fitur | Nama fitur manusia | Simbol pendek | Satuan | Asal dari spektrum | DC dipakai? |
|---:|---|---|---|---|---|
| 1 | energi band getaran | $E$ | unit energi | jumlah `M[k]^2` pada band pilihan | tidak |
| 2 | frekuensi dominan | $F$ | Hz | frekuensi dengan magnitude terbesar | tidak |

Kenapa penamaan fitur perlu rapi? Karena salah urutan fitur membuat PCA salah baca. Kalau training memakai urutan `[E, F]`, inference juga harus memakai urutan `[E, F]`.

### 22.2 Data fitur mentah

| Window | Fitur 1 | Fitur 2 |
|---:|---:|---:|
| W1 |  |  |
| W2 |  |  |
| W3 |  |  |
| W4 |  |  |
| W5 |  |  |

Catatan sebelum lanjut:

| Cek | Jawaban |
|---|---|
| $n$ jumlah window training |  |
| semua window normal? |  |
| urutan fitur sudah tetap? |  |
| satuan fitur sudah jelas? |  |

### 22.3 Mean dan standar deviasi fitur 1

| Window | x | x - mu | (x - mu)^2 |
|---:|---:|---:|---:|
| W1 |  |  |  |
| W2 |  |  |  |
| W3 |  |  |  |
| W4 |  |  |  |
| W5 |  |  |  |
| Jumlah |  |  |  |

$$
\mu_1=\frac{\sum x}{n}=\underline{\hspace{3cm}}
$$

$$
s_1^2=\frac{\sum(x-\mu_1)^2}{n-1}=\underline{\hspace{3cm}}
$$

$$
s_1=\sqrt{s_1^2}=\underline{\hspace{3cm}}
$$

### 22.4 Mean dan standar deviasi fitur 2

| Window | x | x - mu | (x - mu)^2 |
|---:|---:|---:|---:|
| W1 |  |  |  |
| W2 |  |  |  |
| W3 |  |  |  |
| W4 |  |  |  |
| W5 |  |  |  |
| Jumlah |  |  |  |

$$
\mu_2=\frac{\sum x}{n}=\underline{\hspace{3cm}}
$$

$$
s_2^2=\frac{\sum(x-\mu_2)^2}{n-1}=\underline{\hspace{3cm}}
$$

$$
s_2=\sqrt{s_2^2}=\underline{\hspace{3cm}}
$$

### 22.5 Z-score training

| Window |  x1 | z1 = (x1 - mu_1) / s_1 |  x2 | z2 = (x2 - mu_2) / s_2 |
| -----: | --: | ---------------------: | --: | ---------------------: |
|     W1 |     |                        |     |                        |
|     W2 |     |                        |     |                        |
|     W3 |     |                        |     |                        |
|     W4 |     |                        |     |                        |
|     W5 |     |                        |     |                        |

### 22.6 Cek z-score

Isi setelah tabel z-score selesai.

| Cek | Rumus kecil | Hasil | Harapan |
|---|---|---:|---|
| mean z1 | `sum(z1) / n` |  | mendekati 0 |
| mean z2 | `sum(z2) / n` |  | mendekati 0 |
| sample variance z1 | `sum(z1^2) / (n - 1)` |  | mendekati 1 |
| sample variance z2 | `sum(z2^2) / (n - 1)` |  | mendekati 1 |

Kalau hasil tidak mendekati harapan, biasanya ada salah hitung mean, standar deviasi, atau pembulatan terlalu kasar.

### 22.7 Correlation

| Window | z1 | z2 | z1 * z2 |
|---:|---:|---:|---:|
| W1 |  |  |  |
| W2 |  |  |  |
| W3 |  |  |  |
| W4 |  |  |  |
| W5 |  |  |  |
| Jumlah |  |  |  |

$$
r=\frac{\sum z_1z_2}{n-1}=\underline{\hspace{3cm}}
$$

$$
S=\begin{bmatrix}1 & r \\ r & 1\end{bmatrix}=\underline{\hspace{3cm}}
$$

Interpretasi tanda:

| Jika $r$ | Tulis interpretasi |
|---|---|
| positif | dua fitur cenderung naik bersama |
| negatif | satu fitur naik saat fitur lain turun |
| dekat nol | tidak ada arah garis linear yang kuat |

### 22.8 PC1

$$
p_1=[\underline{\hspace{1.5cm}},\underline{\hspace{1.5cm}}]
$$

$$
\lambda_1=\underline{\hspace{3cm}}
$$

$$
\text{variance PC1 dalam kasus dua fitur}=\frac{\lambda_1}{2}=\underline{\hspace{3cm}}
$$

Untuk kasus dua fitur naik bersama, biasanya mulai dengan:

$$
p_1=[0.707,0.707]
$$

$$
\lambda_1=1+r
$$

Untuk kasus satu fitur naik dan satu turun, arah yang masuk akal bisa:

$$
p_1=[0.707,-0.707]
$$

$$
\lambda_1=1+|r|
$$

Pilih arah sesuai tanda hubungan. Jangan pakai arah naik bersama kalau correlation jelas negatif.

### 22.9 Score, rekonstruksi, residual, T2, Q

| Window | z1 | z2 | t | z_hat_1 | z_hat_2 | e1 | e2 | e1^2 | e2^2 | T2 | Q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 |  |  |  |  |  |  |  |  |  |  |  |
| W2 |  |  |  |  |  |  |  |  |  |  |  |
| W3 |  |  |  |  |  |  |  |  |  |  |  |
| W4 |  |  |  |  |  |  |  |  |  |  |  |
| W5 |  |  |  |  |  |  |  |  |  |  |  |

Rumus yang dipakai:

$$
t=zp_1
$$

$$
\hat z=tp_1^\top
$$

$$
e=z-\hat z
$$

$$
T^2=\frac{t^2}{\lambda_1}
$$

$$
Q=e_1^2+e_2^2
$$

### 22.10 Threshold

$$
\text{threshold}_{T^2}=\max(T^2\ \text{training normal})=\underline{\hspace{3cm}}
$$

$$
\text{threshold}_Q=\max(Q\ \text{training normal})=\underline{\hspace{3cm}}
$$

Catatan tanda:

$$
\text{anomali jika skor}>\text{threshold}
$$

Kalau skor sama persis dengan threshold, aturan catatan ini belum menyebutnya melewati threshold.

## Bagian 23. Worksheet lengkap inference window baru

Gunakan worksheet ini setelah model PCA manual selesai dihitung. Jangan menghitung mean, standar deviasi, PC1, eigenvalue, atau threshold dari window baru.

### 23.1 Salin parameter training

Isi tabel ini sebelum menghitung inference.

| Parameter | Nilai training yang disalin | Dari bagian mana |
|---|---:|---|
| $\mu_1$ |  | mean fitur 1 |
| $s_1$ |  | standar deviasi fitur 1 |
| $\mu_2$ |  | mean fitur 2 |
| $s_2$ |  | standar deviasi fitur 2 |
| $p_{1,1}$ |  | komponen pertama PC1 |
| $p_{1,2}$ |  | komponen kedua PC1 |
| $\lambda_1$ |  | eigenvalue PC1 |
| $\text{threshold}_{T^2}$ |  | ambang T2 |
| $\text{threshold}_Q$ |  | ambang Q |
| urutan fitur |  | contoh `[E, F]` |

Checklist sebelum lanjut:

| Cek | Ya atau tidak |
|---|---|
| parameter disalin dari training normal |  |
| urutan fitur inference sama dengan training |  |
| tanda keputusan memakai $>$ |  |
| pembulatan dicatat |  |

### 23.2 Fitur mentah window baru

| Window | Fitur 1 | Fitur 2 | Catatan rasa awal |
|---|---:|---:|---|
| X_new |  |  |  |

### 23.3 Z-score pakai statistik training

$$
z_{1,\text{new}}=\frac{x_{1,\text{new}}-\mu_1}{s_1}=\underline{\hspace{3cm}}
$$

$$
z_{2,\text{new}}=\frac{x_{2,\text{new}}-\mu_2}{s_2}=\underline{\hspace{3cm}}
$$

$$
z_{\text{new}}=[\underline{\hspace{1.5cm}},\underline{\hspace{1.5cm}}]
$$

Tabel arti z-score:

| Fitur | z-score | Arti bayi |
|---|---:|---|
| fitur 1 |  |  |
| fitur 2 |  |  |

### 23.4 Projection

$$
t_{\text{new}}=z_{\text{new}}p_1=z_{1,\text{new}}p_{1,1}+z_{2,\text{new}}p_{1,2}=\underline{\hspace{3cm}}
$$

Arti $t_{\text{new}}$:

| Jika $t_{\text{new}}$ | Rasa |
|---|---|
| negatif besar | jauh di sisi rendah PC1 |
| dekat nol | dekat pusat PC1 |
| positif besar | jauh di sisi tinggi PC1 |

### 23.5 T2

$$
T^2_{\text{new}}=\frac{t_{\text{new}}^2}{\lambda_1}=\underline{\hspace{3cm}}
$$

Bandingkan:

$$
T^2_{\text{new}}>\text{threshold}_{T^2}\ ?
$$

Isi:

| Nilai | Angka |
|---|---:|
| $T^2_{\text{new}}$ |  |
| $\text{threshold}_{T^2}$ |  |
| $T^2_{\text{new}}>\text{threshold}_{T^2}$ | True atau False |

### 23.6 Reconstruction dan residual

$$
\hat z_{\text{new}}=t_{\text{new}}p_1^\top=[\underline{\hspace{1.5cm}},\underline{\hspace{1.5cm}}]
$$

$$
e_{\text{new}}=z_{\text{new}}-\hat z_{\text{new}}=[\underline{\hspace{1.5cm}},\underline{\hspace{1.5cm}}]
$$

Tabel residual:

| Fitur | z asli | z_hat | residual e | e^2 |
|---|---:|---:|---:|---:|
| fitur 1 |  |  |  |  |
| fitur 2 |  |  |  |  |

### 23.7 Q/SPE

$$
Q_{\text{new}}=e_{1,\text{new}}^2+e_{2,\text{new}}^2=\underline{\hspace{3cm}}
$$

Bandingkan:

$$
Q_{\text{new}}>\text{threshold}_Q\ ?
$$

Isi:

| Nilai | Angka |
|---|---:|
| $Q_{\text{new}}$ |  |
| $\text{threshold}_Q$ |  |
| $Q_{\text{new}}>\text{threshold}_Q$ | True atau False |

### 23.8 Decision table dengan score_T2 dan score_Q

Selain keputusan True atau False, tulis skor relatifnya.

$$
\text{score}_{T^2}=\frac{T^2_{\text{new}}}{\text{threshold}_{T^2}}
$$

$$
\text{score}_Q=\frac{Q_{\text{new}}}{\text{threshold}_Q}
$$

$$
\text{score}=\max(\text{score}_{T^2},\text{score}_Q)
$$

| Nilai | Rumus | Hasil | Arti |
|---|---|---:|---|
| $\text{score}_{T^2}$ | `T2_new / threshold_T2` |  | lebih dari 1 berarti T2 lewat |
| $\text{score}_Q$ | `Q_new / threshold_Q` |  | lebih dari 1 berarti Q lewat |
| $\text{score}$ | `max(score_T2, score_Q)` |  | skor ringkas paling besar |

Keputusan akhir:

| Pertanyaan | Jawaban |
|---|---|
| `T2_new > threshold_T2`? |  |
| `Q_new > threshold_Q`? |  |
| $\mathrm{is\_anomaly}$? |  |
| penyebab utama | T2 atau Q |
| kalimat alasan |  |

Tulis alasan singkat:

```text
Window ini normal atau anomali karena:
```

## Bagian 24. Kesalahan umum dan cara memperbaikinya

Gunakan bagian ini seperti tabel diagnosa. Cari gejala yang mirip dengan hasil hitunganmu.

| Kesalahan | Gejala | Cara memperbaiki |
|---|---|---|
| Mean dihitung dari semua data, termasuk window baru | skor terlihat terlalu jinak | hitung mean hanya dari training normal |
| Standar deviasi dihitung ulang saat inference | window baru menormalisasi dirinya sendiri | pakai standar deviasi training |
| Lupa scaling | fitur angka besar mendominasi PCA | hitung z-score dulu |
| Memakai $n$ untuk sample variance | standar deviasi sedikit berbeda | untuk latihan sample pakai $n-1$ |
| Menukar urutan fitur | hasil dot product terasa aneh | samakan urutan training dan inference |
| Salah tanda $p_1$ | score $t$ berbalik tanda | cek apakah fitur naik bersama atau berlawanan |
| Salah tanda residual | Q tetap sama jika hanya tanda terbalik, tetapi kontribusi fitur bisa salah | pakai `e = z - z_hat` secara konsisten |
| Q dijumlahkan tanpa kuadrat | residual positif dan negatif saling hapus | pakai jumlah kuadrat residual |
| Memakai semua PC pada contoh dua fitur | Q menjadi nol atau sangat kecil | simpan 1 PC untuk latihan Q/SPE |
| Lupa membagi T2 dengan eigenvalue | T2 terlalu besar atau tidak sebanding | pakai `T2 = t^2 / lambda_1` |
| Threshold diambil dari test set | evaluasi bocor | threshold dari training normal atau validation normal |
| Memakai $\ge$ padahal metodologi menulis $>$ | keputusan beda saat skor sama dengan threshold | pilih satu aturan dan tulis konsisten |
| Pembulatan terlalu cepat | hasil akhir beda cukup besar | simpan angka antara dengan lebih banyak digit |
| Mengira `0.000` berarti nol persis | Q kecil terlihat hilang | tulis enam desimal untuk residual kuadrat kecil |
| Mengira SKAB adalah data PDAM nyata | laporan menjadi salah konteks | tulis SKAB sebagai surrogate water circulation testbed |
| Menganggap dominant frequency selalu cukup | informasi energi band bisa hilang | pakai beberapa fitur spektral saat proyek nyata |
| Memasukkan DC bin tanpa sadar | fitur bisa lebih membaca offset daripada osilasi | putuskan sejak awal DC dipakai atau diabaikan |

Tabel diagnosa cepat:

| Yang terasa salah | Cek pertama | Cek kedua |
|---|---|---|
| semua z-score besar | mean atau standar deviasi salah | satuan fitur tertukar |
| correlation lebih dari 1 | pembagi salah | z-score tidak memakai sample std |
| panjang $p_1$ bukan 1 | angka loading salah | pembulatan terlalu kasar |
| T2 negatif | ada rumus salah | T2 harus dari kuadrat |
| Q negatif | ada rumus salah | Q harus dari kuadrat residual |
| Q training semua nol | mungkin semua PC dipakai | cek jumlah PC yang disimpan |
| inference terlalu normal | scaler dihitung ulang | threshold bocor dari test |

## Bagian 25. Cara mengecek jawaban sendiri

Gunakan daftar ini setelah selesai menghitung. Tutup rumus utama sebentar, lalu cek dari hasilmu sendiri.

### 25.1 Cek training

| Cek | Harusnya | Status |
|---|---|---|
| mean z-score tiap fitur | mendekati 0 |  |
| sample variance z-score tiap fitur | mendekati 1 |  |
| panjang $p_1$ | mendekati 1 |  |
| $\lambda_1$ untuk kasus `[1 r; r 1]` | `1 + r` jika $r$ positif |  |
| variance PC1 | `lambda_1 / 2` untuk dua fitur |  |
| T2 tidak negatif | selalu benar karena pakai kuadrat |  |
| Q tidak negatif | selalu benar karena pakai kuadrat residual |  |
| threshold_T2 | sama dengan nilai T2 training terbesar jika pakai max |  |
| threshold_Q | sama dengan nilai Q training terbesar jika pakai max |  |

### 25.2 Cek inference

| Cek | Harusnya | Status |
|---|---|---|
| z-score window baru | pakai mean dan standar deviasi training |  |
| $t_{\text{new}}$ | hasil dot product `z_new dot p1` |  |
| `z_hat_new` | sejajar dengan PC1 |  |
| residual | `z_new - z_hat_new` |  |
| T2 decision | pakai `T2_new > threshold_T2` |  |
| Q decision | pakai `Q_new > threshold_Q` |  |
| keputusan | memakai OR antara T2 dan Q |  |
| score gabungan | lebih dari 1 jika salah satu skor lewat |  |

### 25.3 Uji tutup catatan

Tutup catatan ini, lalu jawab dari ingatan. Buka lagi untuk memeriksa.

| Pertanyaan | Jawabanmu |
|---|---|
| Apa arti $n$? |  |
| Apa arti $N$? |  |
| Kenapa varians sample memakai $n-1$? |  |
| Apa arti z-score `2`? |  |
| Apa arti vector `[E, F]`? |  |
| Apa itu dot product? |  |
| Apa arti $r$ positif? |  |
| Apa arti `p1 = [0.707, 0.707]`? |  |
| Apa arti $\lambda_1$ besar? |  |
| Apa beda T2 dan Q? |  |
| Kenapa window $X_{\text{new}}$ anomali? |  |
| Kenapa SKAB tidak boleh disebut data PDAM nyata? |  |

Kunci pendek:

| Pertanyaan | Jawaban pendek |
|---|---|
| $n$ | jumlah window training |
| $N$ | jumlah sample dalam window |
| $n-1$ | pembagi sample variance |
| z-score `2` | dua standar deviasi di atas mean training |
| vector `[E, F]` | kotak berurutan, energi lalu frekuensi |
| dot product | kali pasangan lalu jumlah |
| $r$ positif | dua fitur naik turun bersama |
| `p1 = [0.707, 0.707]` | arah energi dan frekuensi naik bersama |
| $\lambda_1$ besar | variasi normal besar di arah PC1 |
| T2 | jauh tidak di jalan normal |
| Q | keluar tidak dari jalan normal |
| $X_{\text{new}}$ | Q besar karena E tinggi tetapi F tidak ikut tinggi |
| SKAB | surrogate water circulation testbed |

## Bagian 26. Intuisi gambar tanpa menggambar rumit

Untuk dua fitur, bayangkan bidang datar.

```text
z_F
 ^
 |
 |          titik normal tinggi
 |        /
 |      /     garis PC1
 |    /
 |  / titik normal rendah
 +----------------------> z_E
```

Garis PC1 miring karena energi dan frekuensi naik bersama.

T2 melihat jarak sepanjang garis. Q melihat jarak tegak lurus dari garis.

### 26.1 Titik normal

Window normal rendah seperti W1 berada di kiri bawah. Window normal tinggi seperti W5 berada di kanan atas. Keduanya masih dekat garis.

```text
rendah bersama -> dekat garis
tinggi bersama -> dekat garis
```

Karena masih dekat garis, Q kecil. Kalau titik berada jauh di ujung garis, T2 bisa lebih besar, tetapi Q tetap kecil.

### 26.2 Titik $X_{\text{new}}$

Contoh window baru $X_{\text{new}}$:

```text
E tinggi, F tidak tinggi
```

Titiknya berada jauh ke kanan, tetapi tidak naik mengikuti garis. Ia jauh dari garis PC1. Karena itu Q besar.

Gambar rasa:

```text
z_F
 ^
 |
 |          garis PC1 naik
 |        /
 |      /
 |    /                 X_new ada kanan tetapi rendah
 |  /                         x
 +----------------------> z_E
```

Bahasa pompa:

```text
normal: energi naik, frekuensi ikut naik
X_new : energi naik, frekuensi tidak ikut
```

Model tidak hanya melihat energi tinggi. Model melihat hubungan normal antar fitur rusak.

### 26.3 Cara menjelaskan ke orang non teknis

Kalimat satu napas:

```text
PCA belajar jalan normal dari window sehat. T2 mengecek apakah window terlalu jauh di jalan itu. Q mengecek apakah window keluar dari jalan itu. Pada contoh ini, window keluar dari jalan karena energi naik tanpa frekuensi ikut naik.
```

Kalau sudah paham bagian ini, kamu harus bisa menjelaskan PCA T2/Q dengan gambar garis sederhana.

## Bagian 27. Ringkasan angka contoh utama

Sebelum membaca tabel ringkasan, ingat kebijakan pembulatan:

```text
angka tabel = angka latihan tangan yang dibulatkan
angka kode  = sebaiknya disimpan dengan presisi lebih banyak
```

Nilai `p1 = [0.707, 0.707]` adalah bentuk pendek dari `[1/sqrt(2), 1/sqrt(2)]`. Nilai `r = 0.971` adalah bentuk pendek dari sekitar `0.9707`. Perbedaan kecil di digit belakang tidak mengubah cerita utama contoh ini.

### 27.1 Training normal

| Window | $E$ | $F$ | $z_E$ | $z_F$ | $t$ | $T^2$ | $Q$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | 8 | 29 | -1.265 | -1.228 | -1.762 | 1.575 | 0.001 |
| W2 | 9 | 30 | -0.632 | -0.351 | -0.695 | 0.245 | 0.040 |
| W3 | 10 | 30 | 0.000 | -0.351 | -0.248 | 0.031 | 0.062 |
| W4 | 11 | 31 | 0.632 | 0.526 | 0.819 | 0.340 | 0.006 |
| W5 | 12 | 32 | 1.265 | 1.403 | 1.886 | 1.805 | 0.010 |

Catatan Q kecil:

| Window | Q tiga desimal | Cara membaca |
|---:|---:|---|
| W1 | 0.001 | sangat kecil, bukan nol persis |
| W3 | 0.062 | Q training terbesar di contoh ini |
| W5 | 0.010 | kecil dibanding $Q_{\text{new}}$ |

### 27.2 Parameter yang disimpan

| Nama | Nilai |
|---|---:|
| `mu_E` | 10 |
| `s_E` | 1.581 |
| `mu_F` | 30.4 |
| `s_F` | 1.140 |
| $r$ | 0.971 |
| $p_1$ | `[0.707, 0.707]` |
| $\lambda_1$ | 1.971 |
| variance PC1 | sekitar 98.5 persen dari total dua fitur |
| $\text{threshold}_{T^2}$ | 1.805 |
| $\text{threshold}_Q$ | 0.062 |

Parameter ini harus dibawa ke inference. Jangan dihitung ulang dari window baru.

### 27.3 Inference

| Window | $E$ | $F$ | $z_E$ | $z_F$ | $t$ | $T^2$ | $Q$ | Keputusan |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| X_new | 14 | 30 | 2.530 | -0.351 | 1.541 | 1.205 | 4.150 | anomali |

Decision table:

| Skor | Nilai | Threshold | Lewat? | Score relatif |
|---|---:|---:|---|---:|
| $T^2$ | 1.205 | 1.805 | False | 0.668 |
| $Q$ | 4.150 | 0.062 | True | 66.935 |

Alasan:

$$
T^2_{\text{new}}\le\text{threshold}_{T^2}
$$

$$
Q_{\text{new}}>\text{threshold}_Q
$$

Karena keputusan memakai OR, window adalah anomali.

### 27.4 Catatan presisi untuk laporan

Jika menulis laporan, boleh memakai angka tiga desimal untuk tabel utama. Tetapi tulis satu catatan seperti ini:

```text
Angka pada contoh manual dibulatkan untuk keterbacaan. Implementasi numerik menyimpan presisi floating point penuh sehingga hasil threshold dan skor tidak bergantung pada pembulatan tabel.
```

Catatan ini mencegah pembaca bingung saat mereka menghitung ulang dengan kalkulator dan mendapat digit belakang yang sedikit berbeda.

## Bagian 28. Cheat sheet terakhir

### 28.1 Urutan lengkap dari sensor sampai keputusan

```text
1. ambil time-series sensor pompa
2. potong menjadi window
3. catat N dan f_s
4. durasi window = N / f_s
5. hitung atau terima magnitude FFT M[k]
6. ubah bin menjadi frekuensi f_k = k f_s / N
7. abaikan atau pakai DC bin sesuai keputusan fitur
8. hitung fitur spektral, misalnya E_band dan f_dom
9. satu window menjadi satu baris fitur
10. hitung mean dan standar deviasi dari training normal
11. ubah fitur menjadi z-score dengan penggaris training
12. hitung correlation matrix S
13. tentukan PC1 p_1 dan eigenvalue lambda_1
14. proyeksikan window ke PC1 untuk mendapat t
15. hitung T2 = t^2 / lambda_1
16. rekonstruksi z_hat = t p_1
17. hitung residual e = z - z_hat
18. hitung Q = sum(e_j^2)
19. bandingkan T2 dan Q dengan threshold memakai tanda >
20. putuskan normal atau anomali dengan aturan OR
```

### 28.2 Fitur spektral

$$
\text{durasi window}=\frac{N}{f_s}
$$

$$
f_k=\frac{kf_s}{N}
$$

$$
E_{\text{band}}=\sum_{k\in\text{band}}M[k]^2
$$

$$
f_{\text{dom}}=f_k\ \text{dengan}\ M[k]\ \text{terbesar}
$$

$$
f_{\text{centroid}}=\frac{\sum_k f_kM[k]}{\sum_k M[k]}
$$

### 28.3 Scaling

$$
\mu=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

$$
s=\sqrt{\frac{\sum_{i=1}^{n}(x_i-\mu)^2}{n-1}}
$$

$$
z_i=\frac{x_i-\mu}{s}
$$

Ingat:

```text
mu dan s inference = mu dan s training normal
```

### 28.4 Correlation dua fitur z-score

$$
r=\frac{\sum_{i=1}^{n}z_{1,i}z_{2,i}}{n-1}
$$

$$
S=\begin{bmatrix}1 & r \\ r & 1\end{bmatrix}
$$

| $r$ | Arti |
|---:|---|
| positif | dua fitur naik bersama |
| negatif | satu fitur naik, satu turun |
| dekat nol | tidak ada garis utama yang jelas |

### 28.5 PCA 1 PC untuk dua fitur naik bersama

$$
p_1=[0.707,0.707]
$$

$$
\lambda_1=1+r
$$

$$
t=zp_1
$$

$$
\hat z=tp_1^\top
$$

$$
e=z-\hat z
$$

Catatan:

$$
0.707=\frac{1}{\sqrt{2}}\ \text{yang dibulatkan}
$$

### 28.6 T2 dan Q

$$
T^2=\frac{t^2}{\lambda_1}
$$

$$
Q=\sum_{j=1}^{p}e_j^2
$$

Untuk dua fitur:

$$
Q=e_1^2+e_2^2
$$

### 28.7 Threshold dan keputusan

$$
\text{threshold}_{T^2}=\max(T^2\ \text{normal})\ \text{atau}\ p95(T^2\ \text{validation normal})
$$

$$
\text{threshold}_Q=\max(Q\ \text{normal})\ \text{atau}\ p95(Q\ \text{validation normal})
$$

$$
\mathrm{is\_anomaly}=(T^2>\text{threshold}_{T^2})\ \mathrm{OR}\ (Q>\text{threshold}_Q)
$$

$$
\text{score}_{T^2}=\frac{T^2}{\text{threshold}_{T^2}}
$$

$$
\text{score}_Q=\frac{Q}{\text{threshold}_Q}
$$

$$
\text{score}=\max(\text{score}_{T^2},\text{score}_Q)
$$

Tanda perbandingan di catatan ini:

```text
pakai >
bukan >=
```

### 28.8 Arti cepat

| Skor | Kalau tinggi berarti | Pertanyaan bayi |
|---|---|---|
| T2 | window ekstrem di arah pola normal PCA | jauh tidak di jalan normal? |
| Q/SPE | window sulit direkonstruksi oleh pola normal PCA | keluar tidak dari jalan normal? |

### 28.9 Kalimat siap pakai untuk laporan

PCA T2/Q spectral pada PDAM Pump Sentinel membaca window telemetry yang sudah diringkas menjadi fitur spektral. Fitur tersebut di-scaling memakai statistik normal training, lalu diproyeksikan ke subspace PCA. Hotelling T2 mengukur jarak window di arah principal component yang disimpan, sedangkan Q atau SPE mengukur jumlah kuadrat residual yang tidak bisa direkonstruksi oleh PCA. Window dianggap anomali jika T2 atau Q lebih besar dari threshold yang dikalibrasi dari data normal. Pada proyek ini, SKAB dipakai sebagai surrogate water circulation testbed untuk pembelajaran dan demo akademik, bukan sebagai data operasional PDAM nyata.

## Bagian 29. Menghitung bareng dari awal sampai vonis

Contoh utama di Bagian 10 sampai 19 ditulis sambil mengajar konsep, jadi terasa banyak teori di antara angka. Bagian ini berbeda. Di sini kita tidak menjelaskan konsep lagi. Kita hanya duduk bersama, ambil satu soal baru, lalu mengerjakannya berurutan sampai keluar keputusan.

Anggap ini seperti memasak sambil ditemani. Satu langkah, satu hasil. Jangan lompat.

Aturan main bagian ini:

| Aturan | Maksud |
|---|---|
| ikuti urut dari atas ke bawah | jangan lompat langkah |
| satu langkah, satu angka | tulis hasil tiap langkah di kertas |
| rumus lihat cheat sheet Bagian 28 | tidak usah menghafal, cukup ikuti |
| statistik training tidak dihitung ulang saat inference | pakai mean dan standar deviasi training |

Pembulatan sama seperti Bagian 10.0, yaitu tiga desimal. Kadang hasil tangan beda satu digit terakhir karena pembulatan. Itu wajar, jangan panik.

Data baru yang kita kerjakan bersama. Lima window normal, dua fitur:

| Window normal | E energi band | F frekuensi dominan |
|---:|---:|---:|
| W1 | 20 | 40 |
| W2 | 21 | 42 |
| W3 | 22 | 43 |
| W4 | 23 | 44 |
| W5 | 24 | 46 |

Window baru yang akan diuji nanti:

| Window baru | E | F |
|---|---:|---:|
| X_new | 26 | 42 |

Rasa awal: energi `E` naik tinggi ke `26`, tetapi frekuensi `F` hanya `42`, tidak ikut naik tinggi. Kalau pola normal naik bersama, ini calon anomali yang akan ditangkap Q. Kita buktikan dengan hitungan, bukan dengan tebakan.

### 29.0 Rumus dasar yang dipakai

Semua rumus untuk sepuluh langkah di bawah dikumpulkan di sini, jadi kamu tidak perlu scroll ke atas. Untuk contoh ini: jumlah window `n = 5`, jadi `n - 1 = 4`, dan ada dua fitur yaitu `E` dan `F`.

Bekal dasar tiap fitur:

$$
\mu=\frac{\mathrm{sum}}{n},\quad d=x-\mu,\quad s^2=\frac{\sum d^2}{n-1},\quad s=\sqrt{s^2}
$$

$$
z=\frac{x-\mu}{s}
$$

Pola normal dari dua fitur:

$$
r=\frac{\sum_{i=1}^{n} z_{E,i}\,z_{F,i}}{n-1},\quad S=\begin{bmatrix}1 & r \\ r & 1\end{bmatrix}
$$

$$
p_1=[0.707,0.707],\quad \lambda_1=1+r
$$

Skor tiap window:

$$
t=z_E(0.707)+z_F(0.707)
$$

$$
\hat z_E=t(0.707),\quad \hat z_F=t(0.707)
$$

$$
e_E=z_E-\hat z_E,\quad e_F=z_F-\hat z_F
$$

$$
T^2=\frac{t^2}{\lambda_1},\quad Q=e_E^2+e_F^2
$$

Ambang dan vonis:

$$
\text{threshold}_{T^2}=\max(T^2\ \text{training}),\quad \text{threshold}_Q=\max(Q\ \text{training})
$$

$$
\mathrm{is\_anomaly}=(T^2>\text{threshold}_{T^2})\ \mathrm{OR}\ (Q>\text{threshold}_Q)
$$

$$
\text{score}=\max\left(\frac{T^2}{\text{threshold}_{T^2}},\frac{Q}{\text{threshold}_Q}\right)
$$

Arti simbol singkat:

| Simbol | Arti |
|---|---|
| $n$ | jumlah window training, di sini 5 |
| $x$ | nilai fitur mentah |
| $\mu$ | mean fitur dari training normal |
| $s$ | standar deviasi fitur dari training normal |
| $z$ | nilai fitur setelah z-score |
| $r$ | correlation antara z_E dan z_F |
| $p_1$ | arah PC1, di sini `[0.707, 0.707]` |
| $\lambda_1$ | eigenvalue PC1 |
| $t$ | score window pada PC1 |
| $\hat z$ | rekonstruksi z dari PC1 |
| $e$ | residual, yaitu `z - z_hat` |
| $T^2$ | Hotelling T2 |
| $Q$ | Q atau SPE, jumlah kuadrat residual |

Catatan: `0.707` adalah bentuk pendek dari `1 / sqrt(2)`. Untuk vonis pakai tanda `>`, bukan `>=`.

### 29.1 Langkah 1: mean dan standar deviasi training

Fitur E. Data `20, 21, 22, 23, 24`.

Kita jumlah dulu pelan-pelan, seperti menghitung kelereng.

```text
mulai 0
0 + 20 = 20
20 + 21 = 41
41 + 22 = 63
63 + 23 = 86
86 + 24 = 110
```

$$
\mathrm{sum}_E=20+21+22+23+24=110
$$

Sekarang bagi ke 5 data.

```text
110 / 5

20 x 5 = 100 (kurang)
23 x 5 = 115 (lebih)
22 x 5 = 110 (pas)

jadi 110 / 5 = 22
```

$$
\mu_E=\frac{110}{5}=22
$$

Deviation dan kuadratnya:

| Window | E | $d_E=E-22$ | cara kuadrat | $d_E^2$ |
|---:|---:|---:|---|---:|
| W1 | 20 | -2 | $(-2)(-2)=4$ | 4 |
| W2 | 21 | -1 | $(-1)(-1)=1$ | 1 |
| W3 | 22 | 0 | $(0)(0)=0$ | 0 |
| W4 | 23 | 1 | $(1)(1)=1$ | 1 |
| W5 | 24 | 2 | $(2)(2)=4$ | 4 |

Jumlah kuadratnya jangan lompat.

```text
mulai 0
0 + 4 = 4
4 + 1 = 5
5 + 0 = 5
5 + 1 = 6
6 + 4 = 10
```

$$
\sum d_E^2=4+1+0+1+4=10
$$

Sekarang variance sample. Karena data normal ada 5, pembaginya `5 - 1 = 4`.

```text
10 / 4

2 x 4 = 8 (kurang)
3 x 4 = 12 (lebih)
2.5 x 4 = 10 (pas)

jadi 10 / 4 = 2.5
```

Akar dari `2.5` kita raba dengan kuadrat.

```text
1^2 = 1 (kurang)
2^2 = 4 (lebih)
1.5^2 = 2.25 (kurang)
1.58^2 = 2.4964 (kurang sedikit)
1.581^2 = 2.499561 (pas 3 desimal ke 2.5)

jadi sqrt(2.5) = 1.581
```

$$
s_E^2=\frac{10}{4}=2.5,\quad s_E=\sqrt{2.5}=1.581
$$

Fitur F. Data `40, 42, 43, 44, 46`.

Jumlahkan dulu.

```text
mulai 0
0 + 40 = 40
40 + 42 = 82
82 + 43 = 125
125 + 44 = 169
169 + 46 = 215
```

$$
\mathrm{sum}_F=40+42+43+44+46=215
$$

Bagi ke 5 data.

```text
215 / 5

40 x 5 = 200 (kurang)
44 x 5 = 220 (lebih)
43 x 5 = 215 (pas)

jadi 215 / 5 = 43
```

$$
\mu_F=\frac{215}{5}=43
$$

| Window | F | $d_F=F-43$ | cara kuadrat | $d_F^2$ |
|---:|---:|---:|---|---:|
| W1 | 40 | -3 | $(-3)(-3)=9$ | 9 |
| W2 | 42 | -1 | $(-1)(-1)=1$ | 1 |
| W3 | 43 | 0 | $(0)(0)=0$ | 0 |
| W4 | 44 | 1 | $(1)(1)=1$ | 1 |
| W5 | 46 | 3 | $(3)(3)=9$ | 9 |

Jumlah kuadratnya:

```text
mulai 0
0 + 9 = 9
9 + 1 = 10
10 + 0 = 10
10 + 1 = 11
11 + 9 = 20
```

$$
\sum d_F^2=9+1+0+1+9=20
$$

Variance sample:

```text
20 / 4

4 x 4 = 16 (kurang)
6 x 4 = 24 (lebih)
5 x 4 = 20 (pas)

jadi 20 / 4 = 5
```

Akar dari `5` kita raba juga.

```text
2^2 = 4 (kurang)
3^2 = 9 (lebih)
2.2^2 = 4.84 (kurang)
2.23^2 = 4.9729 (kurang)
2.236^2 = 4.999696 (pas 3 desimal ke 5)

jadi sqrt(5) = 2.236
```

$$
s_F^2=\frac{20}{4}=5,\quad s_F=\sqrt{5}=2.236
$$

Simpan dulu:

| Fitur | Mean | Standar deviasi |
|---|---:|---:|
| E | 22 | 1.581 |
| F | 43 | 2.236 |

### 29.2 Langkah 2: ubah jadi z-score

Rumus `z = (x - mu) / s`. Pakai mean dan standar deviasi training tadi.

Untuk fitur E, pembilangnya adalah `E - 22`.

```text
W1: (20 - 22) / 1.581 = -2 / 1.581

1 x 1.581 = 1.581 (kurang)
2 x 1.581 = 3.162 (lebih)
1.265 x 1.581 = 1.999965 (pas ke 2.000)

jadi -2 / 1.581 = -1.265
```

```text
W2: (21 - 22) / 1.581 = -1 / 1.581

0.632 x 1.581 = 0.999192 (kurang)
0.633 x 1.581 = 1.000773 (lebih)
0.63251 x 1.581 = 0.999998 (pas ke 1.000)

jadi -1 / 1.581 = -0.633
```

```text
W3: (22 - 22) / 1.581 = 0 / 1.581 = 0.000
```

```text
W4: (23 - 22) / 1.581 = 1 / 1.581

0.632 x 1.581 = 0.999192 (kurang)
0.633 x 1.581 = 1.000773 (lebih)
0.63251 x 1.581 = 0.999998 (pas ke 1.000)

jadi 1 / 1.581 = 0.633
```

```text
W5: (24 - 22) / 1.581 = 2 / 1.581

1 x 1.581 = 1.581 (kurang)
2 x 1.581 = 3.162 (lebih)
1.265 x 1.581 = 1.999965 (pas ke 2.000)

jadi 2 / 1.581 = 1.265
```

Untuk fitur F, pembilangnya adalah `F - 43`.

```text
W1: (40 - 43) / 2.236 = -3 / 2.236

1 x 2.236 = 2.236 (kurang)
2 x 2.236 = 4.472 (lebih)
1.34168 x 2.236 = 2.999996 (pas ke 3.000)

jadi -3 / 2.236 = -1.342
```

```text
W2: (42 - 43) / 2.236 = -1 / 2.236

0.44 x 2.236 = 0.98384 (kurang)
0.45 x 2.236 = 1.0062 (lebih)
0.44723 x 2.236 = 1.000006 (pas ke 1.000)

jadi -1 / 2.236 = -0.447
```

```text
W3: (43 - 43) / 2.236 = 0 / 2.236 = 0.000
```

```text
W4: (44 - 43) / 2.236 = 1 / 2.236

0.44 x 2.236 = 0.98384 (kurang)
0.45 x 2.236 = 1.0062 (lebih)
0.44723 x 2.236 = 1.000006 (pas ke 1.000)

jadi 1 / 2.236 = 0.447
```

```text
W5: (46 - 43) / 2.236 = 3 / 2.236

1 x 2.236 = 2.236 (kurang)
2 x 2.236 = 4.472 (lebih)
1.34168 x 2.236 = 2.999996 (pas ke 3.000)

jadi 3 / 2.236 = 1.342
```

| Window | E | $z_E=(E-22)/1.581$ | F | $z_F=(F-43)/2.236$ |
|---:|---:|---:|---:|---:|
| W1 | 20 | -1.265 | 40 | -1.342 |
| W2 | 21 | -0.633 | 42 | -0.447 |
| W3 | 22 | 0.000 | 43 | 0.000 |
| W4 | 23 | 0.633 | 44 | 0.447 |
| W5 | 24 | 1.265 | 46 | 1.342 |

Cek cepat: rata-rata tiap kolom z mendekati 0. Aman.

### 29.3 Langkah 3: correlation r

Kalikan `z_E * z_F` tiap baris, lalu jumlahkan.

Untuk W1 dan W5, angkanya sama besar. Tanda negatif kali negatif menjadi positif.

```text
1.265 x 1.342
= 1.265 x (1 + 0.3 + 0.04 + 0.002)
= 1.265 + 0.3795 + 0.0506 + 0.00253
= 1.69763
= 1.698
```

Untuk W2 dan W4, angkanya juga sama besar.

```text
0.633 x 0.447
= 0.633 x (0.4 + 0.04 + 0.007)
= 0.2532 + 0.02532 + 0.004431
= 0.282951
= 0.283
```

Untuk W3, semuanya nol.

```text
0.000 x 0.000 = 0.000
```

| Window | $z_E$ | $z_F$ | $z_Ez_F$ |
|---:|---:|---:|---:|
| W1 | -1.265 | -1.342 | 1.698 |
| W2 | -0.633 | -0.447 | 0.283 |
| W3 | 0.000 | 0.000 | 0.000 |
| W4 | 0.633 | 0.447 | 0.283 |
| W5 | 1.265 | 1.342 | 1.698 |

Jumlahkan sambil jalan.

```text
mulai 0
0 + 1.698 = 1.698
1.698 + 0.283 = 1.981
1.981 + 0.000 = 1.981
1.981 + 0.283 = 2.264
2.264 + 1.698 = 3.962
```

$$
\sum z_Ez_F=1.698+0.283+0.000+0.283+1.698=3.962
$$

Sekarang bagi dengan `n - 1 = 4`.

```text
3.962 / 4

0.9 x 4 = 3.6 (kurang)
1.0 x 4 = 4.0 (lebih)
0.9905 x 4 = 3.962 (pas)

jadi 3.962 / 4 = 0.991
```

$$
r=\frac{3.962}{4}=0.991
$$

$$
S=\begin{bmatrix}1 & 0.991 \\ 0.991 & 1\end{bmatrix}
$$

E dan F naik turun bersama dengan sangat kuat.

### 29.4 Langkah 4: PC1 dan eigenvalue

Dua fitur naik bersama, jadi arah PC1:

Arah mentahnya gampang: kalau E naik dan F juga naik, arah kasarnya `[1, 1]`. Tapi PCA butuh panjang vektor menjadi 1.

```text
panjang [1, 1] = sqrt(1^2 + 1^2)
= sqrt(1 + 1)
= sqrt(2)

1.4^2 = 1.96 (kurang)
1.5^2 = 2.25 (lebih)
1.414^2 = 1.999396 (pas ke 2.000)

jadi panjangnya kira-kira 1.414
```

Sekarang bagi tiap komponen dengan panjang itu.

```text
1 / 1.414

0.7 x 1.414 = 0.9898 (kurang)
0.71 x 1.414 = 1.00394 (lebih)
0.707 x 1.414 = 0.999698 (pas ke 1.000)

jadi 1 / 1.414 = 0.707
```

$$
p_1=[0.707,0.707]
$$

Eigenvalue PC1 untuk dua fitur yang korelasinya `r` adalah `1 + r`.

```text
1 + 0.991 = 1.991
```

$$
\lambda_1=1+r=1+0.991=1.991
$$

### 29.5 Langkah 5: score t tiap window

Rumus `t = z_E*0.707 + z_F*0.707`.

Kita pecah `0.707` menjadi `0.7 + 0.007`.

```text
W1:
-1.265 x 0.707
= -(1.265 x 0.7 + 1.265 x 0.007)
= -(0.8855 + 0.008855)
= -0.894355

-1.342 x 0.707
= -(1.342 x 0.7 + 1.342 x 0.007)
= -(0.9394 + 0.009394)
= -0.948794

t = -0.894355 + -0.948794
t = -1.843149
t = -1.843
```

```text
W2:
-0.633 x 0.707
= -(0.633 x 0.7 + 0.633 x 0.007)
= -(0.4431 + 0.004431)
= -0.447531

-0.447 x 0.707
= -(0.447 x 0.7 + 0.447 x 0.007)
= -(0.3129 + 0.003129)
= -0.316029

t = -0.447531 + -0.316029
t = -0.763560
t = -0.764
```

```text
W3:
0.000 x 0.707 = 0.000
0.000 x 0.707 = 0.000
t = 0.000
```

```text
W4:
0.633 x 0.707
= 0.633 x 0.7 + 0.633 x 0.007
= 0.4431 + 0.004431
= 0.447531

0.447 x 0.707
= 0.447 x 0.7 + 0.447 x 0.007
= 0.3129 + 0.003129
= 0.316029

t = 0.447531 + 0.316029
t = 0.763560
t = 0.764
```

```text
W5:
1.265 x 0.707
= 1.265 x 0.7 + 1.265 x 0.007
= 0.8855 + 0.008855
= 0.894355

1.342 x 0.707
= 1.342 x 0.7 + 1.342 x 0.007
= 0.9394 + 0.009394
= 0.948794

t = 0.894355 + 0.948794
t = 1.843149
t = 1.843
```

| Window | $z_E$ | $z_F$ | $t$ |
|---:|---:|---:|---:|
| W1 | -1.265 | -1.342 | -1.843 |
| W2 | -0.633 | -0.447 | -0.764 |
| W3 | 0.000 | 0.000 | 0.000 |
| W4 | 0.633 | 0.447 | 0.764 |
| W5 | 1.265 | 1.342 | 1.843 |

### 29.6 Langkah 6: rekonstruksi dan residual

Rumus `z_hat = t * 0.707` untuk tiap fitur, lalu `e = z - z_hat`.

Karena PC1 punya dua angka yang sama, `z_hat_E` dan `z_hat_F` juga sama.

```text
W1:
-1.843 x 0.707
= -(1.843 x 0.7 + 1.843 x 0.007)
= -(1.2901 + 0.012901)
= -1.303001
= -1.303

e_E = -1.265 - (-1.303) = 0.038
e_F = -1.342 - (-1.303) = -0.039
```

```text
W2:
-0.764 x 0.707
= -(0.764 x 0.7 + 0.764 x 0.007)
= -(0.5348 + 0.005348)
= -0.540148
= -0.540

e_E = -0.633 - (-0.540) = -0.093
e_F = -0.447 - (-0.540) = 0.093
```

```text
W3:
0.000 x 0.707 = 0.000

e_E = 0.000 - 0.000 = 0.000
e_F = 0.000 - 0.000 = 0.000
```

```text
W4:
0.764 x 0.707
= 0.764 x 0.7 + 0.764 x 0.007
= 0.5348 + 0.005348
= 0.540148
= 0.540

e_E = 0.633 - 0.540 = 0.093
e_F = 0.447 - 0.540 = -0.093
```

```text
W5:
1.843 x 0.707
= 1.843 x 0.7 + 1.843 x 0.007
= 1.2901 + 0.012901
= 1.303001
= 1.303

e_E = 1.265 - 1.303 = -0.038
e_F = 1.342 - 1.303 = 0.039
```

| Window | t | z_hat_E | z_hat_F | e_E | e_F |
|---:|---:|---:|---:|---:|---:|
| W1 | -1.843 | -1.303 | -1.303 | 0.038 | -0.039 |
| W2 | -0.764 | -0.540 | -0.540 | -0.093 | 0.093 |
| W3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| W4 | 0.764 | 0.540 | 0.540 | 0.093 | -0.093 |
| W5 | 1.843 | 1.303 | 1.303 | -0.038 | 0.039 |

Residual kecil semua. Window normal memang menempel garis PC1.

### 29.7 Langkah 7: T2 tiap window dan threshold T2

Rumus `T2 = t^2 / lambda_1`, dengan `lambda_1 = 1.991`.

Kuadrat dulu.

```text
W1 dan W5:
1.843^2 = 1.843 x 1.843
= 1.843 x (1.8 + 0.04 + 0.003)
= 3.3174 + 0.07372 + 0.005529
= 3.396649
= 3.397
```

```text
W2 dan W4:
0.764^2 = 0.764 x 0.764
= 0.764 x (0.7 + 0.06 + 0.004)
= 0.5348 + 0.04584 + 0.003056
= 0.583696
= 0.584
```

```text
W3:
0.000^2 = 0.000
```

Sekarang bagi dengan `1.991`.

```text
Untuk 3.397 / 1.991:

1 x 1.991 = 1.991 (kurang)
2 x 1.991 = 3.982 (lebih)
1.706 x 1.991 = 3.396646 (pas ke 3.397)

jadi 3.397 / 1.991 = 1.706
```

```text
Untuk 0.584 / 1.991:

0.2 x 1.991 = 0.3982 (kurang)
0.4 x 1.991 = 0.7964 (lebih)
0.2933 x 1.991 = 0.5839603 (pas ke 0.584)

jadi 0.584 / 1.991 = 0.293
```

| Window | t | $t^2$ | $T^2=t^2/1.991$ |
|---:|---:|---:|---:|
| W1 | -1.843 | 3.397 | 1.706 |
| W2 | -0.764 | 0.584 | 0.293 |
| W3 | 0.000 | 0.000 | 0.000 |
| W4 | 0.764 | 0.584 | 0.293 |
| W5 | 1.843 | 3.397 | 1.706 |

Ambil yang paling besar dari kolom T2 training.

```text
mulai max = 1.706
bandingkan 0.293 -> tetap 1.706
bandingkan 0.000 -> tetap 1.706
bandingkan 0.293 -> tetap 1.706
bandingkan 1.706 -> tetap 1.706
```

$$
\text{threshold}_{T^2}=\max(T^2\ \text{training})=1.706
$$

### 29.8 Langkah 8: Q tiap window dan threshold Q

Rumus `Q = e_E^2 + e_F^2`.

Kuadrat residual kecil boleh ditulis 6 desimal dulu, baru Q dibulatkan 3 desimal.

```text
W1:
0.038^2 = 0.038 x 0.038 = 0.001444
(-0.039)^2 = 0.039 x 0.039 = 0.001521

Q = 0.001444 + 0.001521
Q = 0.002965
Q = 0.003
```

```text
W2:
(-0.093)^2 = 0.093 x 0.093 = 0.008649
0.093^2 = 0.093 x 0.093 = 0.008649

Q = 0.008649 + 0.008649
Q = 0.017298
Q = 0.017
```

```text
W3:
0.000^2 + 0.000^2 = 0.000
```

```text
W4:
0.093^2 = 0.093 x 0.093 = 0.008649
(-0.093)^2 = 0.093 x 0.093 = 0.008649

Q = 0.008649 + 0.008649
Q = 0.017298
Q = 0.017
```

```text
W5:
(-0.038)^2 = 0.038 x 0.038 = 0.001444
0.039^2 = 0.039 x 0.039 = 0.001521

Q = 0.001444 + 0.001521
Q = 0.002965
Q = 0.003
```

| Window | e_E | e_F | $Q$ |
|---:|---:|---:|---:|
| W1 | 0.038 | -0.039 | 0.003 |
| W2 | -0.093 | 0.093 | 0.017 |
| W3 | 0.000 | 0.000 | 0.000 |
| W4 | 0.093 | -0.093 | 0.017 |
| W5 | -0.038 | 0.039 | 0.003 |

Ambil Q training yang paling besar.

```text
mulai max = 0.003
bandingkan 0.017 -> ganti jadi 0.017
bandingkan 0.000 -> tetap 0.017
bandingkan 0.017 -> tetap 0.017
bandingkan 0.003 -> tetap 0.017
```

$$
\text{threshold}_Q=\max(Q\ \text{training})=0.017
$$

### 29.9 Langkah 9: kumpulkan parameter model

Sampai sini model PCA manual sudah jadi. Kunci semua angka ini:

Ini bukan angka baru. Ini cuma kotak bekal. Nanti saat `X_new` datang, kita tidak menghitung mean baru, tidak menghitung standar deviasi baru, dan tidak membuat PCA baru. Kita pakai bekal ini saja.

| Parameter | Nilai |
|---|---:|
| `mu_E` | 22 |
| `s_E` | 1.581 |
| `mu_F` | 43 |
| `s_F` | 2.236 |
| $r$ | 0.991 |
| $p_1$ | `[0.707, 0.707]` |
| $\lambda_1$ | 1.991 |
| $\text{threshold}_{T^2}$ | 1.706 |
| $\text{threshold}_Q$ | 0.017 |

Cara bacanya:

```text
untuk standardisasi E, pakai mu_E = 22 dan s_E = 1.581
untuk standardisasi F, pakai mu_F = 43 dan s_F = 2.236
untuk proyeksi, pakai p1 = [0.707, 0.707]
untuk T2, bagi dengan lambda1 = 1.991
untuk vonis, bandingkan T2 dengan 1.706 dan Q dengan 0.017
```

### 29.10 Langkah 10: uji window baru sampai keputusan

Window baru `X_new = [26, 42]`. Pakai parameter di atas. Jangan hitung mean atau standar deviasi baru.

Standardisasi:

Untuk E:

```text
z_E = (26 - 22) / 1.581
z_E = 4 / 1.581

2 x 1.581 = 3.162 (kurang)
3 x 1.581 = 4.743 (lebih)
2.530 x 1.581 = 3.99993 (pas ke 4.000)

jadi z_E = 2.530
```

$$
z_E=\frac{26-22}{1.581}=\frac{4}{1.581}=2.530
$$

Untuk F:

```text
z_F = (42 - 43) / 2.236
z_F = -1 / 2.236

0.44 x 2.236 = 0.98384 (kurang)
0.45 x 2.236 = 1.0062 (lebih)
0.44723 x 2.236 = 1.000006 (pas ke 1.000)

jadi z_F = -0.447
```

$$
z_F=\frac{42-43}{2.236}=\frac{-1}{2.236}=-0.447
$$

$$
z_{\text{new}}=[2.530,-0.447]
$$

E jauh di atas rata-rata, F sedikit di bawah rata-rata. Tanda awal hubungan rusak.

Proyeksi ke PC1:

Pakai `0.707 = 0.7 + 0.007`.

```text
2.530 x 0.707
= 2.530 x 0.7 + 2.530 x 0.007
= 1.771 + 0.01771
= 1.78871
= 1.789
```

```text
-0.447 x 0.707
= -(0.447 x 0.7 + 0.447 x 0.007)
= -(0.3129 + 0.003129)
= -0.316029
= -0.316
```

```text
t_new = 1.78871 + (-0.316029)
t_new = 1.472681
t_new = 1.473
```

$$
t_{\text{new}}=(2.530)(0.707)+(-0.447)(0.707)=1.789+(-0.316)=1.473
$$

T2:

Kuadratkan `t_new` dulu.

```text
1.473^2 = 1.473 x 1.473
= 1.473 x (1.4 + 0.07 + 0.003)
= 2.0622 + 0.10311 + 0.004419
= 2.169729
= 2.170
```

Lalu bagi dengan `lambda_1 = 1.991`.

```text
2.170 / 1.991

1 x 1.991 = 1.991 (kurang)
1.2 x 1.991 = 2.3892 (lebih)
1.090 x 1.991 = 2.17019 (pas ke 2.170)

jadi T2_new = 1.090
```

$$
T^2_{\text{new}}=\frac{(1.473)^2}{1.991}=\frac{2.170}{1.991}=1.090
$$

Bandingkan dengan threshold T2.

```text
threshold_T2 = 1.706

1.090 lebih kecil dari 1.706
jadi 1.090 > 1.706 adalah False
```

$$
1.090>1.706\ \text{adalah False}
$$

Dari T2 saja, window ini lolos. Belum terlalu jauh di sepanjang jalan normal.

Rekonstruksi dan residual:

Hitung `z_hat` dari `t_new * p1`.

```text
1.473 x 0.707
= 1.473 x 0.7 + 1.473 x 0.007
= 1.0311 + 0.010311
= 1.041411
= 1.041
```

Karena dua komponen `p1` sama-sama `0.707`, dua hasil rekonstruksi juga sama.

$$
\hat z_{\text{new}}=1.473[0.707,0.707]=[1.041,1.041]
$$

Sekarang residual, yaitu `z asli - z_hat`.

```text
e_E = 2.530 - 1.041 = 1.489
e_F = -0.447 - 1.041 = -1.488
```

$$
e_{\text{new}}=[2.530,-0.447]-[1.041,1.041]=[1.489,-1.488]
$$

Q:

Kuadratkan residual satu-satu.

```text
1.489^2 = 1.489 x 1.489
= 1.489 x (1.4 + 0.08 + 0.009)
= 2.0846 + 0.11912 + 0.013401
= 2.217121
= 2.217
```

```text
(-1.488)^2 = 1.488 x 1.488
= 1.488 x (1.4 + 0.08 + 0.008)
= 2.0832 + 0.11904 + 0.011904
= 2.214144
= 2.214
```

Jumlahkan.

```text
Q_new = 2.217 + 2.214
Q_new = 4.431
```

$$
Q_{\text{new}}=(1.489)^2+(-1.488)^2=2.217+2.214=4.431
$$

Bandingkan dengan threshold Q.

```text
threshold_Q = 0.017

4.431 jauh lebih besar dari 0.017
jadi 4.431 > 0.017 adalah True
```

$$
4.431>0.017\ \text{adalah True}
$$

Keputusan dengan aturan OR:

```text
T2 flag = False
Q flag = True

False OR True = True
```

$$
\mathrm{is\_anomaly}=(1.090>1.706)\ \mathrm{OR}\ (4.431>0.017)=\mathrm{False}\ \mathrm{OR}\ \mathrm{True}=\mathrm{True}
$$

Skor gabungan:

Untuk skor T2:

```text
score_T2 = 1.090 / 1.706

0.6 x 1.706 = 1.0236 (kurang)
0.7 x 1.706 = 1.1942 (lebih)
0.639 x 1.706 = 1.090134 (pas ke 1.090)

jadi score_T2 = 0.639
```

$$
\text{score}_{T^2}=\frac{1.090}{1.706}=0.639
$$

Untuk skor Q:

```text
score_Q = 4.431 / 0.017

200 x 0.017 = 3.4 (kurang)
300 x 0.017 = 5.1 (lebih)
260.647 x 0.017 = 4.430999 (pas ke 4.431)

jadi score_Q = 260.647
```

$$
\text{score}_Q=\frac{4.431}{0.017}=260.647
$$

Ambil skor paling besar.

```text
bandingkan 0.639 dan 260.647
260.647 lebih besar

score = 260.647
```

$$
\text{score}=\max(0.639,260.647)=260.647
$$

Vonis: anomali, karena Q, bukan karena T2.

Bahasa operator:

```text
Window ini tidak terlalu jauh di sepanjang pola normal, jadi T2 masih lolos.
Tetapi window keluar jauh dari pola normal, jadi Q sangat besar.
Energi band tinggi, frekuensi dominan tidak ikut naik seperti kebiasaan normal.
```

Kalau kamu bisa mengikuti sepuluh langkah ini sampai vonis, kamu sudah bisa menghitung bareng. Sekarang giliranmu mengerjakan sendiri di Bagian 30.

## Bagian 30. Latihan soal berkunci

Pola belajarnya lengkap sekarang. Bagian 10 sampai 19 adalah contoh yang dikerjakan penulis. Bagian 29 tadi kita kerjakan bersama. Sekarang giliranmu.

Kerjakan dulu di kertas sampai selesai. Kunci jawaban ada tepat di bawah tiap latihan. Jangan diintip sebelum mencoba.

| Latihan | Data normal | Window baru | Cerita | Kunci |
|---|---|---|---|---|
| Latihan 1 | E 6 sampai 10, F 20 sampai 24 | `X_new = 11, 21` | calon anomali | bagian 30.8 |
| Latihan 2 | E 30 sampai 38, F 60 sampai 67 | `X_new = 35, 65` | calon normal | bagian 30.10 |

### 30.1 Data latihan 1

Data normal baru:

| Window | E | F |
|---:|---:|---:|
| W1 | 6 | 20 |
| W2 | 7 | 21 |
| W3 | 8 | 22 |
| W4 | 9 | 22 |
| W5 | 10 | 24 |

Window baru:

| Window | E | F |
|---|---:|---:|
| X_new | 11 | 21 |

Petunjuk rasa: $X_{\text{new}}$ punya energi tinggi tetapi frekuensi tidak ikut tinggi. Kalau pola normalnya naik bersama, Q kemungkinan akan menjadi sinyal penting.

### 30.2 Tabel kerja mean dan standar deviasi

| Fitur | sum | n | mean | sum kuadrat deviation | variance sample | std sample |
|---|---:|---:|---:|---:|---:|---:|
| E |  | 5 |  |  |  |  |
| F |  | 5 |  |  |  |  |

### 30.3 Tabel kerja z-score

| Window | $E$ | $z_E$ | $F$ | $z_F$ |
|---:|---:|---:|---:|---:|
| W1 | 6 |  | 20 |  |
| W2 | 7 |  | 21 |  |
| W3 | 8 |  | 22 |  |
| W4 | 9 |  | 22 |  |
| W5 | 10 |  | 24 |  |

### 30.4 Tabel kerja correlation

| Window | $z_E$ | $z_F$ | z_E * z_F |
|---:|---:|---:|---:|
| W1 |  |  |  |
| W2 |  |  |  |
| W3 |  |  |  |
| W4 |  |  |  |
| W5 |  |  |  |
| Jumlah |  |  |  |

```text
r =
p1 =
lambda_1 =
```

### 30.5 Tabel kerja training PCA

| Window | $z_E$ | $z_F$ | t | z_hat_E | z_hat_F | e_E | e_F | T2 | Q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 |  |  |  |  |  |  |  |  |  |
| W2 |  |  |  |  |  |  |  |  |  |
| W3 |  |  |  |  |  |  |  |  |  |
| W4 |  |  |  |  |  |  |  |  |  |
| W5 |  |  |  |  |  |  |  |  |  |

```text
threshold_T2 =
threshold_Q =
```

### 30.6 Tabel kerja inference

| Langkah | Hasil |
|---|---:|
| `z_E_new` |  |
| `z_F_new` |  |
| $t_{\text{new}}$ |  |
| $T^2_{\text{new}}$ |  |
| `z_hat_E_new` |  |
| `z_hat_F_new` |  |
| `e_E_new` |  |
| `e_F_new` |  |
| $Q_{\text{new}}$ |  |
| $\text{score}_{T^2}$ |  |
| $\text{score}_Q$ |  |
| $\text{score}$ |  |

Keputusan:

| Pertanyaan | Jawaban |
|---|---|
| `T2_new > threshold_T2`? |  |
| `Q_new > threshold_Q`? |  |
| Normal atau anomali? |  |
| Penyebab utama T2 atau Q? |  |
| Satu kalimat alasan |  |

### 30.7 Latihan konsep tanpa angka

Isi dengan $T^2$, $Q$, atau `T2 dan Q`.

| Situasi | Skor yang kemungkinan naik |
|---|---|
| Window jauh di sepanjang pola normal |  |
| Window keluar dari pola normal |  |
| Energi dan frekuensi sama-sama sangat tinggi, masih searah |  |
| Energi tinggi tetapi frekuensi rendah |  |
| Titik dekat pusat dan dekat garis |  |

### 30.8 Kunci jawaban Latihan 1

Bandingkan hasilmu dengan kunci ini. Kalau beda tipis di digit terakhir, biasanya hanya pembulatan.

Kita hitung pelan-pelan. Ini versi raw, seperti anak kecil meraba angka. Angka akhir tetap yang di tabel.

Mean dan standar deviasi:

```text
E:
sum jalan:
6
6 + 7 = 13
13 + 8 = 21
21 + 9 = 30
30 + 10 = 40

40 / 5:
5 x 7 = 35 (kurang)
5 x 8 = 40 (pas)
mean E = 8
```

Deviasi E dari mean 8:

```text
W1: 6 - 8 = -2
W2: 7 - 8 = -1
W3: 8 - 8 = 0
W4: 9 - 8 = 1
W5: 10 - 8 = 2

kuadrat:
(-2)^2 = 2 x 2 = 4
(-1)^2 = 1 x 1 = 1
0^2 = 0
1^2 = 1
2^2 = 4

sum d^2 jalan:
4
4 + 1 = 5
5 + 0 = 5
5 + 1 = 6
6 + 4 = 10
```

```text
variance E = 10 / (5 - 1) = 10 / 4

10 / 4:
4 x 2 = 8 (kurang)
4 x 2.5 = 10 (pas)
variance E = 2.5
```

Akar standar deviasi E:

```text
sqrt(2.5), coba kuadrat:
1^2 = 1 (kurang)
2^2 = 4 (lebih)

1.5^2 = 1.5 x (1 + 0.5)
      = 1.5 + 0.75
      = 2.25 (kurang)

1.6^2 = 1.6 x (1 + 0.6)
      = 1.6 + 0.96
      = 2.56 (lebih)

1.58^2 = 1.58 x (1 + 0.5 + 0.08)
       = 1.58 + 0.79 + 0.1264
       = 2.4964 (kurang)

1.581^2 = 1.581 x (1 + 0.5 + 0.08 + 0.001)
        = 1.581 + 0.7905 + 0.12648 + 0.001581
        = 2.499561 (pas dekat)
std E = 1.581
```

```text
F:
sum jalan:
20
20 + 21 = 41
41 + 22 = 63
63 + 22 = 85
85 + 24 = 109

109 / 5:
5 x 21 = 105 (kurang)
5 x 21.8 = 109 (pas)
mean F = 21.8
```

Deviasi F dari mean 21.8:

```text
W1: 20 - 21.8 = -1.8
W2: 21 - 21.8 = -0.8
W3: 22 - 21.8 = 0.2
W4: 22 - 21.8 = 0.2
W5: 24 - 21.8 = 2.2

kuadrat:
1.8^2 = 1.8 x (1 + 0.8)
      = 1.8 + 1.44
      = 3.24

0.8^2 = 0.8 x 0.8 = 0.64

0.2^2 = 0.2 x 0.2 = 0.04
0.2^2 = 0.2 x 0.2 = 0.04

2.2^2 = 2.2 x (2 + 0.2)
      = 4.4 + 0.44
      = 4.84

sum d^2 jalan:
3.24
3.24 + 0.64 = 3.88
3.88 + 0.04 = 3.92
3.92 + 0.04 = 3.96
3.96 + 4.84 = 8.80
```

```text
variance F = 8.8 / (5 - 1) = 8.8 / 4

8.8 / 4:
4 x 2 = 8 (kurang)
4 x 2.2 = 8.8 (pas)
variance F = 2.2
```

Akar standar deviasi F:

```text
sqrt(2.2), coba kuadrat:
1^2 = 1 (kurang)
2^2 = 4 (lebih)

1.4^2 = 1.4 x (1 + 0.4)
      = 1.4 + 0.56
      = 1.96 (kurang)

1.5^2 = 1.5 x (1 + 0.5)
      = 1.5 + 0.75
      = 2.25 (lebih)

1.48^2 = 1.48 x (1 + 0.4 + 0.08)
       = 1.48 + 0.592 + 0.1184
       = 2.1904 (kurang)

1.483^2 = 1.483 x (1 + 0.4 + 0.08 + 0.003)
        = 1.483 + 0.5932 + 0.11864 + 0.004449
        = 2.199289 (pas dekat)
std F = 1.483
```

Tabel hasil akhirnya:

| Fitur | sum | n | mean | sum d^2 | variance | std |
|---|---:|---:|---:|---:|---:|---:|
| E | 40 | 5 | 8 | 10 | 2.5 | 1.581 |
| F | 109 | 5 | 21.8 | 8.8 | 2.2 | 1.483 |

Z-score:

Rumus kecilnya:

$$
z=\frac{x-\text{mean}}{\text{std}}
$$

Untuk E:

```text
W1: (6 - 8) / 1.581 = -2 / 1.581

2 / 1.581:
1.581 x 1.2 = 1.8972 (kurang)
1.581 x 1.26 = 1.99206 (kurang)
1.581 x 1.265 = 1.999965 (pas)
z_E W1 = -1.265

W2: (7 - 8) / 1.581 = -1 / 1.581

1 / 1.581:
1.581 x 0.6 = 0.9486 (kurang)
1.581 x 0.63 = 0.99603 (kurang)
1.581 x 0.633 = 1.000773 (pas dekat, dibulatkan)
z_E W2 = -0.633

W3: (8 - 8) / 1.581 = 0 / 1.581 = 0.000

W4 sama jarak dengan W2 tetapi positif:
z_E W4 = 0.633

W5 sama jarak dengan W1 tetapi positif:
z_E W5 = 1.265
```

Untuk F:

```text
W1: (20 - 21.8) / 1.483 = -1.8 / 1.483

1.8 / 1.483:
1.483 x 1.2 = 1.7796 (kurang)
1.483 x 1.21 = 1.79443 (kurang)
1.483 x 1.214 = 1.800362 (pas dekat)
z_F W1 = -1.214

W2: (21 - 21.8) / 1.483 = -0.8 / 1.483

0.8 / 1.483:
1.483 x 0.5 = 0.7415 (kurang)
1.483 x 0.54 = 0.80082 (lebih sedikit)
1.483 x 0.539 = 0.799337 (pas dekat)
z_F W2 = -0.539

W3: (22 - 21.8) / 1.483 = 0.2 / 1.483

0.2 / 1.483:
1.483 x 0.13 = 0.19279 (kurang)
1.483 x 0.135 = 0.200205 (pas dekat)
z_F W3 = 0.135

W4 sama dengan W3:
z_F W4 = 0.135

W5: (24 - 21.8) / 1.483 = 2.2 / 1.483

2.2 / 1.483:
1.483 x 1.4 = 2.0762 (kurang)
1.483 x 1.48 = 2.19484 (kurang)
1.483 x 1.483 = 2.199289 (pas dekat, dibulatkan)
z_F W5 = 1.483
```

Tabel hasil akhirnya:

| Window | $z_E$ | $z_F$ |
|---:|---:|---:|
| W1 | -1.265 | -1.214 |
| W2 | -0.633 | -0.539 |
| W3 | 0.000 | 0.135 |
| W4 | 0.633 | 0.135 |
| W5 | 1.265 | 1.483 |

Correlation dan PC1:

Kalikan satu-satu dulu. Ini perkalian nilai tempat.

```text
W1:
(-1.265) x (-1.214) = positif
1.214 x (1 + 0.2 + 0.06 + 0.005)
= 1.214 + 0.2428 + 0.07284 + 0.00607
= 1.53571

W2:
(-0.633) x (-0.539) = positif
0.539 x (0.6 + 0.03 + 0.003)
= 0.3234 + 0.01617 + 0.001617
= 0.341187

W3:
0.000 x 0.135 = 0

W4:
0.633 x 0.135
0.135 x (0.6 + 0.03 + 0.003)
= 0.081 + 0.00405 + 0.000405
= 0.085455

W5:
1.265 x 1.483
1.483 x (1 + 0.2 + 0.06 + 0.005)
= 1.483 + 0.2966 + 0.08898 + 0.007415
= 1.875995
```

Penjumlahan running:

```text
1.535710
1.535710 + 0.341187 = 1.876897
1.876897 + 0 = 1.876897
1.876897 + 0.085455 = 1.962352
1.962352 + 1.875995 = 3.838347
```

```text
r = 3.838347 / (5 - 1) = 3.838347 / 4

3.838347 / 4:
4 x 0.95 = 3.8 (kurang)
4 x 0.959 = 3.836 (kurang)
4 x 0.960 = 3.840 (pas dekat)
r = 0.960
```

Untuk dua fitur yang geraknya searah, PC1-nya garis tengah:

$$
p_1=[0.707,0.707]
$$

Eigenvalue pertama:

```text
lambda_1 = 1 + r
lambda_1 = 1 + 0.960 = 1.960
```

$$
r=0.960,\quad p_1=[0.707,0.707],\quad \lambda_1=1.960
$$

Training PCA:

Score $t$ itu z ditempel ke arah PC1.

$$
t=z_E(0.707)+z_F(0.707)=(z_E+z_F)(0.707)
$$

```text
W1:
z_E + z_F = -1.265 + (-1.214) = -2.479
2.479 x 0.707 = 2.479 x (0.7 + 0.007)
              = 1.7353 + 0.017353
              = 1.752653
t = -1.753

W2:
-0.633 + (-0.539) = -1.172
1.172 x 0.707 = 1.172 x (0.7 + 0.007)
              = 0.8204 + 0.008204
              = 0.828604
t = -0.829

W3:
0.000 + 0.135 = 0.135
0.135 x 0.707 = 0.135 x (0.7 + 0.007)
              = 0.0945 + 0.000945
              = 0.095445
t = 0.095

W4:
0.633 + 0.135 = 0.768
0.768 x 0.707 = 0.768 x (0.7 + 0.007)
              = 0.5376 + 0.005376
              = 0.542976
t = 0.543

W5:
1.265 + 1.483 = 2.748
2.748 x 0.707 = 2.748 x (0.7 + 0.007)
              = 1.9236 + 0.019236
              = 1.942836
t = 1.943
```

Rekonstruksi $z_{\text{hat}}=t \times 0.707$. Karena PC1 punya dua angka sama, `z_hat_E` dan `z_hat_F` sama.

```text
W1:
-1.753 x 0.707
1.753 x (0.7 + 0.007) = 1.2271 + 0.012271 = 1.239371
z_hat = -1.239

W2:
-0.829 x 0.707
0.829 x (0.7 + 0.007) = 0.5803 + 0.005803 = 0.586103
z_hat = -0.586

W3:
0.095 x 0.707
0.095 x (0.7 + 0.007) = 0.0665 + 0.000665 = 0.067165
z_hat = 0.067

W4:
0.543 x 0.707
0.543 x (0.7 + 0.007) = 0.3801 + 0.003801 = 0.383901
z_hat = 0.384

W5:
1.943 x 0.707
1.943 x (0.7 + 0.007) = 1.3601 + 0.013601 = 1.373701
z_hat = 1.374
```

Residual itu sisa: `e = z - z_hat`.

```text
W1:
e_E = -1.265 - (-1.239) = -0.026
e_F = -1.214 - (-1.239) = 0.025

W2:
e_E = -0.633 - (-0.586) = -0.047
e_F = -0.539 - (-0.586) = 0.047

W3:
e_E = 0.000 - 0.067 = -0.067
e_F = 0.135 - 0.067 = 0.068

W4:
e_E = 0.633 - 0.384 = 0.249
e_F = 0.135 - 0.384 = -0.249

W5:
e_E = 1.265 - 1.374 = -0.109
e_F = 1.483 - 1.374 = 0.109
```

Sekarang $T^2=t^2/\lambda_1$. Kuadrat dulu, lalu bagi.

```text
W1:
1.753^2 = 1.753 x (1 + 0.7 + 0.05 + 0.003)
        = 1.753 + 1.2271 + 0.08765 + 0.005259
        = 3.073009

3.073009 / 1.960:
1.960 x 1.56 = 3.0576 (kurang)
1.960 x 1.568 = 3.07328 (pas dekat)
T2 = 1.568

W2:
0.829^2 = 0.829 x (0.8 + 0.02 + 0.009)
        = 0.6632 + 0.01658 + 0.007461
        = 0.687241

0.687241 / 1.960:
1.960 x 0.35 = 0.686 (kurang)
1.960 x 0.351 = 0.68796 (pas dekat)
T2 = 0.351

W3:
0.095^2 = 0.095 x (0.09 + 0.005)
        = 0.00855 + 0.000475
        = 0.009025

0.009025 / 1.960:
1.960 x 0.004 = 0.00784 (kurang)
1.960 x 0.005 = 0.00980 (pas dekat)
T2 = 0.005

W4:
0.543^2 = 0.543 x (0.5 + 0.04 + 0.003)
        = 0.2715 + 0.02172 + 0.001629
        = 0.294849

0.294849 / 1.960:
1.960 x 0.15 = 0.294 (pas dekat)
T2 = 0.150

W5:
1.943^2 = 1.943 x (1 + 0.9 + 0.04 + 0.003)
        = 1.943 + 1.7487 + 0.07772 + 0.005829
        = 3.775249

3.775249 / 1.960:
1.960 x 1.92 = 3.7632 (kurang)
1.960 x 1.926 = 3.77496 (pas dekat)
T2 = 1.926
```

Sekarang $Q=e_E^2+e_F^2$. Ini sisa kuadrat, kecil-kecil saja.

```text
W1:
(-0.026)^2 = 0.000676
0.025^2 = 0.000625
Q = 0.000676 + 0.000625 = 0.001301 = 0.001

W2:
(-0.047)^2 = 0.002209
0.047^2 = 0.002209
Q = 0.002209 + 0.002209 = 0.004418 = 0.004

W3:
(-0.067)^2 = 0.004489
0.068^2 = 0.004624
Q = 0.004489 + 0.004624 = 0.009113 = 0.009

W4:
0.249^2 = 0.249 x (0.2 + 0.04 + 0.009)
        = 0.0498 + 0.00996 + 0.002241
        = 0.062001
(-0.249)^2 = 0.062001
Q = 0.062001 + 0.062001 = 0.124002 = 0.124

W5:
(-0.109)^2 = 0.011881
0.109^2 = 0.011881
Q = 0.011881 + 0.011881 = 0.023762 = 0.024
```

Tabel hasil akhirnya:

| Window | t | z_hat_E | z_hat_F | e_E | e_F | T2 | Q |
|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | -1.753 | -1.239 | -1.239 | -0.026 | 0.025 | 1.568 | 0.001 |
| W2 | -0.829 | -0.586 | -0.586 | -0.047 | 0.047 | 0.351 | 0.004 |
| W3 | 0.095 | 0.067 | 0.067 | -0.067 | 0.068 | 0.005 | 0.009 |
| W4 | 0.543 | 0.384 | 0.384 | 0.249 | -0.249 | 0.150 | 0.124 |
| W5 | 1.943 | 1.374 | 1.374 | -0.109 | 0.109 | 1.926 | 0.024 |

Threshold:

Ambil nilai terbesar dari training.

```text
T2 jalan:
max awal W1 = 1.568
banding W2 0.351, tetap 1.568
banding W3 0.005, tetap 1.568
banding W4 0.150, tetap 1.568
banding W5 1.926, naik jadi 1.926

Q jalan:
max awal W1 = 0.001
banding W2 0.004, naik jadi 0.004
banding W3 0.009, naik jadi 0.009
banding W4 0.124, naik jadi 0.124
banding W5 0.024, tetap 0.124
```

$$
\text{threshold}_{T^2}=1.926,\quad \text{threshold}_Q=0.124
$$

Inference `X_new = [11, 21]`:

Z-score window baru:

```text
z_E_new = (11 - 8) / 1.581 = 3 / 1.581

3 / 1.581:
1.581 x 1.8 = 2.8458 (kurang)
1.581 x 1.89 = 2.98809 (kurang)
1.581 x 1.898 = 3.000738 (pas dekat)
z_E_new = 1.898

z_F_new = (21 - 21.8) / 1.483 = -0.8 / 1.483

0.8 / 1.483:
1.483 x 0.53 = 0.78599 (kurang)
1.483 x 0.539 = 0.799337 (pas dekat)
z_F_new = -0.539
```

Score baru:

```text
z_E_new + z_F_new = 1.898 + (-0.539) = 1.359

t_new = 1.359 x 0.707
1.359 x (0.7 + 0.007)
= 0.9513 + 0.009513
= 0.960813
t_new = 0.961
```

$T^2$ baru:

```text
0.961^2 = 0.961 x (0.9 + 0.06 + 0.001)
        = 0.8649 + 0.05766 + 0.000961
        = 0.923521

0.923521 / 1.960:
1.960 x 0.47 = 0.9212 (kurang)
1.960 x 0.471 = 0.92316 (pas dekat)
T2_new = 0.471
```

Rekonstruksi baru:

```text
z_hat_new = t_new x 0.707
0.961 x (0.7 + 0.007)
= 0.6727 + 0.006727
= 0.679427
z_hat_E_new = 0.679
z_hat_F_new = 0.679
```

Residual baru:

```text
e_E_new = 1.898 - 0.679 = 1.219
e_F_new = -0.539 - 0.679 = -1.218
```

$Q$ baru:

```text
1.219^2 = 1.219 x (1 + 0.2 + 0.01 + 0.009)
        = 1.219 + 0.2438 + 0.01219 + 0.010971
        = 1.485961

1.218^2 = 1.218 x (1 + 0.2 + 0.01 + 0.008)
        = 1.218 + 0.2436 + 0.01218 + 0.009744
        = 1.483524

Q_new = 1.485961 + 1.483524 = 2.969485 = 2.969
```

Skor terhadap threshold:

```text
score_T2 = 0.471 / 1.926

0.471 / 1.926:
1.926 x 0.24 = 0.46224 (kurang)
1.926 x 0.245 = 0.47187 (pas dekat)
score_T2 = 0.245

score_Q = 2.969 / 0.124

2.969 / 0.124:
0.124 x 20 = 2.480 (kurang)
0.124 x 23 = 2.852 (kurang)
0.124 x 23.944 = 2.969056 (pas dekat)
score_Q = 23.944
```

Tabel hasil akhirnya:

| Langkah | Hasil |
|---|---:|
| `z_E_new` | 1.898 |
| `z_F_new` | -0.539 |
| $t_{\text{new}}$ | 0.961 |
| $T^2_{\text{new}}$ | 0.471 |
| `z_hat_E_new` | 0.679 |
| `z_hat_F_new` | 0.679 |
| `e_E_new` | 1.219 |
| `e_F_new` | -1.218 |
| $Q_{\text{new}}$ | 2.969 |
| $\text{score}_{T^2}$ | 0.245 |
| $\text{score}_Q$ | 23.944 |

Keputusan:

```text
T2_new kecil:
0.471 masih di bawah 1.926

Q_new besar:
2.969 jauh di atas 0.124
```

| Pertanyaan | Jawaban |
|---|---|
| `T2_new > threshold_T2`? | 0.471 > 1.926 = False |
| `Q_new > threshold_Q`? | 2.969 > 0.124 = True |
| Normal atau anomali? | anomali |
| Penyebab utama | Q |
| Satu kalimat alasan | energi naik tetapi frekuensi tidak ikut, jadi window keluar dari pola normal |

Kunci 30.7, latihan konsep tanpa angka:

| Situasi | Skor yang kemungkinan naik |
|---|---|
| Window jauh di sepanjang pola normal | T2 |
| Window keluar dari pola normal | Q |
| Energi dan frekuensi sama-sama sangat tinggi, masih searah | T2 |
| Energi tinggi tetapi frekuensi rendah | Q |
| Titik dekat pusat dan dekat garis | tidak ada, T2 dan Q sama-sama rendah |

### 30.9 Latihan 2, kasus yang mungkin normal

Latihan ini sengaja berbeda. Window barunya mengikuti pola normal, jadi kamu belajar rasa "lolos", bukan hanya rasa "anomali".

Data normal:

| Window | E | F |
|---:|---:|---:|
| W1 | 30 | 60 |
| W2 | 32 | 63 |
| W3 | 34 | 64 |
| W4 | 36 | 66 |
| W5 | 38 | 67 |

Window baru:

| Window | E | F |
|---|---:|---:|
| X_new | 35 | 65 |

Petunjuk rasa: energi `35` dan frekuensi `65` dua-duanya berada di tengah atas, naik bersama seperti pola normal. Kemungkinan besar T2 dan Q sama-sama lolos.

Worksheet mean dan standar deviasi:

| Fitur | sum | n | mean | sum d^2 | variance | std |
|---|---:|---:|---:|---:|---:|---:|
| E |  | 5 |  |  |  |  |
| F |  | 5 |  |  |  |  |

Worksheet z-score:

| Window | $E$ | $z_E$ | $F$ | $z_F$ |
|---:|---:|---:|---:|---:|
| W1 | 30 |  | 60 |  |
| W2 | 32 |  | 63 |  |
| W3 | 34 |  | 64 |  |
| W4 | 36 |  | 66 |  |
| W5 | 38 |  | 67 |  |

Worksheet correlation:

| Window | $z_E$ | $z_F$ | z_E * z_F |
|---:|---:|---:|---:|
| W1 |  |  |  |
| W2 |  |  |  |
| W3 |  |  |  |
| W4 |  |  |  |
| W5 |  |  |  |
| Jumlah |  |  |  |

```text
r =
p1 =
lambda_1 =
```

Worksheet training PCA:

| Window | $z_E$ | $z_F$ | t | z_hat_E | z_hat_F | e_E | e_F | T2 | Q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 |  |  |  |  |  |  |  |  |  |
| W2 |  |  |  |  |  |  |  |  |  |
| W3 |  |  |  |  |  |  |  |  |  |
| W4 |  |  |  |  |  |  |  |  |  |
| W5 |  |  |  |  |  |  |  |  |  |

```text
threshold_T2 =
threshold_Q =
```

Worksheet inference:

| Langkah | Hasil |
|---|---:|
| `z_E_new` |  |
| `z_F_new` |  |
| $t_{\text{new}}$ |  |
| $T^2_{\text{new}}$ |  |
| `z_hat_E_new` |  |
| `z_hat_F_new` |  |
| `e_E_new` |  |
| `e_F_new` |  |
| $Q_{\text{new}}$ |  |
| $\text{score}_{T^2}$ |  |
| $\text{score}_Q$ |  |
| $\text{score}$ |  |

### 30.10 Kunci jawaban Latihan 2

Sekarang kasus yang rasanya normal. Tetap kita raba angka satu-satu, jangan lompat.

Mean dan standar deviasi:

```text
E:
sum jalan:
30
30 + 32 = 62
62 + 34 = 96
96 + 36 = 132
132 + 38 = 170

170 / 5:
5 x 30 = 150 (kurang)
5 x 34 = 170 (pas)
mean E = 34
```

Deviasi E dari mean 34:

```text
W1: 30 - 34 = -4
W2: 32 - 34 = -2
W3: 34 - 34 = 0
W4: 36 - 34 = 2
W5: 38 - 34 = 4

kuadrat:
(-4)^2 = 4 x 4 = 16
(-2)^2 = 2 x 2 = 4
0^2 = 0
2^2 = 4
4^2 = 16

sum d^2 jalan:
16
16 + 4 = 20
20 + 0 = 20
20 + 4 = 24
24 + 16 = 40
```

```text
variance E = 40 / (5 - 1) = 40 / 4

40 / 4:
4 x 9 = 36 (kurang)
4 x 10 = 40 (pas)
variance E = 10
```

Akar standar deviasi E:

```text
sqrt(10), coba kuadrat:
3^2 = 9 (kurang)
4^2 = 16 (lebih)

3.1^2 = 3.1 x (3 + 0.1)
      = 9.3 + 0.31
      = 9.61 (kurang)

3.2^2 = 3.2 x (3 + 0.2)
      = 9.6 + 0.64
      = 10.24 (lebih)

3.16^2 = 3.16 x (3 + 0.1 + 0.06)
       = 9.48 + 0.316 + 0.1896
       = 9.9856 (kurang)

3.162^2 = 3.162 x (3 + 0.1 + 0.06 + 0.002)
        = 9.486 + 0.3162 + 0.18972 + 0.006324
        = 9.998244 (pas dekat)
std E = 3.162
```

```text
F:
sum jalan:
60
60 + 63 = 123
123 + 64 = 187
187 + 66 = 253
253 + 67 = 320

320 / 5:
5 x 60 = 300 (kurang)
5 x 64 = 320 (pas)
mean F = 64
```

Deviasi F dari mean 64:

```text
W1: 60 - 64 = -4
W2: 63 - 64 = -1
W3: 64 - 64 = 0
W4: 66 - 64 = 2
W5: 67 - 64 = 3

kuadrat:
(-4)^2 = 16
(-1)^2 = 1
0^2 = 0
2^2 = 4
3^2 = 9

sum d^2 jalan:
16
16 + 1 = 17
17 + 0 = 17
17 + 4 = 21
21 + 9 = 30
```

```text
variance F = 30 / (5 - 1) = 30 / 4

30 / 4:
4 x 7 = 28 (kurang)
4 x 7.5 = 30 (pas)
variance F = 7.5
```

Akar standar deviasi F:

```text
sqrt(7.5), coba kuadrat:
2^2 = 4 (kurang)
3^2 = 9 (lebih)

2.7^2 = 2.7 x (2 + 0.7)
      = 5.4 + 1.89
      = 7.29 (kurang)

2.8^2 = 2.8 x (2 + 0.8)
      = 5.6 + 2.24
      = 7.84 (lebih)

2.73^2 = 2.73 x (2 + 0.7 + 0.03)
       = 5.46 + 1.911 + 0.0819
       = 7.4529 (kurang)

2.739^2 = 2.739 x (2 + 0.7 + 0.03 + 0.009)
        = 5.478 + 1.9173 + 0.08217 + 0.024651
        = 7.502121 (pas dekat)
std F = 2.739
```

Tabel hasil akhirnya:

| Fitur | sum | n | mean | sum d^2 | variance | std |
|---|---:|---:|---:|---:|---:|---:|
| E | 170 | 5 | 34 | 40 | 10 | 3.162 |
| F | 320 | 5 | 64 | 30 | 7.5 | 2.739 |

Z-score:

Untuk E:

```text
W1: (30 - 34) / 3.162 = -4 / 3.162

4 / 3.162:
3.162 x 1.2 = 3.7944 (kurang)
3.162 x 1.26 = 3.98412 (kurang)
3.162 x 1.265 = 3.99993 (pas)
z_E W1 = -1.265

W2: (32 - 34) / 3.162 = -2 / 3.162

2 / 3.162:
3.162 x 0.6 = 1.8972 (kurang)
3.162 x 0.63 = 1.99206 (kurang)
3.162 x 0.633 = 2.001546 (pas dekat, dibulatkan)
z_E W2 = -0.633

W3: (34 - 34) / 3.162 = 0 / 3.162 = 0.000

W4 sama jarak dengan W2 tetapi positif:
z_E W4 = 0.633

W5 sama jarak dengan W1 tetapi positif:
z_E W5 = 1.265
```

Untuk F:

```text
W1: (60 - 64) / 2.739 = -4 / 2.739

4 / 2.739:
2.739 x 1.4 = 3.8346 (kurang)
2.739 x 1.46 = 3.99894 (pas dekat)
z_F W1 = -1.460

W2: (63 - 64) / 2.739 = -1 / 2.739

1 / 2.739:
2.739 x 0.36 = 0.98604 (kurang)
2.739 x 0.365 = 0.999735 (pas dekat)
z_F W2 = -0.365

W3: (64 - 64) / 2.739 = 0 / 2.739 = 0.000

W4: (66 - 64) / 2.739 = 2 / 2.739

2 / 2.739:
2.739 x 0.7 = 1.9173 (kurang)
2.739 x 0.73 = 1.99947 (pas dekat)
z_F W4 = 0.730

W5: (67 - 64) / 2.739 = 3 / 2.739

3 / 2.739:
2.739 x 1.09 = 2.98551 (kurang)
2.739 x 1.095 = 2.999205 (pas dekat)
z_F W5 = 1.095
```

Tabel hasil akhirnya:

| Window | $z_E$ | $z_F$ |
|---:|---:|---:|
| W1 | -1.265 | -1.460 |
| W2 | -0.633 | -0.365 |
| W3 | 0.000 | 0.000 |
| W4 | 0.633 | 0.730 |
| W5 | 1.265 | 1.095 |

Correlation dan PC1:

Kalikan z_E dan z_F per window.

```text
W1:
(-1.265) x (-1.460) = positif
1.460 x (1 + 0.2 + 0.06 + 0.005)
= 1.460 + 0.292 + 0.0876 + 0.0073
= 1.8469

W2:
(-0.633) x (-0.365) = positif
0.365 x (0.6 + 0.03 + 0.003)
= 0.219 + 0.01095 + 0.001095
= 0.231045

W3:
0.000 x 0.000 = 0

W4:
0.633 x 0.730
0.730 x (0.6 + 0.03 + 0.003)
= 0.438 + 0.0219 + 0.00219
= 0.462090

W5:
1.265 x 1.095
1.095 x (1 + 0.2 + 0.06 + 0.005)
= 1.095 + 0.219 + 0.0657 + 0.005475
= 1.385175
```

Penjumlahan running:

```text
1.846900
1.846900 + 0.231045 = 2.077945
2.077945 + 0 = 2.077945
2.077945 + 0.462090 = 2.540035
2.540035 + 1.385175 = 3.925210
```

```text
r = 3.925210 / (5 - 1) = 3.925210 / 4

3.925210 / 4:
4 x 0.98 = 3.92 (kurang)
4 x 0.981 = 3.924 (pas dekat)
r = 0.981
```

PC1 masih garis tengah karena dua fitur naik bareng:

$$
p_1=[0.707,0.707]
$$

```text
lambda_1 = 1 + r
lambda_1 = 1 + 0.981 = 1.981
```

$$
r=0.981,\quad p_1=[0.707,0.707],\quad \lambda_1=1.981
$$

Training PCA:

Score $t$:

```text
W1:
z_E + z_F = -1.265 + (-1.460) = -2.725
2.725 x 0.707 = 2.725 x (0.7 + 0.007)
              = 1.9075 + 0.019075
              = 1.926575
t = -1.927

W2:
-0.633 + (-0.365) = -0.998
0.998 x 0.707 = 0.998 x (0.7 + 0.007)
              = 0.6986 + 0.006986
              = 0.705586
t = -0.706

W3:
0.000 + 0.000 = 0.000
t = 0.000

W4:
0.633 + 0.730 = 1.363
1.363 x 0.707 = 1.363 x (0.7 + 0.007)
              = 0.9541 + 0.009541
              = 0.963641
t = 0.964

W5:
1.265 + 1.095 = 2.360
2.360 x 0.707 = 2.360 x (0.7 + 0.007)
              = 1.652 + 0.01652
              = 1.66852
t = 1.669
```

Rekonstruksi $z_{\text{hat}}=t \times 0.707$:

```text
W1:
-1.927 x 0.707
1.927 x (0.7 + 0.007) = 1.3489 + 0.013489 = 1.362389
z_hat = -1.362

W2:
-0.706 x 0.707
0.706 x (0.7 + 0.007) = 0.4942 + 0.004942 = 0.499142
z_hat = -0.499

W3:
0.000 x 0.707 = 0.000
z_hat = 0.000

W4:
0.964 x 0.707
0.964 x (0.7 + 0.007) = 0.6748 + 0.006748 = 0.681548
z_hat = 0.682

W5:
1.669 x 0.707
1.669 x (0.7 + 0.007) = 1.1683 + 0.011683 = 1.179983
z_hat = 1.180
```

Residual:

```text
W1:
e_E = -1.265 - (-1.362) = 0.097
e_F = -1.460 - (-1.362) = -0.098

W2:
e_E = -0.633 - (-0.499) = -0.134
e_F = -0.365 - (-0.499) = 0.134

W3:
e_E = 0.000 - 0.000 = 0.000
e_F = 0.000 - 0.000 = 0.000

W4:
e_E = 0.633 - 0.682 = -0.049
e_F = 0.730 - 0.682 = 0.048

W5:
e_E = 1.265 - 1.180 = 0.085
e_F = 1.095 - 1.180 = -0.085
```

$T^2=t^2/\lambda_1$:

```text
W1:
1.927^2 = 1.927 x (1 + 0.9 + 0.02 + 0.007)
        = 1.927 + 1.7343 + 0.03854 + 0.013489
        = 3.713329

3.713329 / 1.981:
1.981 x 1.87 = 3.70447 (kurang)
1.981 x 1.874 = 3.712394 (pas dekat)
T2 = 1.874

W2:
0.706^2 = 0.706 x (0.7 + 0.006)
        = 0.4942 + 0.004236
        = 0.498436

0.498436 / 1.981:
1.981 x 0.25 = 0.49525 (kurang)
1.981 x 0.252 = 0.499212 (pas dekat)
T2 = 0.252

W3:
0.000^2 = 0.000
T2 = 0.000

W4:
0.964^2 = 0.964 x (0.9 + 0.06 + 0.004)
        = 0.8676 + 0.05784 + 0.003856
        = 0.929296

0.929296 / 1.981:
1.981 x 0.46 = 0.91126 (kurang)
1.981 x 0.469 = 0.929089 (pas dekat)
T2 = 0.469

W5:
1.669^2 = 1.669 x (1 + 0.6 + 0.06 + 0.009)
        = 1.669 + 1.0014 + 0.10014 + 0.015021
        = 2.785561

2.785561 / 1.981:
1.981 x 1.40 = 2.7734 (kurang)
1.981 x 1.406 = 2.785286 (pas dekat)
T2 = 1.406
```

$Q=e_E^2+e_F^2$:

```text
W1:
0.097^2 = 0.009409
(-0.098)^2 = 0.009604
Q = 0.009409 + 0.009604 = 0.019013 = 0.019

W2:
(-0.134)^2 = 0.017956
0.134^2 = 0.017956
Q = 0.017956 + 0.017956 = 0.035912 = 0.036

W3:
0.000^2 + 0.000^2 = 0.000

W4:
(-0.049)^2 = 0.002401
0.048^2 = 0.002304
Q = 0.002401 + 0.002304 = 0.004705 = 0.005

W5:
0.085^2 = 0.007225
(-0.085)^2 = 0.007225
Q = 0.007225 + 0.007225 = 0.014450 = 0.014
```

Tabel hasil akhirnya:

| Window | t | z_hat_E | z_hat_F | e_E | e_F | T2 | Q |
|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | -1.927 | -1.362 | -1.362 | 0.097 | -0.098 | 1.874 | 0.019 |
| W2 | -0.706 | -0.499 | -0.499 | -0.134 | 0.134 | 0.252 | 0.036 |
| W3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| W4 | 0.964 | 0.682 | 0.682 | -0.049 | 0.048 | 0.469 | 0.005 |
| W5 | 1.669 | 1.180 | 1.180 | 0.085 | -0.085 | 1.406 | 0.014 |

Threshold:

```text
T2 jalan:
max awal W1 = 1.874
banding W2 0.252, tetap 1.874
banding W3 0.000, tetap 1.874
banding W4 0.469, tetap 1.874
banding W5 1.406, tetap 1.874

Q jalan:
max awal W1 = 0.019
banding W2 0.036, naik jadi 0.036
banding W3 0.000, tetap 0.036
banding W4 0.005, tetap 0.036
banding W5 0.014, tetap 0.036
```

$$
\text{threshold}_{T^2}=1.874,\quad \text{threshold}_Q=0.036
$$

Inference `X_new = [35, 65]`:

Z-score window baru:

```text
z_E_new = (35 - 34) / 3.162 = 1 / 3.162

1 / 3.162:
3.162 x 0.3 = 0.9486 (kurang)
3.162 x 0.316 = 0.999192 (pas dekat)
z_E_new = 0.316

z_F_new = (65 - 64) / 2.739 = 1 / 2.739

1 / 2.739:
2.739 x 0.36 = 0.98604 (kurang)
2.739 x 0.365 = 0.999735 (pas dekat)
z_F_new = 0.365
```

Score baru:

```text
z_E_new + z_F_new = 0.316 + 0.365 = 0.681

t_new = 0.681 x 0.707
0.681 x (0.7 + 0.007)
= 0.4767 + 0.004767
= 0.481467
t_new = 0.481
```

$T^2$ baru:

```text
0.481^2 = 0.481 x (0.4 + 0.08 + 0.001)
        = 0.1924 + 0.03848 + 0.000481
        = 0.231361

0.231361 / 1.981:
1.981 x 0.11 = 0.21791 (kurang)
1.981 x 0.117 = 0.231777 (pas dekat)
T2_new = 0.117
```

Rekonstruksi baru:

```text
z_hat_new = 0.481 x 0.707
0.481 x (0.7 + 0.007)
= 0.3367 + 0.003367
= 0.340067
z_hat_E_new = 0.340
z_hat_F_new = 0.340
```

Residual baru:

```text
e_E_new = 0.316 - 0.340 = -0.024
e_F_new = 0.365 - 0.340 = 0.025
```

$Q$ baru:

```text
(-0.024)^2 = 0.000576
0.025^2 = 0.000625

Q_new = 0.000576 + 0.000625
Q_new = 0.001201 = 0.001
```

Skor terhadap threshold:

```text
score_T2 = 0.117 / 1.874

0.117 / 1.874:
1.874 x 0.06 = 0.11244 (kurang)
1.874 x 0.062 = 0.116188 (pas dekat)
score_T2 = 0.062

score_Q = 0.001 / 0.036

0.001 / 0.036:
0.036 x 0.02 = 0.00072 (kurang)
0.036 x 0.028 = 0.001008 (pas dekat)
score_Q = 0.028

score akhir = max(0.062, 0.028) = 0.062
```

Tabel hasil akhirnya:

| Langkah | Hasil |
|---|---:|
| `z_E_new` | 0.316 |
| `z_F_new` | 0.365 |
| $t_{\text{new}}$ | 0.481 |
| $T^2_{\text{new}}$ | 0.117 |
| `z_hat_E_new` | 0.340 |
| `z_hat_F_new` | 0.340 |
| `e_E_new` | -0.024 |
| `e_F_new` | 0.025 |
| $Q_{\text{new}}$ | 0.001 |
| $\text{score}_{T^2}$ | 0.062 |
| $\text{score}_Q$ | 0.028 |
| $\text{score}$ | 0.062 |

Keputusan:

```text
T2_new kecil:
0.117 masih di bawah 1.874

Q_new kecil:
0.001 masih di bawah 0.036

Dua-duanya lolos.
```

| Pertanyaan | Jawaban |
|---|---|
| `T2_new > threshold_T2`? | 0.117 > 1.874 = False |
| `Q_new > threshold_Q`? | 0.001 > 0.036 = False |
| Normal atau anomali? | normal |
| Penyebab utama | tidak ada, dua skor lolos |
| Satu kalimat alasan | energi dan frekuensi naik bersama mengikuti pola normal, jadi T2 dan Q kecil |

Karena `score = 0.062` lebih kecil dari 1, window ini normal. Inilah rasa window sehat: tidak jauh di jalan, tidak keluar dari jalan.

## Bagian 31. Penutup

PCA T2/Q terlihat seperti model matematika besar, tetapi hitungan manualnya bisa dipahami sebagai rangkaian langkah kecil.

Mulai dari data normal, kita hitung rata-rata dan standar deviasi. Fitur diskalakan supaya adil. Hubungan antar fitur dibaca melalui correlation. PCA memilih arah variasi utama. Window diproyeksikan ke arah itu untuk mendapatkan score $t$. T2 melihat apakah score itu ekstrem. Rekonstruksi menggambar ulang window dari PC yang disimpan. Residual adalah sisa gambar yang tidak cocok. Q/SPE menjumlahkan kuadrat sisa itu.

Untuk PDAM Pump Sentinel, cara berpikir ini membantu menjelaskan detector ke manusia. Model tidak sekadar berkata anomali. Model bisa dibaca sebagai dua pertanyaan sederhana:

```text
T2: jauh tidak di jalan normal?
Q : keluar tidak dari jalan normal?
```

Kalau kamu bisa menghitung contoh kecil ini di kertas, kamu sudah memegang inti PCA T2/Q spectral. Kode proyek hanya melakukan langkah yang sama dengan lebih banyak window, lebih banyak fitur, dan presisi angka yang lebih tinggi.

Pelan-pelan saja. PCA bukan sulap. PCA hanya cara rapi untuk membaca kebiasaan normal pompa, lalu menandai window yang terlalu jauh atau keluar dari kebiasaan itu.
