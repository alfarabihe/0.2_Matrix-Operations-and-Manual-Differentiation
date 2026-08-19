# Interpretasi Hasil — Operasi Matriks dan Turunan Manual

Dokumen ini menginterpretasikan output aktual dari eksekusi
`Operasi_Matriks_dan_Turunan_Manual.ipynb`, menghubungkan angka-angka yang dihasilkan dengan teori yang
mendasarinya.

---

## 1. Verifikasi Kebenaran Hasil (Perkalian Matriks)

```
Manual vs NumPy  sama (allclose)? : True
NumPy  vs PyTorch sama (allclose)? : True
Manual vs PyTorch sama (allclose)? : True
Selisih absolut maksimum (manual vs NumPy): 1.07e-14
```

**Interpretasi:**

- Ketiga pendekatan — implementasi manual dengan tiga `for`-loop, `numpy` (operator `@`), dan
  `torch.matmul` — menghasilkan matriks $C = AB$ berukuran $100 \times 100$ yang **secara matematis
  identik**.
- Selisih absolut maksimum yang tercatat ($1.07\times10^{-14}$) **bukan bug**, melainkan konsekuensi
  alami dari representasi bilangan *floating-point* (IEEE 754 double precision, presisi ±$10^{-16}$
  per operasi). Urutan penjumlahan pada rumus $C_{ij}=\sum_k A_{ik}B_{kj}$ berbeda antar implementasi
  (loop Python murni vs rutin BLAS yang mungkin membagi penjumlahan ke beberapa *thread*/blok memori),
  sehingga pembulatan terakumulasi sedikit berbeda. Selisih pada orde $10^{-14}$ jauh di bawah presisi
  praktis yang relevan untuk aplikasi apa pun — ini mengonfirmasi bahwa ketiga implementasi **benar**,
  bukan mendekati benar.
- Kesimpulan praktis: rumus perkalian matriks $C_{ij}=\sum_k A_{ik}B_{kj}$ yang menjadi dasar operasi
  seperti $W_f[h_{t-1}, x_t]$ pada neural network dapat dipercaya diimplementasikan dengan benar oleh
  NumPy maupun PyTorch — keduanya hanyalah cara komputasi yang lebih cepat dari definisi matematis yang
  sama.

---

## 2. Perbandingan Waktu Eksekusi

```
=============================================
Metode                          Waktu (detik)
=============================================
Manual (3 for-loop)                  0.127694
NumPy (@)                            0.016413
PyTorch (matmul)                     0.031723
=============================================
NumPy lebih cepat 8x dibanding manual
PyTorch lebih cepat 4x dibanding manual
```

**Interpretasi:**

- Untuk matriks $100\times100$ (yaitu $10^6$ elemen output, masing-masing memerlukan 100 perkalian +
  100 penjumlahan → total sekitar $2\times10^{8}$ operasi *floating-point*), implementasi manual
  memerlukan **≈0,128 detik**, sedangkan NumPy hanya **≈0,016 detik** — sekitar **8× lebih cepat**.
- PyTorch (**≈0,032 detik**, 4× lebih cepat dari manual) sedikit lebih lambat dibanding NumPy pada
  kasus ini. Ini wajar dan **bukan berarti PyTorch "lebih buruk"** secara umum — pada matriks sekecil
  ini, *overhead* PyTorch (pembuatan objek tensor, pengecekan graf autograd, dispatch ke backend) belum
  bisa diimbangi oleh keunggulan performanya. Keunggulan PyTorch (dan dukungan GPU-nya) baru terasa
  signifikan pada matriks yang jauh lebih besar (ribuan × ribuan) atau saat menjalankan banyak operasi
  berantai sekaligus seperti pada training neural network sungguhan.
- Poin utama yang perlu digarisbawahi: **kedua library yang teroptimasi (NumPy & PyTorch) sudah jauh
  lebih cepat daripada loop Python murni**, bahkan untuk matriks yang tergolong kecil (100×100).
  Ini menjelaskan mengapa tidak ada framework deep learning yang menghitung perkalian matriks dengan
  loop Python murni — pada jaringan nyata, operasi semacam ini terjadi jutaan–miliaran kali per langkah
  training, sehingga selisih kecepatan 8×–ratusan× ini menjadi krusial (perbedaan antara training
  yang selesai dalam menit vs berjam-jam).

---

## 3. Turunan Manual pada Beberapa Titik Uji

```
x |    y = sin(x^2) |    dy/dx manual
---------------------------------------------
 -2.00 |       -0.756802 |        2.614574
 -1.00 |        0.841471 |       -1.080605
 -0.50 |        0.247404 |       -0.968912
  0.00 |        0.000000 |        0.000000
  0.50 |        0.247404 |        0.968912
  1.00 |        0.841471 |        1.080605
  1.50 |        0.778073 |       -1.884521
  2.00 |       -0.756802 |       -2.614574
  3.00 |        0.412118 |       -5.466782
```

**Interpretasi:**

- Nilai `dy/dx` di atas dihasilkan dari rumus manual $\dfrac{dy}{dx}=2x\cos(x^2)$, hasil penerapan
  *chain rule* pada $y=\sin(x^2)$ (turunan luar $\cos(u)$ dikalikan turunan dalam $2x$, dengan
  $u=x^2$).
- Perhatikan **simetri hasil**: pada $x=-2.00$ dan $x=2.00$, nilai $y$ sama persis ($-0.756802$) karena
  $\sin(x^2)$ adalah fungsi genap ($x^2$ selalu sama untuk $\pm x$). Namun nilai turunannya berlawanan
  tanda ($2.614574$ vs $-2.614574$) — ini konsisten dengan sifat matematis bahwa **turunan dari fungsi
  genap adalah fungsi ganjil**. Pola yang sama terlihat pada pasangan $(-1.00, 1.00)$ dan
  $(-0.50, 0.50)$.
- Pada $x=0.00$, baik $y$ maupun $dy/dx$ bernilai nol — masuk akal karena $\sin(0^2)=\sin(0)=0$, dan
  $2(0)\cos(0)=0$. Titik ini adalah titik potong dengan sumbu-x sekaligus titik kritis dari fungsi.
- Pada $x=1.50$ dan $x=2.00$, nilai $y$ maupun turunan mulai berosilasi lebih cepat dibanding di sekitar
  $x=0$ — ini adalah efek dari suku $x^2$ di dalam $\sin(\cdot)$: semakin besar $|x|$, semakin cepat
  argumen $\sin$ berubah, sehingga frekuensi osilasi fungsi (dan turunannya) meningkat secara non-linear.

---

## 4. Verifikasi dengan `torch.autograd`

```
x |   manual (2x cos(x^2)) |    autograd (x.grad) |    selisih
----------------------------------------------------------------------
 -2.00 |           2.6145744835 |         2.6145744835 |   0.00e+00
 -1.00 |          -1.0806046117 |        -1.0806046117 |   0.00e+00
 -0.50 |          -0.9689124217 |        -0.9689124217 |   0.00e+00
  0.00 |           0.0000000000 |         0.0000000000 |   0.00e+00
  0.50 |           0.9689124217 |         0.9689124217 |   0.00e+00
  1.00 |           1.0806046117 |         1.0806046117 |   0.00e+00
  1.50 |          -1.8845208682 |        -1.8845208682 |   0.00e+00
  2.00 |          -2.6145744835 |        -2.6145744835 |   0.00e+00
  3.00 |          -5.4667815713 |        -5.4667815713 |   0.00e+00
```

**Interpretasi:**

- Selisih antara turunan manual dan hasil `x.grad` dari `torch.autograd` adalah **`0.00e+00` di semua
  titik uji** — kecocokan sampai ke digit ke-10 di belakang koma. Ini bahkan lebih presisi dibanding
  hasil perkalian matriks di atas (yang menyisakan selisih $10^{-14}$), karena setiap titik di sini
  hanya melibatkan satu operasi turunan tunggal (bukan akumulasi $10^6$ penjumlahan), sehingga hampir
  tidak ada ruang untuk akumulasi galat pembulatan.
- Hasil ini membuktikan secara empiris bahwa `torch.autograd` **tidak melakukan estimasi numerik**
  (seperti *finite difference*, yang biasanya punya galat lebih besar), melainkan menghitung turunan
  secara **eksak secara simbolik** melalui *reverse-mode automatic differentiation* — yaitu, dengan
  menerapkan chain rule secara otomatis pada setiap operasi elementer di graf komputasi
  ($x \to x^2 \to \sin(\cdot)$), persis seperti langkah manual yang dilakukan di atas.
- **Relevansi terhadap neural network:** proses `y.backward()` di sini adalah versi paling sederhana
  dari *backpropagation*. Pada jaringan sungguhan — misalnya pada $f_t=\sigma(W_f[h_{t-1},x_t]+b_f)$ —
  graf komputasinya jauh lebih panjang (perkalian matriks → penjumlahan bias → fungsi aktivasi →
  ... → loss), tetapi mekanismenya identik: `autograd` menelusuri graf itu mundur (dari loss ke setiap
  parameter) sambil mengalikan turunan lokal di setiap simpul, sama seperti mengalikan $\cos(x^2)$
  dengan $2x$ di atas. Inilah alasan hasil verifikasi pada notebook ini bisa digeneralisasi menjadi
  keyakinan bahwa `autograd` dapat dipercaya untuk menghitung gradien pada model yang jauh lebih
  kompleks.

---

## Kesimpulan Umum

Dua eksperimen di notebook ini memvalidasi dua pilar utama deep learning secara empiris:

1. **Operasi matriks** — implementasi manual, NumPy, dan PyTorch menghasilkan nilai yang sama
   (selisih $\sim10^{-14}$, murni akibat floating-point), tetapi NumPy/PyTorch **4×–8× lebih cepat**
   pada kasus 100×100 ini, dan keunggulannya akan jauh lebih besar lagi pada ukuran matriks yang
   dipakai di jaringan nyata.
2. **Turunan/gradien** — hasil chain rule manual untuk $\sin(x^2)$ **cocok sampai presisi mesin**
   dengan hasil `torch.autograd`, mengonfirmasi bahwa mekanisme backpropagation pada PyTorch memang
   menerapkan chain rule secara eksak, bukan pendekatan.

Dengan demikian, notasi $f_t=\sigma(W_f[h_{t-1},x_t]+b_f)$ yang menjadi motivasi awal notebook ini dapat
dipahami sebagai gabungan dari dua operasi yang sudah diverifikasi kebenarannya di sini: **perkalian
matriks** (untuk forward pass) dan **turunan berantai** (untuk backward pass/training), keduanya
dijalankan secara efisien dan akurat oleh library seperti NumPy dan PyTorch.
