# Operasi Matriks dan Turunan Manual

Notebook edukatif yang membangun intuisi bahwa **neural network adalah tumpukan operasi matriks**, dan
bahwa **backpropagation adalah aplikasi *chain rule*** — fondasi matematis untuk memahami notasi seperti:

$$
f_t = \sigma\big(W_f [h_{t-1}, x_t] + b_f\big)
$$

Project ini membandingkan implementasi **manual (Python murni)** dengan **NumPy** dan **PyTorch** untuk
dua operasi paling dasar di balik neural network: perkalian matriks dan turunan/gradien.

---

## 📂 Isi Repo

```
Operasi_Matriks_dan_Turunan_Manual.ipynb
INTERPRETASI.md
```

| File | Deskripsi |
|---|---|
| `Operasi_Matriks_dan_Turunan_Manual.ipynb` | Notebook utama: implementasi, eksperimen, dan penjelasan teori |
| `INTERPRETASI.md` | Pembahasan interpretatif atas setiap output notebook — memaknai angka verifikasi kebenaran hasil, perbandingan waktu eksekusi, dan kecocokan gradien manual vs. `autograd` dalam konteks teori di baliknya. |

---

## 🎯 Tujuan

1. Menunjukkan bahwa perkalian matriks $C_{ij}=\sum_k A_{ik}B_{kj}$ — dasar dari operasi seperti
   $W_f[h_{t-1}, x_t]$ — bisa diimplementasikan dari nol, dan hasilnya identik dengan `numpy`/`torch`.
2. Menunjukkan bahwa `torch.autograd` menghitung turunan secara **eksak** lewat *chain rule*, dengan
   memverifikasinya terhadap perhitungan turunan manual.
3. Membandingkan waktu eksekusi untuk memberi intuisi *mengapa* library teroptimasi selalu dipakai dalam
   praktik deep learning, bukan loop Python murni.

---

## 🧩 Struktur Notebook

**Bagian A — Perkalian Matriks**
- A.1 Implementasi manual dengan tiga `for`-loop bersarang
- A.2 Verifikasi kebenaran hasil (manual vs NumPy vs PyTorch) pada matriks 100×100
- A.3 Perbandingan waktu eksekusi

**Bagian B — Turunan Manual vs Autograd**
- B.1 Turunan manual $\dfrac{d}{dx}\sin(x^2) = 2x\cos(x^2)$ via *chain rule*
- B.2 Verifikasi dengan `torch.autograd` (`requires_grad=True` → `y.backward()` → `x.grad`)

**Bagian C — Kesimpulan** menghubungkan kedua eksperimen kembali ke notasi $f_t=\sigma(W_f[h_{t-1},x_t]+b_f)$.

---

## 📊 Hasil Eksperimen

### Verifikasi Kebenaran Hasil (Perkalian Matriks 100×100)

```
Manual vs NumPy  sama (allclose)? : True
NumPy  vs PyTorch sama (allclose)? : True
Manual vs PyTorch sama (allclose)? : True
Selisih absolut maksimum (manual vs NumPy): 1.07e-14
```

Ketiga implementasi menghasilkan nilai yang identik secara matematis; selisih $10^{-14}$ murni berasal
dari akumulasi galat pembulatan *floating-point*, bukan kesalahan implementasi.

### Perbandingan Waktu Eksekusi

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

NumPy dan PyTorch jauh lebih cepat karena memanggil rutin BLAS teroptimasi (C/Fortran, vektorisasi
SIMD), sesuatu yang tidak mungkin didapat dari interpreter Python murni.

### Turunan Manual $\dfrac{d}{dx}\sin(x^2)$

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

### Verifikasi dengan `torch.autograd`

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

Turunan manual dan `torch.autograd` cocok **sampai presisi mesin** (selisih `0.00e+00` di semua titik
uji) — mengonfirmasi bahwa *automatic differentiation* PyTorch menghitung gradien secara eksak lewat
*chain rule*, bukan pendekatan numerik.

> Lihat pembahasan lengkap tiap hasil di [`INTERPRETASI.md`](./INTERPRETASI.md).

---

## 🛠️ Prasyarat

- Python ≥ 3.8
- `numpy`
- `torch`
- `matplotlib` (untuk visualisasi opsional pada Bagian B)

Instalasi:

```bash
pip install numpy torch matplotlib
```

---

## ▶️ Cara Menjalankan

```bash
git clone https://github.com/alfarabihe/0.2_Matrix-Operations-and-Manual-Differentiation/
cd 0.2_Matrix-Operations-and-Manual-Differentiation
jupyter notebook Operasi_Matriks_dan_Turunan_Manual.ipynb
```

Jalankan seluruh cell secara berurutan (**Run All**). Waktu eksekusi pada bagian perbandingan performa
dapat sedikit berbeda tergantung spesifikasi mesin, tetapi kesimpulan relatif (NumPy/PyTorch lebih cepat
dari loop manual; ketiga hasil numerik identik) akan tetap konsisten.

---

## 📌 Lisensi & Disclaimer

- Project ini dilisensikan di bawah [MIT License](LICENSE).
- Notebook ini bersifat **edukatif**, bukan untuk keperluan produksi — implementasi manual dengan
  `for`-loop sengaja dibuat naif agar mudah dipahami, bukan untuk dioptimasi.
- Fokus utama adalah membangun **intuisi konseptual**: neural network = operasi matriks berlapis,
  training = chain rule otomatis via `autograd`.
