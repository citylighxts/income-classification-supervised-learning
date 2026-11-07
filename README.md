# Supervised learning pada Income: decision tree vs SVM vs ANN vs CNN

perbandingan empat metode supervised learning, **Decision Tree**, **support vector machine**,
**artificial neural network (MLP)**, dan **CNN 1D**, buat klasifikasi biner penghasilan seseorang
(`<=50K` atau `>50K` per tahun) di dataset **Income classification** (Kaggle). isinya EDA, data cleaning
dan preprocessing, tuning hyperparameter, evaluasi (akurasi, precision, recall, F1), uji signifikansi, dan
analisis kelebihan kekurangan tiap metode.

### How to run

dataset tidak disertakan di repositori (lisensi di halaman Kaggle tertulis Unknown). unduh
`income_evaluation.csv` dari https://www.kaggle.com/datasets/lodetomasi1995/income-classification lalu taruh
di folder yang sama dengan notebook, atau di folder `data/`.

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn torch jupyter
jupyter notebook income_classification_supervised_learning.ipynb
```

seluruh notebook makan waktu sekitar 8 sampai 9 menit.

## Isi notebook

| bagian | isi |
|---|---|
| eksplorasi data (EDA) | kamus variabel, statistik deskriptif, nilai `?` dan duplikat, distribusi kelas dan fitur, peluang `>50K` per kategori, outlier (IQR), `capitalGain` dan `capitalLoss`, korelasi |
| data cleaning & preprocessing | `?` jadi kategori `tidakDiketahui`, buang duplikat, `education` dan `fnlwgt` dibuang, `log(1 + x)` untuk fitur modal, one-hot, standardisasi, split 80/20 stratified |
| implementasi supervised learning | Decision Tree, SVM, ANN (`GridSearchCV` 5-fold), CNN 1D (PyTorch, early stopping) |
| perbandingan hasil | tabel metrik, selang kepercayaan bootstrap, performa per kelas, skor CV vs uji, uji McNemar, kestabilan 5 seed |
| analisis dan kesimpulan | tabel kelebihan kekurangan, kriteria pemilihan metode, keterbatasan |

## Ringkasan hasil

data uji 6.508 orang (split stratified 80/20, seed 42), metrik untuk kelas `>50K`:

| | Decision Tree | SVM (RBF) | ANN (MLP) | CNN 1D |
|---|---|---|---|---|
| akurasi | 0,8582 | 0,8617 | 0,8620 | 0,8642 |
| precision | 0,7279 | 0,7577 | 0,7445 | 0,7537 |
| recall | 0,6569 | 0,6263 | 0,6505 | 0,6480 |
| F1 | 0,6906 | 0,6858 | 0,6943 | 0,6968 |
| F1, rata-rata 5 seed | – | – | 0,7009 | 0,6872 |
| waktu latih | ± 0,08 detik | ± 5 detik | ± 0,5 detik | ± 78 detik |

**catatan:** keempat metode pada dasarnya setara. selang kepercayaan bootstrap saling tumpang tindih dan uji
McNemar tidak menemukan satu pun pasangan yang berbeda signifikan (p terkecil 0,09). yang membedakan adalah
jenis kesalahan (SVM paling hati-hati, precision tinggi tapi recall rendah) dan biaya latih, bukan akurasi.
semua model lemah di kelas `>50K` (recall 0,63 sampai 0,66) karena kelasnya cuma seperempat data dan ambang
keputusan masih 0,5.

rekomendasi praktis buat dataset ini adalah **Decision Tree**: performanya setara dengan metode yang lebih
rumit, paling cepat dilatih, deterministik, dan alasan prediksinya bisa dibaca. CNN tidak menambah akurasi yang
bisa dibuktikan meski jauh lebih mahal, wajar karena datanya tabular.

## Tech stack

- **Python**: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn
- **preprocessing** dengan `ColumnTransformer` dan `Pipeline` (log, `StandardScaler`, `OneHotEncoder`), di-fit hanya pada data latih
- **Decision Tree, SVM, MLP** dari scikit-learn dengan `GridSearchCV` (F1 kelas positif)
- **CNN 1D** pakai PyTorch (`BatchNorm`, `Dropout`, Adam, early stopping)
- **evaluasi:** akurasi, precision, recall, F1 (kelas positif dan macro), confusion matrix, bootstrap CI, uji McNemar

## Catatan reproduksi

semua seed dikunci (`randomState = 42`). bagian scikit-learn seharusnya menghasilkan angka yang sama
di mesin lain dengan versi pustaka yang mirip. bagian CNN dikunci pakai
`torch.use_deterministic_algorithms(True)` dan 1 thread supaya bisa diulang persis di mesin yang sama,
tapi belum tentu identik di mesin atau versi torch lain. tuning SVM dilakukan di subsampel 8.000 baris latih
karena waktu latih SVM naik kira-kira kuadratik terhadap jumlah data.

## Sumber dataset

Income classification [Dataset]. Kaggle.
https://www.kaggle.com/datasets/lodetomasi1995/income-classification, lisensi tidak disebutkan di halaman Kaggle.
