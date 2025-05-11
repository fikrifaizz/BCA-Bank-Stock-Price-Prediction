# Laporan Proyek Machine Learning - Ica Nur Halimah
## Domain Proyek
Domain proyek ini akan membahas mengenai permasalahan dalam bidang ekonomi yang dibuat untuk mengetahui prediksi harga Bitcoin USD.

<img src="https://github.com/user-attachments/assets/244e1f16-1d4a-4142-9a48-a7d50a25ffaf" alt="Ilustrasi Harga Bitcoin USD" title="Ilustrasi Harga Bitcoin USD">

Perkembangan pesat teknologi finansial telah mendorong adopsi aset digital seperti Bitcoin sebagai instrumen investasi alternatif. Namun, volatilitas harga Bitcoin yang ekstrem kerap menjadi tantangan utama bagi investor dalam mengambil keputusan. Tidak jarang pergerakan harga Bitcoin dalam waktu singkat menunjukkan fluktuasi signifikan, yang dapat memicu kerugian maupun keuntungan besar [1]. Oleh karena itu, upaya untuk memprediksi harga Bitcoin menjadi sangat relevan, tidak hanya dalam konteks akademik, tetapi juga praktis bagi pelaku pasar.

Dengan meningkatnya ketersediaan data historis perdagangan dan kemajuan algoritma machine learning, analisis prediktif (predictive analytics) menjadi pendekatan yang tepat dalam memproyeksikan harga di masa mendatang. Teknik ini mengandalkan data historis (seperti harga pembukaan, penutupan, volume, dll.) untuk melatih model yang mampu mengenali pola tersembunyi di balik pergerakan harga [2]. Melalui pendekatan ini, investor dan sistem perdagangan otomatis dapat membuat keputusan yang lebih terukur dan berbasis data.

## Business Understanding
### Problem Statements
Berdasarkan latar belakang yang telah dijelaskan di atas, maka diperoleh rumusan masalah yang akan diselesaikan pada proyek ini, yaitu:
1. Bagaimana cara melakukan tahap persiapan data sebelum digunakan untuk membuat model machine learning?
2. Bagaimana cara membuat model machine learning untuk melakukan prediksi harga bitcoin?

### Goals
Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka didapatkan tujuan dari proyek ini, yaitu:
1. Melakukan tahap persiapan data (data preparation) sehingga data dapat digunakan pada model machine learning dengan baik.
2. Membuat model machine learning untuk melakukan analisis prediksi harga bitcoin dengan tingkat error yang cukup rendah.

### Solution Statements
Berdasarkan penjelasan di atas, terdapat beberapa solusi yang dapat dilakukan untuk dapat mencapai tujuan dari proyek ini, yaitu:
1. Tahap persiapan data (data preparation) dapat dilakukan dengan beberapa teknik, sebagai berikut:
    - Melakukan pembagian data menjadi 2, yaitu data latih (training data) dan data uji (testing data) dengan perbandingan rasio sebesar 80 : 20 yang akan digunakan ketika membangun model machine learning.
    - Melakukan standarisasi nilai pada data fitur numerik untuk mencegah terjadinya penyimpangan nilai data yang cukup besar.
2. Tahap pembuatan model machine learning akan digunakan 3 model. Algoritma yang akan digunakan adalah K-Nearest Neighbors, Linier Regression, dan Random Forest. Dari ketiga model tersebut akan dilakukan evaluasi performa dan kinerja masing-masing algoritma dan akan dipilih satu algoritma yang memberikan hasil prediksi yang terbaik.
    - Algoritma K-Nearest Neighbors

      K-Nearest Neighbors (KNN) adalah algoritma non-parametrik yang digunakan baik untuk klasifikasi maupun regresi. Dalam konteks regresi harga Bitcoin, KNN bekerja dengan cara mencari sejumlah k titik data (dalam hal ini: data historis perdagangan Bitcoin) yang paling dekat dengan titik data yang ingin diprediksi. Nilai prediksi diperoleh dari rata-rata nilai-nilai target dari k tetangga terdekat tersebut [3]. Cara kerja algoritma KNN sebagai berikut:

      Misalkan terdapat dataset pelatihan $D = {(x_1,y_1), (x_2,y_2), \dots, (x_n,y_n)}$ dengan:
      - $x_i \in \mathbb{R}^d$ adalah vektor fitur berdimensi $d$
      - $y_i \in \mathbb{R}$ adalah nilai target.

      Untuk melakukan prediksi pada data baru $x_q$, KNN regresi bekerja sebagai berikut:
      1. Hitung jarak euclidean antara $x_q$ dan seluruh titik $x_i$:

          $d(x_q, x_i) = \sqrt{\sum_{j=1}^{d} (x_{qj} - x_{ij})^2}$
        
      2. Ambil $k$ tetangga terdekat berdasarkan nilai jarak terkecil.
      3. Prediksi nilai target $\hat{y}_q$ dihitung dengan rata-rata nilai target dari tetangga:

         $\hat{y_q}= \frac{1}{k} \sum_{\substack{i \in \mathcal{N}_k(x_q)}} y_i$
         
         dimana $\mathcal{N}_k(x_q)$ adalah indeks dari $k$ tetangga terdekat terhadap $x_q$.

      Cara Kerja KNN dalam Proyek Ini

      Dalam proyek prediksi harga Bitcoin:
       - Fitur (input) dapat berupa harga sebelumnya: Open, High, Low, Close, Volume, dll.
    	 - Target (output) adalah harga penutupan (Close) yang ingin diprediksi.
       - Algoritma KNN akan mencari k waktu terdahulu yang memiliki pola pergerakan fitur paling mirip.
       - Semakin kecil nilai k, semakin sensitif model terhadap noise. Sebaliknya, semakin besar k, prediksi menjadi lebih stabil namun bisa kehilangan detail lokal.

      Adapun Kelebihan dari Algoritma KNN yaitu :
      - Tidak membuat asumsi bentuk fungsi regresi (nonparametrik).
      - Fleksibel dalam menangkap pola lokal.
      - Sederhana secara konsep dan dapat dijelaskan dengan analogi rata-rata lokal.
      - Kinerja seragam dalam variansi jika span tetap.
      - Baik untuk estimasi awal atau eksplorasi data.

      Adapun Kekurangan dari Algoritma KNN yaitu :
      - Estimasi bersifat “step function” atau diskret.
      - Sensitif terhadap pemilihan $k$ (span).
      - Tidak tahan terhadap outlier.

    - Algoritma Linier Regression

      Linear Regression adalah metode statistik untuk memodelkan hubungan linier antara variabel prediktor dan respons. Dengan pendekatan Ordinary Least Squares (OLS), model ini mencari garis terbaik yang meminimalkan selisih kuadrat antara nilai prediksi dan data aktual. Sifatnya yang sederhana dan interpretatif menjadikan regresi linier sebagai dasar penting dalam analisis data dan pemodelan prediktif [4]. Model matematis Linier Regression berganda dinyatakan sebagai bentuk:
  
      $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \varepsilon$
  
      Dalam notasi matriks, model tersebut dapat ditulis:
  
      $Y = X\beta + \varepsilon$
  
      Dengan:
      - $\mathbf{Y}$: vektor respons berukuran $n \times 1$
      - $\mathbf{X}$: matriks prediktor $n \times (p+1)$ (dengan kolom pertama biasanya berupa 1 untuk intercept)
      - $\boldsymbol{\beta}$: vektor parameter regresi $(p+1) \times 1$
      - $\boldsymbol{\varepsilon}$: vektor error residual $n \times 1$

      Estimasi parameter dengan metode Ordinary Least Squares (OLS) diberikan oleh:
  
      $\mathbf{\hat{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{Y}$
  
      Ini adalah solusi dari minimisasi residual sum of squares (RSS):

      $\text{RSS}(\boldsymbol{\beta}) = (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})$
  
      Cara kerja Linier Regression sebagai berikut:
      1. Pemilihan Variabel
  
         - Target (Y): Harga penutupan Bitcoin (misalnya kolom Close).
         - Fitur (X): Variabel input seperti Open, High, Low, Volume dan lainnya.

      2. Model menghitung koefisien $\beta$ menggunakan metode Ordinary Least Squares (OLS).
      3. Setelah dilatih, model digunakan untuk memprediksi harga penutupan Bitcoin ke depan berdasarkan input terbaru
      4. Model dievaluasi dengan metrik Mean Squared Error (MSE)

      Adapun Kelebihan dari Algoritma Linear Regression yaitu :
      - Mudah Dipahami dan Diimplementasikan.
      - Estimasi Optimal (BLUE).
      - Komputasi Efisien.
      - Dapat Digunakan Sebagai Dasar Model Lebih Kompleks.

      Adapun Kekurangan dari Algoritma Linear Regression yaitu :
      - Sensitif terhadap Outlier.
      - Asumsi Ketat.
      - Kurang Fleksibel untuk Pola Nonlinier.
      - Hasil Dapat Menipu Jika Asumsi Tidak Dipenuhi.

    - Algoritma Random Forest
  
      Random Forest adalah algoritma pembelajaran ansambel untuk regresi dan klasifikasi yang terdiri dari banyak pohon keputusan (decision trees) yang dibuat secara acak. Dalam konteks regresi, prediksi akhir adalah rata-rata prediksi dari semua pohon. Random Forest dibangun berdasarkan Resampling tanpa pengembalian dari data pelatihan (subsampling), Pemilihan acak subset fitur untuk setiap pemisahan pada node pohon, Penggunaan kriteria CART (squared error untuk regresi) dan Agregasi akhir dengan rata-rata prediksi semua pohon [5]. Model matematis untuk Random Forest dinyatakan sebagai berikut:
  
      Misalkan:
      - Data pelatihan: $\mathcal{D}_n = \{(X_1, Y_1), \dots, (X_n, Y_n)\}$,
      - Tujuan: memprediksi nilai $Y \in \mathbb{R}$ dari input $X \in \mathbb{R}^p$.
  
      Estimasi dengan Forest Finite:

      $\hat{m}{M,n}(x) = \frac{1}{M} \sum{j=1}^{M} \hat{m}(x; \Theta_j, \mathcal{D}_n)$
      - $\Theta_j$: parameter acak untuk membangun pohon ke-$j$,
      - $\hat{m}(x; \Theta_j, \mathcal{D}_n)$: prediksi dari pohon ke-$j$.
  
      Estimasi dengan Infinite Forest:
      $\hat{m}n(x) = \mathbb{E}\Theta[\hat{m}(x; \Theta, \mathcal{D}_n)]$
      - Rata-rata ekspektasi atas semua prediksi dari semua pohon acak,
      - Konsisten dalam arti $L^2: \mathbb{E}[(\hat{m}_n(X) - m(X))^2] \to 0$ saat $n \to \infty$.
     
      Cara Kerja Random Forest untuk Prediksi
      1.	Ambil Subset Data:
         - Untuk setiap pohon, ambil $a_n$ data dari $n$ data total, tanpa pengembalian.
      2.	Bangun Pohon:
         - Untuk setiap node pohon, pilih acak $mtry$ fitur dari $p$ total fitur.
         - Lakukan pemisahan berdasarkan kriteria CART (minimasi squared error).
      3. Prediksi:
         - Setiap pohon menghasilkan prediksi $\hat{y}_j$ untuk input $x$.
         - Prediksi akhir adalah:
            $\hat{y}(x) = \frac{1}{M} \sum_{j=1}^{M} \hat{y}_j(x)$
      4. Evaluasi Konsistensi:
         - Jika jumlah pohon $M \to \infty$, dan parameter seperti jumlah daun atau ukuran subsample dipilih dengan tepat, maka model terbukti konsisten secara statistik
  
      Adapun Kelebihan dari Random Forest yaitu :
      - Akurasi Tinggi dalam Praktik.
      - Cocok untuk Data Kecil dan Dimensi Tinggi.
      - Tidak Membutuhkan Banyak Parameter.
      - Konsistensi Terbukti Secara Teoretis.
      - Adaptif terhadap Sparsity.

      Adapun Kekurangan dari Algoritma Linear Regression yaitu :
      - Sulit Dianalisis secara Teoretis.
      - Tantangan Konsistensi di Dimensi Sangat Tinggi.
      - Estimasi Sulit Dikendalikan tanpa Subsampling Ketat.
      - Waktu Komputasi Besar pada Dataset Sangat Besar.

### Data Understanding
<img src="https://github.com/user-attachments/assets/694af6e1-c7d2-4f36-bc0e-56f84b75737a" alt="Kaggle Harga Bitcoin" title="Kaggle Harga Bitcoin">

Data yang digunakan dalam proyek ini adalah dataset yang diambil dari Kaggle Dataset Bitcoin Price (USD) dengan kategori dataset, yaitu Currencies and Foreign Exchange. Dalam dataset tersebut terdapat sebuah file atau berkas dengan nama main.csv yang berekstensi (file format) .csv atau comma-separated values berukuran 28,44 MB.

Kemudian dilakukan proses Exploratory Data Analysis (EDA) sebagai investigasi awal untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data dengan menggunakan teknik statistik dan representasi grafis atau visualisasi.
1. Deskripsi Variabel

   Berikut adalah informasi mengenai variabel-variabel yang terdapat pada dataset Harga Bitcoin adalah sebagai berikut

   <img src="https://github.com/user-attachments/assets/223e1d6a-5b0a-444c-9362-5d70c26f2cb3" alt="Deskripsi Variabel" title="Deskripsi Variabel" width="300" height="300">

   Dari gambar di atas dapat dilihat bahwa terdapat 188317 baris data dan 11 kolom atribut atau fitur. Di antaranya adalah delapan (8) atribut/variabel dengan tipe data float64 non-null dan tiga (3) atribut/variabel dengan tipe data int64[ns]. Berikut adalah keterangan untuk masing-masing variabel,
   - Open Time : Waktu pembukaan candle dalam bentuk timestamp Unix (milidetik).
   - Open : Harga Bitcoin (USD) saat awal periode (waktu pembukaan candle).
   - High : Harga tertinggi Bitcoin selama periode waktu tersebut.
   - Low : Harga terendah Bitcoin selama periode waktu tersebut.
   - Close : Harga penutupan Bitcoin pada akhir periode waktu tersebut.
   - Volume : Volume transaksi Bitcoin (dalam satuan BTC) yang terjadi selama periode tersebut.
   - Close Time : Waktu penutupan candle dalam bentuk timestamp Unix (milidetik).
   - Quote asset volume : Total volume transaksi dalam quote asset (biasanya USD), selama periode tersebut.
   - Number of trades : Jumlah total transaksi (trades) selama periode.
   - Taker buy base asset volume : Volume pembelian oleh “taker” dalam aset dasar (BTC).
   - Taker buy quote asset volume : Volume pembelian oleh “taker” dalam aset quote (USD).

2. Deskripsi Statistik

    <img src="https://github.com/user-attachments/assets/3ede3d4e-bb28-4299-8b70-a5dcd7147d8c" alt="Deskripsi Variabel" title="Deskripsi Variabel" width="600" height="500">

3. Pengecekan Missing Value

   Melakukan pengecekan apakah pada dataframe `data` terdapat nilai yang null/kosong.

   <img src="https://github.com/user-attachments/assets/4d8e043c-65fd-4ab3-a7cf-ffc4f26ee1d4" alt="Missing Value" title="Missing Value" width="600" height="500">

   Pada dataframe `data` ternyata tidak ditemukan adanya nilai null/kosong di setiap atribut/kolom.

4. Pengecekan Duplikat data

   Melakukan pengecekan apakah pada dataframe `data` terdapat nilai yang duplikat.

   <img width="235" alt="image" src="https://github.com/user-attachments/assets/cd33d950-29e4-4f6c-a3cc-2b4f08793421" />

   Pada dataframe `data` ternyata tidak ditemukan adanya data yang duplikat.

5. Pengecekan Outliers

   Melakukan pengecekan pada dataframe terdapat data outliers atau sampel data yang nilainya berada sangat jauh dari cakupan umum data utama yang dapat merusak hasil analisis data. Pengecekan dilakukan dengan cara visualisasi data menggunakan `boxplot` dengan bantuan library `seaborn`.

    <img src="https://github.com/user-attachments/assets/639380d3-a633-4caf-a20b-32c6fc223d24" alt="Outliers" title="Outliers" width="700" height="500">

   Dapat dilihat pada diagram `boxplot` diatas, terdapat 2 fitur numerik yang memiliki *outliers* seperti, `Volume` dan `Quote asset volume`. Untuk mengatasi *outliers*, dilakukan pendekatan menggunakan metode IQR (*Inter Quartile Range*) di `Data Preparation`.

4. Univariate Analysis

   Melakukan proses analisis data univariate pada fitur-fitur numerik. Proses analisis ini menggunakan bantuan visualisasi histogram untuk masing-masing fitur numerik.

   <img src="https://github.com/user-attachments/assets/a34c18a9-1e81-4e4b-97ef-fb41c94a1f26" alt="Univariate Analysis" title="Univariate Analysis" width="1000" height="800">

   Dari data histogram di atas diperoleh informasi, yaitu:

   - Data perdagangan menunjukkan penyebaran harga dengan beberapa cluster harga dominan.
   - Aktivitas transaksi (volume, jumlah perdagangan) didominasi oleh transaksi kecil, hanya sebagian kecil yang bernilai besar.
   - Distribusi waktu bersifat stabil/merata selama periode pengamatan.

6. Multivariate Analysis

   Melakukan visualisasi distribusi data pada fitur-fitur numerik dari data. Visualisasi dilakukan dengan bantuan library seaborn pairplot menggunakan parameter diag_kind, yaitu kde, untuk melihat perkiraan distribusi probabilitas antar fitur numerik.

   <img src="https://github.com/user-attachments/assets/39fc1859-d8c6-46ce-8e3e-1235fe34ec0a" alt="Multivariate Analysis" title="Multivariate Analysis">

8. Correlation Matrix with Heatmap

   Melakukan pengecekan korelasi antar fitur numerik dengan menggunakan visualisasi diagram heatmap correlation matrix.
   
   <img src="https://github.com/user-attachments/assets/021bb9c3-a569-4f4d-9a00-9b4043a04a4a" alt="Correlation Matrix" title="Correlation Matrix">

   Dapat dilihat pada diagram heatmap di atas memiliki range atau rentang angka dari -1.0 hingga 1.0 dengan keterangan sebagai berikut,

  - Jika semakin mendekati 1, maka korelasi antar fitur numerik semakin kuat bernilai positif.
  - Jika semakin mendekati 0, maka korelasi antar fitur numerik semakin rendah.
  - Jika semakin mendekati -1, maka korelasi antar fitur numerik semakin kuat bernilai negatif.

    Jika korelasi bernilai positif, berarti nilai kedua fitur numerik cenderung meningkat bersama-sama. Jika korelasi bernilai negatif, berarti nilai salah satu fitur numerik cenderung meningkat ketika nilai fitur numerik yang lain menurun.

9. Analisis Korelasi antar Fitur

   Fitur `Close` memiliki korelasi yang cukup rendah dengan Fitur `Volume`, `Quote asset volume`, `Number of trades`, `Taker buy base asset volume` dan `Taker buy quote asset volume`. Sehingga Fitur tersebut harus dihapuskan dari data karena sangat berkorelasi rendah dengan fitur `Close`. Namun pada penelitian kali ini tidak melibatkan waktu untuk prediksi sehingga fitur `Open time` dan `Close time` akan dihapuskan.

### Data Preparation
Pada tahap persiapan data atau data preparation dilakukan berdasarkan penjelasan yang sudah dipaparkan pada bagian Solution Statements. Tahap ini penting dilakukan untuk mempersiapkan data sehingga dapat digunakan untuk melatih model machine learning/deep learning dengan baik. Berikut adalah dua tahapan data preparation yang dilakukan, yaitu,

1. Menangani Outliers

   $IQR = Q3 − Q1$

   Kemudian membuat batas bawah dan batas atas untuk mencakup outliers dengan menggunakan,

   $Batas Bawah = Q1 − 1.5 * IQR$
   
   $Batas Atas = Q3 − 1.5 * IQR$

    Setelah dilakukan pembersihan outliers, dilakukan kembali visualisasi outliers untuk melakukan pengecekan kembali sebagai berikut,

    <img src="https://github.com/user-attachments/assets/84f69727-f24f-4a66-b271-b1fc02c3222a" alt="Handle Outliers" title="Handle Outliers" width="700" height="500">

    Dari gambar di atas dapat dilihat bahwa outliers telah berkurang. Meskipun outliers masih terdapat pada fitur Nilai, Volume dan Frekuensi, tetapi masih dalam batas aman.

2. Sorting Berdasarkan Tanggal

- Mengubah format atau tipe data pada kolom Tanggal dari object menjadi datetime64[ns]:
  ```python
  data['Tanggal'] = pd.to_datetime(data['Tanggal'])
  ```
- Karena tujuan dari penelitian ini adalah untuk memprediksi harga saham 1 hari kedepan maka dilakukan pengurutan berdasarkan tanggal dari terlama hingga terbaru
  ```python
  data = data.sort_values('Tanggal')
  ```

3. Penetapan Fitur yang digunakan

   Pada prediksi harga saham bank bca untuk 1 hari kedepan, fitur yang digunakan hanya fitur `Penutupan` sehingga selain fitur `Penutupan` tidak digunakan dalam prediksi.

4. Split Data

   Pembagian data dilakukan untuk memisahkan data keseluruhan menjadi dua (2) bagian, yaitu data latih (training data) dan data uji (testing data) dengan perbandingan rasio sebesar 80 : 20

   ```python
   split = int(len(data) * 0.8)
   train = data[:split]
   test = data[split:]
   ```
   Setelah itu mengubah bentuk data train dan test menjadi array 2D. Langkah ini diperlukan untuk kompatibilitas dengan algoritma machine learning dan deep learning.

   ```python
   train = train.values.reshape(-1, 1)
   test = test.values.reshape(-1, 1)
   ```

   `-1` berarti ukuran dimensi pertama disesuaikan otomatis, `1` menandakan satu fitur/kolom.

5. Normalisasi pada Fitur Numerik

   Min Max Normalisasi adalah penskalaan ulang data dari rentang asli sehingga semua nilai berada dalam rentang baru 0 dan 1. Penskalaan fitur agar terletak di antara nilai minimum dan maksimum, biasanya antara nol dan satu, atau menskalakan nilai absolut maksimum setiap fitur ke ukuran satuan. Motivasi untuk menggunakan MinMaxScaler adalah untuk mencakup ketahanan terhadap standar deviasi fitur yang sangat kecil.

   ```python
   scaler = MinMaxScaler()
   xtrain = scaler.fit_transform(train)
   xtest = scaler.transform(test)
   ```

6. Windowing Data

   Mengubah data time series menjadi urutan untuk prediksi berbasis jendela waktu (windowing). Pada dasarnya, cell ini adalah "mesin penerjemah" data time series menjadi potongan-potongan informasi yang bermakna bagi model machine learning. Proses ini memungkinkan model "belajar" bagaimana harga bergerak berdasarkan sejarah sebelumnya, seperti seorang analis yang mempelajari tren masa lalu untuk memprediksi masa depan. Intinya, windowing adalah "mesin waktu" yang mengubah data mentah menjadi cerita berurutan yang dapat dipahami oleh model machine learning/deep learning.

   - Windowing Machine learning

     ```python
     def create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
     return np.array(X), np.array(y)
     window_size = 30
     X_train_seq, y_train_seq = create_sequences(xtrain, window_size)
     X_test_seq, y_test_seq = create_sequences(xtest, window_size)
     X_train_seq = X_train_seq.reshape(X_train_seq.shape[0], -1)
     X_test_seq = X_test_seq.reshape(X_test_seq.shape[0], -1)
     ```
   - Windowing Deep Learning

     ```python
     def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
         ds = tf.data.Dataset.from_tensor_slices(series)
         ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
         ds = ds.flat_map(lambda w: w.batch(window_size + 1))
         ds = ds.shuffle(shuffle_buffer)
         ds = ds.map(lambda w: (w[:-1], w[1:]))
     return ds.batch(batch_size).prefetch(1)
     window_size=30
     batch_size=32
     shuffle_buffer_size=1000
     train_dataset = windowed_dataset(xtrain, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
     test_dataset = windowed_dataset(xtest, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
     ```
     Pada kedua windowing, proses dilakukan dengan mengambil 30 hari sebelumnya untuk memprediksi harga saham 1 hari kedepan.
     
### Modelling

Setelah dilakukannya tahap data preparation, selanjutnya adalah melakukan tahap persiapan model terlebih dahulu sebelum mengembangkan model menggunakan algoritma yang telah ditentukan. Tahap persiapan dataframe untuk analisis model menggunakan parameter index, yaitu train_mse dan test_mse, serta parameter columns yang merupakan algoritma yang akan digunakan untuk melakukan prediksi, yaitu algoritma XGBoost, Support Vector Regression (SVR), dan Long Short-Term Memory (LSTM).

```python
models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['XGBoost', 'SVR', 'LSTM'])
```

Kemudian terapkan ketiga algoritma ke dalam model tersebut.
1. XGBoost

   XGBoost (Extreme Gradient Boosting) bekerja dengan membangun kumpulan (ensemble) pohon keputusan secara berurutan, di mana setiap pohon baru berusaha memperbaiki kesalahan prediksi pohon sebelumnya. Algoritma ini menggunakan teknik gradient boosting, dengan meminimalkan fungsi loss melalui proses boosting iteratif. Setiap prediksi didasarkan pada agregasi (penjumlahan) hasil semua pohon, membuat model ini kuat dalam menangani data non-linear dan memiliki performa tinggi pada dataset tabular.

   ```python
   XGBoost = xgb.XGBRegressor()
   ```

   Kemudian akan dilakukan analisis prediksi error menggunakan Mean Squared Error (MSE) pada data latih (training data) dan data uji (testing data)

   ```python
   XGBoost.fit(X_train_seq, y_train_seq)
   models.loc['train_mse','XGBoost'] = mean_squared_error(y_pred = XGBoost.predict(X_train_seq), y_true=y_train_seq)
   models.loc['test_mse','XGBoost'] = mean_squared_error(y_pred = XGBoost.predict(X_test_seq), y_true=y_test_seq)
   ```

2. Support Vector Regression (SVR)

   SVR adalah penerapan Support Vector Machine (SVM) untuk regresi. Algoritma ini berusaha menemukan fungsi regresi terbaik yang memiliki deviasi maksimal ε dari semua titik data, sambil tetap menjaga model sesederhana mungkin. Konsep kerjanya adalah mencari hyperplane di ruang berdimensi tinggi dengan margin tertentu, dan memanfaatkan kernel trick (seperti RBF kernel) untuk menangkap pola non-linear pada data. SVR efektif dalam menangani prediksi dengan data yang memiliki noise atau fluktuasi.

   ```python
   svr = SVR()
   ```
   Kemudian akan dilakukan analisis prediksi error menggunakan Mean Squared Error (MSE) pada data latih (training data) dan data uji (testing data)

   ```python
   svr.fit(X_train_seq, y_train_seq.ravel())
   models.loc['train_mse','SVR'] = mean_squared_error(y_pred = svr.predict(X_train_seq), y_true=y_train_seq.ravel())
   models.loc['test_mse','SVR'] = mean_squared_error(y_pred = svr.predict(X_test_seq), y_true=y_test_seq.ravel())
   ```
   `.ravel()`: Mengubah array multidimensi menjadi 1D

3. Long Short-Term Memory

   Model LSTM yang dibangun merupakan pendekatan canggih untuk prediksi time series menggunakan deep learning. Alur kerjanya melibatkan sel memori dan tiga gerbang utama (input gate, forget gate, output gate) yang bersama-sama mengontrol aliran informasi masuk, tersimpan, dan keluar. Arsitektur model diawali dengan Sequential model dari TensorFlow Keras, yang memungkinkan pembangunan jaringan saraf berlapis. Struktur model terdiri dari dua layer LSTM berurutan - pertama dengan 64 unit dan kedua dengan 32 unit, keduanya menggunakan aktivasi ReLU. Layer pertama dikonfigurasi untuk mengembalikan urutan penuh, yang memungkinkan layer berikutnya menerima informasi komprehensif. Kemudian, layer Dense tunggal berfungsi sebagai output, menghasilkan prediksi numerik tunggal. Proses kompilasi menggunakan optimizer Adam dan loss function Mean Squared Error, yang merupakan pilihan umum untuk masalah regresi. Pelatihan dilakukan selama 10 epoch, dengan dataset pelatihan dan validasi yang terpisah untuk memastikan model tidak overfitting.

   ```python
   model_lstm = tf.keras.models.Sequential()
   model_lstm.add(tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(window_size, 1)))
   model_lstm.add(tf.keras.layers.LSTM(32, activation='relu'))
   model_lstm.add(tf.keras.layers.Dense(1))
   model_lstm.compile(optimizer='adam', loss='mean_squared_error')
   history = model_lstm.fit(train_dataset, epochs=10, validation_data=test_dataset)
   y_pred = model_lstm.predict(test_dataset)
   models.loc['train_mse', 'LSTM'] = model_lstm.evaluate(train_dataset, verbose=0)
   models.loc['test_mse', 'LSTM'] = model_lstm.evaluate(test_dataset)
   ```

### Evaluation
Evaluasi performa model dilakukan dengan menggunakan metrik **Mean Squared Error (MSE)**. MSE adalah ukuran kesalahan prediksi yang dihitung dengan cara mengambil rata-rata dari kuadrat selisih antara nilai aktual dengan nilai prediksi. Metrik ini banyak digunakan dalam regresi karena memberikan penalti yang lebih besar terhadap kesalahan prediksi yang signifikan (outlier). Semakin kecil nilai MSE, semakin mendekati nilai prediksi terhadap nilai aktual, sehingga model dianggap memiliki performa yang lebih baik secara kuantitatif. Proses dimulai dengan membuat DataFrame kosong bernama `results_df` yang memiliki dua index utama: `train_mse` dan `test_mse`. Ini memungkinkan perbandingan langsung performa model pada dataset pelatihan dan pengujian. Selanjutnya, sel mengisi DataFrame dengan nilai MSE untuk tiga model yang berbeda: XGBoost, SVR (Support Vector Regression), dan LSTM. Setiap kolom model akan berisi dua nilai - MSE untuk data latih dan data uji.
```python
results_df = pd.DataFrame(index=['train_mse', 'test_mse'])
results_df['XGBoost'] = [models.loc['train_mse', 'XGBoost'], models.loc['test_mse', 'XGBoost']]
results_df['SVR'] = [models.loc['train_mse', 'SVR'], models.loc['test_mse', 'SVR']]
results_df['LSTM'] = [models.loc['train_mse', 'LSTM'], models.loc['test_mse', 'LSTM']]
```
Dengan menampilkan `results_df`, peneliti dapat dengan mudah membandingkan performa model berdasarkan tingkat kesalahan prediksi pada dataset pelatihan dan pengujian. Semakin rendah nilai MSE, semakin baik model dalam memprediksi data. Berikut merupakan output perbandingan performa model:

<img src="https://github.com/user-attachments/assets/15974109-72d9-4bba-b9df-217d68d907ab" alt="MSE Model" title="MSE Model">

Dari gambar tersebut, model yang memiliki tingkat kestabilan antara data train dan test melalui perhitungan MSE adalah model LSTM, sehingga Model LSTM akan digunakan untuk prediksi harga saham bank bca untuk 1 hari kedepan.

```python
last_window = xtrain[-window_size:].reshape(1, -1)
last_window_reshaped = last_window.reshape(1, window_size, 1)
prediksi_hari_depan = model_lstm.predict(last_window_reshaped)
prediksi_asli = scaler.inverse_transform(prediksi_hari_depan)
print("Prediksi harga penutupan hari berikutnya:", prediksi_asli[0][0])
```
<img src="https://github.com/user-attachments/assets/fb9b7509-7d02-43c4-9e93-0dd6a500430e" alt="LSTM Prediction" title="LSTM Prediction">

Dari prediksi harga saham bank bca, model memprediksi bahwa untuk 1 hari kedepan akan diperoleh penutupan sebesar Rp. 9581.318.

Berdasarkan hasil evaluasi yang telah dilakukan menggunakan metrik Mean Squared Error (MSE), diperoleh temuan bahwa algoritma **Long Short-Term Memory (LSTM)** memiliki performa prediksi yang paling optimal dibandingkan dengan algoritma XGBoost dan Support Vector Regression (SVR). Hal ini ditunjukkan melalui nilai MSE yang lebih rendah dan stabil baik pada data pelatihan maupun data pengujian, mengindikasikan kemampuan LSTM dalam memodelkan pola historis data harga saham secara lebih efektif.

Melalui implementasi dan analisis yang dilakukan dalam penelitian ini, permasalahan utama yang diajukan sejak awal, yaitu bagaimana memprediksi harga saham Bank BCA untuk satu hari ke depan berdasarkan data historis harga saham, telah terjawab dengan baik. Tujuan penelitian untuk membandingkan performa beberapa algoritma machine learning dan deep learning dalam prediksi harga saham juga berhasil dicapai.

Temuan ini diharapkan dapat memberikan kontribusi sebagai referensi awal dalam pengembangan sistem prediksi harga saham berbasis machine learning, khususnya pada konteks pasar modal Indonesia. Untuk penelitian selanjutnya, disarankan dilakukan pengayaan variabel dengan memasukkan indikator teknikal atau faktor makroekonomi, serta eksplorasi arsitektur deep learning lainnya guna meningkatkan akurasi dan robustitas model prediksi.


## Referensi
[1] Zhang, W., Wang, P., & Li, X. (2020). Predicting the price of Bitcoin using deep learning. Annals of Operations Research, 1-28. DOI: 10.1007/s10479-020-03669-5

[2] McNally, S., Roche, J., & Caton, S. (2018). Predicting the Price of Bitcoin Using Machine Learning. In 26th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP). IEEE. DOI: 10.1109/PDP2018.2018.00060

[3] Altman, N. S. (1992). An Introduction to Kernel and Nearest-Neighbor Nonparametric Regression. The American Statistician, Vol. 46, No. 3, pp. 175–185

[4] Weisberg, S. (2005). Applied Linear Regression (3rd ed.). Wiley-Interscience.

[5] Scornet, E., Biau, G., & Vert, J. P. (2015). Consistency of random forests. Annals of Statistics, 43(4), 1716–1741. https://doi.org/10.1214/15-AOS1321
