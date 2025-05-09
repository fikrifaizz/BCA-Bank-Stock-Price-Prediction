# Laporan Proyek Machine Learning - Fikri Faiz Zulfadhli
## Domain Proyek
Domain proyek ini akan membahas mengenai permasalahan dalam bidang ekonomi yang dibuat untuk mengetahui prediksi harga saham bank BCA berdasarkan data harga penutupan di setiap hari nya yang telah dikumpulkan mulai dari tahun 2021 hingga 2025.

<img src="https://github.com/user-attachments/assets/e6b82915-9c6a-449d-a277-b31869d72142" alt="Ilustrasi Harga Saham Bank BCA" title="Ilustrasi Harga Saham Bank BCA">

Saham PT Bank Central Asia Tbk. (BBCA) merupakan salah satu aset investasi paling likuid dan stabil di Bursa Efek Indonesia [1][2]. Sebagai bank swasta terbesar di Indonesia berdasarkan kapitalisasi pasar, kinerja BBCA sering menjadi indikator penting dalam menggambarkan kesehatan sektor keuangan nasional [2]. Dengan dinamika ekonomi global, perubahan suku bunga, serta volatilitas pasar keuangan, kemampuan untuk memprediksi pergerakan harga saham BBCA menjadi semakin relevan, baik bagi investor individu maupun institusi. Di tengah kompleksitas faktor-faktor yang mempengaruhi pasar, pengembangan model prediktif berbasis Machine Learning dan Deep Learning menjadi pendekatan modern yang menjanjikan untuk membaca pola pergerakan harga dari data historis secara lebih adaptif dan akurat [3][4]. Pendekatan ini sejalan dengan perkembangan teknologi dalam dunia keuangan yang mengutamakan kecepatan analisis dan pengambilan keputusan berbasis data.

Dalam proyek ini, data historis saham BBCA yang tersedia melalui platform resmi seperti Yahoo Finance [1] dan Bursa Efek Indonesia [2] akan digunakan sebagai dasar pemodelan. Literatur utama yang dijadikan rujukan meliputi Deep Learning oleh Ian Goodfellow et al. (MIT Press, 2016) [3] dan penelitian oleh Fischer dan Krauss (2018) mengenai penggunaan Long Short-Term Memory (LSTM) dalam prediksi pasar keuangan [4]. Proyek ini juga berangkat dari pemahaman atas prinsip Efficient Market Hypothesis yang dikembangkan oleh Eugene Fama, yang menyatakan bahwa pasar keuangan pada dasarnya mencerminkan seluruh informasi yang tersedia, namun tetap membuka kemungkinan terbentuknya pola jangka pendek yang dapat dimanfaatkan untuk prediksi [5]. Dengan landasan data yang valid dan kerangka metodologis yang terbukti secara akademik, proyek ini bertujuan untuk membangun model prediksi harga saham BBCA yang dapat menjadi dasar pengambilan keputusan investasi yang lebih terinformasi.

## Business Understanding
### Problem Statements
Berdasarkan latar belakang yang telah dijelaskan di atas, maka diperoleh rumusan masalah yang akan diselesaikan pada proyek ini, yaitu:
1. Bagaimana cara melakukan tahap persiapan data sebelum digunakan untuk membuat model machine learning/deep learning?
2. Bagaimana cara membuat model machine learning untuk melakukan prediksi harga saham bank bca untuk 1 hari kedepan?

### Goals
Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka didapatkan tujuan dari proyek ini, yaitu:
1. Melakukan tahap persiapan data (data preparation) sehingga data dapat digunakan pada model machine learning/deep learning dengan baik.
2. Membuat model machine learning/deep learning untuk melakukan analisis prediksi harga saham bank bca untuk 1 hari kedepan dengan tingkat error yang cukup rendah.

### Solution Statements
Berdasarkan penjelasan di atas, terdapat beberapa solusi yang dapat dilakukan untuk dapat mencapai tujuan dari proyek ini, yaitu:
1. Tahap persiapan data (data preparation) dapat dilakukan dengan beberapa teknik, sebagai berikut:
    - Melakukan pembagian data menjadi 2, yaitu data latih (training data) dan data uji (testing data) dengan perbandingan rasio sebesar 80 : 20 yang akan digunakan ketika membangun model machine learning/deep learning.
    - Melakukan standarisasi nilai pada data fitur numerik untuk mencegah terjadinya penyimpangan nilai data yang cukup besar.
2. Tahap pembuatan model machine learning dan deep learning akan digunakan 3 model dengan 2 algoritma machine learning yang berbeda dan 1 algoritma deep learning. Algoritma yang akan digunakan adalah XGBoost Regressor, Support Vector Regressor, dan Long Short-Term Memory Algorithm. Dari ketiga model tersebut akan dilakukan evaluasi performa dan kinerja masing-masing algoritma dan akan dipilih satu algoritma yang memberikan hasil prediksi yang terbaik.
    - Algoritma XGBoost

      XGBoost (Extreme Gradient Boosting) adalah algoritma ensemble learning berbasis gradient boosting yang dirancang untuk membangun model prediksi dengan akurasi tinggi dan skalabilitas luar biasa. XGBoost memperbaiki pendekatan gradient boosting tradisional melalui optimasi sistematis pada penggunaan memori, pengolahan data sparsity, regularisasi model, dan parallelisasi proses pembelajaran. Secara matematis, XGBoost bertujuan meminimalkan fungsi objektif gabungan antara fungsi loss (mengukur kesalahan prediksi) dan fungsi regularisasi (mengendalikan kompleksitas model), sehingga tidak hanya fokus pada fitting data tetapi juga menghindari overfitting [6].

      Model prediksi dibangun secara aditif, di mana pada setiap iterasi $t$, model baru $f_t(x)$ ditambahkan untuk mengurangi error dari prediksi sebelumnya. Fungsi objektif pada iterasi ke- $t$ dirumuskan sebagai [6]:
      
      $\mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)$

      dengan $\Omega(f) = \gamma T + \frac{1}{2} \lambda \lVert w \rVert^2$ di mana $T$ adalah jumlah daun dalam pohon, $w_j$ adalah bobot nilai di daun ke- $j$, dan $\gamma$, $\lambda$ adalah parameter regularisasi.

      Untuk efisiensi, fungsi loss didekati menggunakan Taylor expansion orde dua:

      $\tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^{n} \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$

      dengan $g_i$ dan $h_i$ berturut-turut sebagai gradien pertama dan kedua dari fungsi loss terhadap prediksi sebelumnya.

      Cara kerja XGBoost meliputi:

      - Menghitung gradien dan hessian untuk semua data.
      - Membangun pohon keputusan baru dengan memilih split yang memaksimalkan gain:

        $Gain = \frac{1}{2} \left[ \frac{\left( \sum_{i \in I_L} g_i \right)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{\left( \sum_{i \in I_R} g_i \right)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{\left( \sum_{i \in I} g_i \right)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$
      - Mengupdate prediksi dengan menambahkan kontribusi dari pohon baru.
      - Mengulangi proses hingga mencapai jumlah iterasi maksimum atau konvergensi.

      Adapun Kelebihan dari Algoritma XGBoost yaitu :
      - Sangat Cepat dan Efisien
      - Skalabilitas Tinggi
      - Regularisasi Terintegrasi
      - Mendukung Data Sparse
      - Approximate Learning (Quantile Sketch)

      Adapun Kekurangan dari Algoritma XGBoost yaitu :
      - Kompleksitas Tuning Hyperparameter
      - Butuh Sumber Daya untuk Dataset Sangat Besar
      - Kurang Efisien untuk Data Kecil Sederhana
    - Algoritma Support Vector Regressor

      Support Vector Regression (SVR) adalah algoritma regresi berbasis prinsip Support Vector Machines (SVM) yang bertujuan untuk menemukan fungsi $f(x)$ yang menyimpang maksimal sebesar $\varepsilon$ dari nilai aktual target $y_i$, sekaligus menjaga fungsi tetap sesederhana mungkin. SVR berupaya meminimalkan kompleksitas model dengan menjaga vektor bobot $w$ tetap kecil, menggunakan konsep $ε$-insensitive loss function. Artinya, prediksi yang meleset kurang dari $\varepsilon$ diabaikan, sementara deviasi lebih besar dari $\varepsilon$ dihukum secara proporsional. Pendekatan ini sangat sesuai untuk aplikasi prediksi harga saham seperti Bank BCA (BBCA), di mana toleransi terhadap deviasi kecil penting dalam memperhalus model terhadap fluktuasi minor pasar [7].

      Secara matematis, SVR memformulasikan masalah optimasi sebagai berikut [7]:

      $\text{minimize} \quad \frac{1}{2} \lVert w \rVert^2 + C \sum_{i=1}^{\ell} (\xi_i + \xi_i^*)$

      dengan kendala :

      $y_i - \langle w, x_i \rangle - b \leq \varepsilon + \xi_i$

      $\langle w, x_i \rangle + b - y_i \leq \varepsilon + \xi_i^*$

      $\xi_i, \xi_i^* \geq 0$

      di mana C adalah parameter trade-off antara kompleksitas model dan toleransi error ￼. Solusi optimal w dan b diperoleh melalui teknik optimasi dual menggunakan kernel k(x, x{\prime}), sehingga dalam prediksi non-linear, fungsi regresi diekspresikan sebagai:
$f(x) = \sum_{i=1}^{\ell} (\alpha_i - \alpha_i^*) k(x_i, x) + b.$

      Penggunaan kernel seperti Radial Basis Function (RBF) memungkinkan SVR menangkap hubungan non-linear antara variabel-variabel fundamental atau teknikal saham BBCA. Dengan pendekatan ini, SVR dapat membangun model prediktif yang robust, efisien, dan adaptif terhadap dinamika pasar saham Indonesia.

   - Algoritma Long Short-Term Memory

     Long Short-Term Memory (LSTM) adalah salah satu jenis jaringan saraf tiruan yang dikembangkan oleh Sepp Hochreiter dan Jürgen Schmidhuber pada tahun 1997. Arsitektur ini dirancang untuk mengatasi kelemahan utama pada Recurrent Neural Network (RNN), yaitu hilangnya gradien (vanishing gradient) ketika memproses data sekuensial dalam jangka panjang. LSTM sangat sesuai digunakan untuk data deret waktu (time series), termasuk dalam prediksi harga saham, karena kemampuannya dalam menyimpan dan mengelola informasi historis dalam periode waktu yang panjang [8].

     Sebelum membahas alur kerja LSTM secara detail, berikut adalah fungsi-fungsi aktivasi utama yang digunakan dalam proses perhitungan [8]:

     a. Fungsi Aktivasi
  
        - Fungsi Sigmoid : digunakan pada unit gerbang (gate), menghasilkan output dalam rentang [0, 1]:
  
             $f(x) = \frac{1}{1 + \exp(-x)}$
            
        - Fungsi aktivasi output: digunakan untuk transformasi nilai cell state menjadi output unit:
  
          $h(x) = \frac{2}{1 + \exp(-x)} - 1$
        - Fungsi kandidat memori: digunakan dalam proses pembentukan kandidat nilai yang akan disimpan di memori:
          $g(x) = \frac{4}{1 + \exp(-x)} - 2$
          
     b. Input Gate
  
     Input gate menentukan seberapa besar informasi baru $x_t$ akan ditambahkan ke memori internal pada waktu $t$. Aktivasi gerbang input dihitung berdasarkan sinyal dari waktu sebelumnya $y_u(t - 1)$:
  
     $net_{inj}(t) = \sum_u w_{inj,u} \cdot y_u(t - 1)$
  
     $y_{inj}(t) = f_{inj}(net_{inj}(t))$
    
     c. Output Gate

     Output gate mengontrol seberapa besar informasi dari memori internal dikeluarkan sebagai output pada waktu $t$:
  
     $net_{outj}(t) = \sum_u w_{outj,u} \cdot y_u(t - 1)$
  
     $y_{outj}(t) = f_{outj}(net_{outj}(t))$

     d. Memory Cell (State dan Output)
  
     Memory cell menyimpan informasi historis dalam bentuk internal state. Proses pembaruan memori cell terdiri dari dua bagian: perhitungan nilai kandidat memori, dan pembaruan internal state.
     
     - Kandidat memori dan pembaruan state:

       $net_{cj}(t) = \sum_u w_{cj,u} \cdot y_u(t - 1)$
       
       $s_{cj}(0) = 0, \quad s_{cj}(t) = s_{cj}(t - 1) + y_{inj}(t) \cdot g(net_{cj}(t))$
     - Output dari cell

       $y_{cj}(t) = y_{outj}(t) \cdot h(s_{cj}(t))$
     
     e. Output Layer

     Output akhir dari jaringan (misalnya prediksi harga) diperoleh dari aktivasi unit output yang menggabungkan sinyal dari cell-cell memori:

     $net_k(t) = \sum_{u: u \text{ not a gate}} w_{k,u} \cdot y_u(t - 1)$

     $y_k(t) = f_k(net_k(t))$

### Data Understanding
<img src="https://github.com/user-attachments/assets/50b06b86-ce51-438e-a5e4-011d8b1012a6" alt="Profil Bank BCA" title="Profil Bank BCA">

Data yang digunakan dalam proyek ini adalah data yang diambil dari [Bursa Efek Indonesia](https://www.idx.co.id/id/perusahaan-tercatat/profil-perusahaan-tercatat/BBCA) mulai dari tahun 2021 hingga 2025. Dari data tersebut, masih perlu dilakukan penyesuaian hingga dataset dapat benar-benar digunakan

Kemudian dilakukan proses Exploratory Data Analysis (EDA) sebagai investigasi awal untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data dengan menggunakan teknik statistik dan representasi grafis atau visualisasi.
1. Deskripsi Variabel

   Berikut adalah informasi mengenai variabel-variabel yang terdapat pada dataset Harga Saham Bank BCA adalah sebagai berikut

   <img src="https://github.com/user-attachments/assets/17d235e8-7dc3-4050-b9bc-ca85cea1c434" alt="Deskripsi Variabel" title="Deskripsi Variabel" width="300" height="300">

   Dari gambar di atas dapat dilihat bahwa terdapat 1.000 baris data dan 9 kolom atribut atau fitur. Di antaranya adalah delapan (8) atribut/variabel dengan tipe data int64 non-null dan satu (1) atribut/variabel dengan tipe data datetime64[ns]. Berikut adalah keterangan untuk masing-masing variabel,
   - No : Nomor urut data/informasi pada tabel.
   - Tanggal : Tanggal terjadinya data harga saham tersebut (berformat YYYY-MM-DD).
   - Harga Pembukaan : Harga saham saat pasar dibuka pada hari itu.
   - Tertinggi : Harga tertinggi yang dicapai saham selama hari perdagangan.
   - Terendah : Harga terendah yang dicapai saham selama hari perdagangan.
   - Penutupan : Harga saham saat pasar ditutup pada hari itu.
   - Volume : Jumlah lembar saham yang diperdagangkan selama hari tersebut.
   - Nilai : Total nilai perdagangan saham pada hari itu.
   - Frekuensi : Jumlah transaksi yang berlangsung selama hari perdagangan tersebut.

2. Deskripsi Statistik

    <img src="https://github.com/user-attachments/assets/9461713c-3c9f-40b4-ab46-f6b1c14e3526" alt="Deskripsi Variabel" title="Deskripsi Variabel" width="600" height="500">

3. Menangani Outliers

    Outliers merupakan sampel data yang nilainya berada sangat jauh dari cakupan umum data utama yang dapat merusak hasil analisis data. Berikut adalah visualisasi boxplot untuk melakukan pengecekan keberadaan outliers.

    <img src="https://github.com/user-attachments/assets/f4101397-1200-4c09-aa78-3dca62306e39" alt="Outliers" title="Outliers" width="700" height="500">

    Berdasarkan gambar tersebut, terdapat outliers pada semua fitur kecuali fitur No. Sehingga dilakukan proses pembersihan outliers dengan metode IQR (Inter Quartile Range).

    $IQR = Q3 − Q1$

   Kemudian membuat batas bawah dan batas atas untuk mencakup outliers dengan menggunakan,

   $Batas Bawah = Q1 − 1.5 * IQR$
   
   $Batas Atas = Q3 − 1.5 * IQR$

    Setelah dilakukan pembersihan outliers, dilakukan kembali visualisasi outliers untuk melakukan pengecekan kembali sebagai berikut,

    <img src="https://github.com/user-attachments/assets/84f69727-f24f-4a66-b271-b1fc02c3222a" alt="Handle Outliers" title="Handle Outliers" width="700" height="500">

    Dari gambar di atas dapat dilihat bahwa outliers telah berkurang. Meskipun outliers masih terdapat pada fitur Nilai, Volume dan Frekuensi, tetapi masih dalam batas aman.

4. Univariate Analysis

   Melakukan proses analisis data univariate pada fitur-fitur numerik. Proses analisis ini menggunakan bantuan visualisasi histogram untuk masing-masing fitur numerik.

   <img src="https://github.com/user-attachments/assets/999ef5ef-eab0-43a4-92d3-ae2a6e998020" alt="Univariate Analysis" title="Univariate Analysis" width="1000" height="800">

   Dari data histogram di atas diperoleh informasi, yaitu:

   - No: Nilai berkisar 0-16, mayoritas data berada di rentang 8-14.
   - Tanggal: Data terkumpul dari 2021-2025, menunjukkan fluktuasi periodik.
   - Harga Pembukaan: Distribusi bimodal dengan puncak di sekitar 9000 dan 7500-8000.
   - Tertinggi: Puncak di sekitar 8500-9000, nilai maksimum sekitar 40.
   - Terendah: Puncak di sekitar 9000, pola serupa dengan data "Tertinggi".
   - Penutupan: Puncak di sekitar 9000, konsisten dengan data harga lainnya.
   - Volume: Distribusi condong kanan, mayoritas di sekitar 0,6-0,8 (x10^8).
   - Nilai: Condong kanan, konsentrasi di sekitar 0,5-0,8 (x10^12).
   - Frekuensi: Condong kanan, mayoritas berada di 10000-15000.

5. Multivariate Analysis

   Melakukan visualisasi distribusi data pada fitur-fitur numerik dari data. Visualisasi dilakukan dengan bantuan library seaborn pairplot menggunakan parameter diag_kind, yaitu kde, untuk melihat perkiraan distribusi probabilitas antar fitur numerik.

   <img src="https://github.com/user-attachments/assets/f6eeae97-51d1-4b5e-b8bb-af6a0052250e" alt="Multivariate Analysis" title="Multivariate Analysis">

6. Correlation Matrix with Heatmap

   Melakukan pengecekan korelasi antar fitur numerik dengan menggunakan visualisasi diagram heatmap correlation matrix.

   <img src="https://github.com/user-attachments/assets/f7ec0d76-e046-4f57-b99b-9dc44b18c2c0" alt="Correlation Matrix" title="Correlation Matrix">

### Data Preparation
Pada tahap persiapan data atau data preparation dilakukan berdasarkan penjelasan yang sudah dipaparkan pada bagian Solution Statements. Tahap ini penting dilakukan untuk mempersiapkan data sehingga dapat digunakan untuk melatih model machine learning/deep learning dengan baik. Berikut adalah dua tahapan data preparation yang dilakukan, yaitu,
1. Sorting Berdasarkan Tanggal

- Mengubah format atau tipe data pada kolom Tanggal dari object menjadi datetime64[ns]:
  ```python
  data['Tanggal'] = pd.to_datetime(data['Tanggal'])
  ```
- Karena tujuan dari penelitian ini adalah untuk memprediksi harga saham 1 hari kedepan maka dilakukan pengurutan berdasarkan tanggal dari terlama hingga terbaru
  ```python
  data = data.sort_values('Tanggal')
  ```

2. Penetapan Fitur yang digunakan

   Pada prediksi harga saham bank bca untuk 1 hari kedepan, fitur yang digunakan hanya fitur `Penutupan` sehingga selain fitur `Penutupan` tidak digunakan dalam prediksi.

3. Split Data

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

4. Normalisasi pada Fitur Numerik

   Min Max Normalisasi adalah penskalaan ulang data dari rentang asli sehingga semua nilai berada dalam rentang baru 0 dan 1. Penskalaan fitur agar terletak di antara nilai minimum dan maksimum, biasanya antara nol dan satu, atau menskalakan nilai absolut maksimum setiap fitur ke ukuran satuan. Motivasi untuk menggunakan MinMaxScaler adalah untuk mencakup ketahanan terhadap standar deviasi fitur yang sangat kecil.

   ```python
   scaler = MinMaxScaler()
   xtrain = scaler.fit_transform(train)
   xtest = scaler.transform(test)
   ```

5. Windowing Data

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
[1] Yahoo Finance, BBCA Historical Data, diakses dari https://finance.yahoo.com/quote/BBCA.JK/history

[2] Bursa Efek Indonesia, Profil Emiten BBCA, diakses dari https://www.idx.co.id/id/data-pasar/profil-emiten/bbca/

[3] Goodfellow, Ian, Bengio, Yoshua, Courville, Aaron. Deep Learning. MIT Press, 2016.

[4] Fischer, Thomas, dan Krauss, Christopher. Deep learning with long short-term memory networks for financial market predictions. Neurocomputing, 2018.

[5] Fama, Eugene F. Efficient Capital Markets: A Review of Theory and Empirical Work. Journal of Finance, 1970.

[6] Chen, T., University of Washington, Guestrin, C., & University of Washington. (2016). XGBOOST: a scalable tree boosting System (p. 785) [Journal-article]. http://dx.doi.org/10.1145/2939672.2939785

[7] Smola, A., & Scholkopf, B. (2004). A tutorial on support vector regression. Statistics and Computing, 14, 199–222.

[8] Hochreiter, S., PhD, Schmidhuber, J., Ronald Williams, Fakultät für Informatik, Technische Universität München, & IDSIA. (1997). Long Short-Term memory. In Massachusetts Institute of Technology, Neural Computation (Vols. 9–9, pp. 1735–1780). https://deeplearning.cs.cmu.edu/F23/document/readings/LSTM.pdf
