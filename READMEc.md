<img width="232" alt="image" src="https://github.com/user-attachments/assets/432e3cc3-2a52-4526-9e08-5c23749555cf" /># Laporan Proyek Machine Learning - Ica Nur Halimah
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

Data yang digunakan dalam proyek ini adalah dataset yang diambil dari Kaggle [Dataset Bitcoin Price (USD)](https://www.kaggle.com/datasets/aakashverma8900/bitcoin-price-usd) dengan kategori dataset, yaitu Currencies and Foreign Exchange. Dalam dataset tersebut terdapat sebuah file atau berkas dengan nama main.csv yang berekstensi (file format) .csv atau comma-separated values berukuran 28,44 MB.

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
   
   <img src="https://github.com/user-attachments/assets/49f420d9-6d74-4172-be3e-9ea5856cd9f2" alt="Handle Outliers" title="Handle Outliers" width="700" height="500">

    Seletah dilakukan pembersihan outliers menggunakan metode IQR (Inter Quartile Range), dapat dilihat bahwa outliers telah tidak ada dalam data.

2. Split Data

   Pembagian data dilakukan untuk memisahkan data keseluruhan menjadi dua (2) bagian, yaitu data latih (training data) dan data uji (testing data) dengan perbandingan rasio sebesar 90 : 10 menggunakan `train_test_split`. Kemudian diperoleh hasil pembagian data masing-masing, yaitu sebagai berikut,
   ```
   Total seluruh sampel : 188317
   Total data train     : 150653
   Total data test      : 37664
   ```

5. Normalisasi pada Fitur Numerik

   Min Max Normalisasi adalah penskalaan ulang data dari rentang asli sehingga semua nilai berada dalam rentang baru 0 dan 1. Penskalaan fitur agar terletak di antara nilai minimum dan maksimum, biasanya antara nol dan satu, atau menskalakan nilai absolut maksimum setiap fitur ke ukuran satuan. Motivasi untuk menggunakan MinMaxScaler adalah untuk mencakup ketahanan terhadap standar deviasi fitur yang sangat kecil.

   ```python
   scaler = MinMaxScaler()
   xtrain = scaler.fit_transform(train)
   xtest = scaler.transform(test)
   ```
     
### Model Development

Setelah dilakukannya tahap data preparation, selanjutnya adalah melakukan tahap persiapan model terlebih dahulu sebelum mengembangkan model menggunakan algoritma yang telah ditentukan. Tahap persiapan dataframe untuk analisis model menggunakan parameter index, yaitu train_mse dan test_mse, serta parameter columns yang merupakan algoritma yang akan digunakan untuk melakukan prediksi, yaitu algoritma K-Nearest Neighbors (KNN), Linear Regression (LR), dan Random Forest (RF).

```python
models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'LR', 'RF'])
```

Kemudian terapkan ketiga algoritma ke dalam model tersebut.
1. K-Nearest Neighbors

   KNN adalah algoritma machine learning yang memprediksi berdasarkan kedekatan data. Cara kerjanya sederhana: menyimpan semua data training, menghitung jarak ke data baru, memilih K tetangga terdekat, lalu mengambil keputusan dari mayoritas (klasifikasi) atau rata-rata (regresi) tetangga tersebut. Meski mudah dipahami dan fleksibel, KNN membutuhkan komputasi tinggi untuk dataset besar dan sensitif terhadap skala fitur.

   ```python
   knn = KNeighborsRegressor(n_neighbors=5,
    weights='uniform',
    metric='manhattan',
    p=1,
    algorithm='auto'
    )
   ```

   Kemudian akan dilakukan analisis prediksi error menggunakan Mean Squared Error (MSE) pada data latih (training data) dan data uji (testing data)

   ```python
   knn.fit(xtrain, ytrain)
   models.loc['train_mse','KNN'] = mean_squared_error(y_pred = knn.predict(xtrain), y_true=ytrain)
   models.loc['test_mse','KNN'] = mean_squared_error(y_pred = knn.predict(xtest), y_true=ytest)
   ```

2. Linear Regression

   Linear Regression adalah algoritma supervised learning yang memodelkan hubungan linear antara variabel input (independen, fitur) dan variabel output (dependen, target) dengan menyesuaikan garis lurus (atau hyperplane untuk multi-dimensi) ke data yang diamati.
   
   ```python
   lr = LinearRegression()
   ```
   Kemudian akan dilakukan analisis prediksi error menggunakan Mean Squared Error (MSE) pada data latih (training data) dan data uji (testing data)

   ```python
   lr.fit(xtrain, ytrain)
   models.loc['train_mse','LR'] = mean_squared_error(y_pred = lr.predict(xtrain), y_true=ytrain)
   models.loc['test_mse','LR'] = mean_squared_error(y_pred = lr.predict(xtest), y_true=ytest)
   ```

3. Random Forest

   Algoritma Random Forest merupakan algoritma supervised learning yang termasuk pada golongan ensemble (group) learning. Oleh karena itu, algoritma Random Forest terdiri dari beberapa model yang akan bekerja bersama-sama secara independen, dan prediksi dari setiap model ensemble akan digabungkan untuk membuat hasil prediksi akhir.

   ```python
   rf = RandomForestRegressor(
    n_estimators=200,         # Lebih banyak pohon untuk menangkap pola kompleks
    max_depth=15,             # Kedalaman cukup dalam tapi tidak berlebihan
    min_samples_split=5,      # Mencegah overfitting
    min_samples_leaf=2,       # Lebih stabil
    max_features='sqrt',      # Standar untuk regresi
    random_state=55,          # Konsistensi hasil
    n_jobs=-1                 # Memanfaatkan semua core CPU
   )
   ```

   Kemudian akan dilakukan analisis prediksi error menggunakan Mean Squared Error (MSE) pada data latih (training data) dan data uji (testing data)

   ```python
   rf.fit(xtrain, ytrain)
   models.loc['train_mse', 'RF'] = mean_squared_error(y_pred=rf.predict(xtrain), y_true=ytrain)
   models.loc['test_mse','RF'] = mean_squared_error(y_pred = rf.predict(xtest), y_true=ytest)
   ```

### Evaluation
    
Kemudian evaluasi dari ketiga model, yaitu algoritma K-Nearest Neighbors (KNN), Linear Regression (LR), dan Random Forest (RF) untuk masing-masing data latih (training data) dan data uji (testing data) dengan melihat tingkat error-nya menggunakan Mean Squared Error (MSE)

$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - y\_{pred}_i)^2$

di mana, nilai $N$ adalah jumlah dataset, nilai $y_i$ merupakan nilai sebenarnya, dan $y_pred$ yaitu nilai prediksinya.

Penggunaan metode metrik Mean Squared Error (MSE) memiliki kelebihan, yaitu cukup sederhana dan mudah dipahami dalam melakukan perhitungan. Meskipun begitu, terdapat kelemahan pada metrik ini, yaitu hasil akurasi prediksi yang kecil karena tidak dapat membandingan hasil peramalan tersebut dengan kenyataannya. 

```python
    mse = pd.DataFrame(columns=['train', 'test'], index=['KNN', 'LR', 'RF'])
    modelDict = {'KNN'     : knn, 'LR'      : lr, 'RF': rf}
    for name, model in modelDict.items():
        mse.loc[name, 'train'] = mean_squared_error(y_true=ytrain, y_pred=model.predict(xtrain))/1e3
        mse.loc[name, 'test']  = mean_squared_error(y_true=ytest,  y_pred=model.predict(xtest))/1e3
```

<img width="284" alt="image" src="https://github.com/user-attachments/assets/82d98798-ed93-44b6-9ccc-c9e9052a8d4f" />

Dari data tabel tersebut dapat divisualisasikan pada grafik batang berikut.

<img alt="image" src="https://github.com/user-attachments/assets/5fb8580e-4dfd-4c3a-b20f-c9c760ed3f91" />

Dari visualisasi diagram di atas dapat disimpulkan bahwa,

1. Model dengan algoritma Linear Regression memberikan nilai error yang paling kecil pada data test, yaitu sebesar 0.90.
2. Model dengan algoritma K-Nearest Neighbor memiliki tingkat error yang sedang di antara dua algoritma lainnya.
3. Model dengan algoritma Random Forest mengalami error yang paling besar pada data testing namun nilai error paling kecil pada data train dengan nilai training error sebesar 0.55, dan nilai testing error sebesar 1.15.

Selanjutnya adalah Melakukan pengujian prediksi dengan menggunakan beberapa nilai harga bitcoin dari data uji (testing)

<img width="738" alt="image" src="https://github.com/user-attachments/assets/0feff357-5f40-48ba-b6a3-2659e40337f4" />

Dapat dilihat prediksi pada model dengan algoritma Linier Regression memberikan hasil yang paling mendekati dengan nilai `y_true` jika dibandingkan dengan algoritma model yang lainnya.

Nilai `y_true` sebesar 38501.94 dan nilai prediksi `Linier Regression` sebesar 38559.8.

Kesimpulannya adalah model yang digunakan untuk melakukan prediksi harga bitcoin menghasilkan tingkat error yang paling rendah dengan menggunakan algoritma Linier Regression pada model yang telah dibangun.

## Referensi
[1] Zhang, W., Wang, P., & Li, X. (2020). Predicting the price of Bitcoin using deep learning. Annals of Operations Research, 1-28. DOI: 10.1007/s10479-020-03669-5

[2] McNally, S., Roche, J., & Caton, S. (2018). Predicting the Price of Bitcoin Using Machine Learning. In 26th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP). IEEE. DOI: 10.1109/PDP2018.2018.00060

[3] Altman, N. S. (1992). An Introduction to Kernel and Nearest-Neighbor Nonparametric Regression. The American Statistician, Vol. 46, No. 3, pp. 175–185

[4] Weisberg, S. (2005). Applied Linear Regression (3rd ed.). Wiley-Interscience.

[5] Scornet, E., Biau, G., & Vert, J. P. (2015). Consistency of random forests. Annals of Statistics, 43(4), 1716–1741. https://doi.org/10.1214/15-AOS1321
