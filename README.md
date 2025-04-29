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

      XGBoost (Extreme Gradient Boosting) adalah algoritma ensemble learning berbasis gradient boosting yang dirancang untuk membangun model prediksi dengan akurasi tinggi dan skalabilitas luar biasa. XGBoost memperbaiki pendekatan gradient boosting tradisional melalui optimasi sistematis pada penggunaan memori, pengolahan data sparsity, regularisasi model, dan parallelisasi proses pembelajaran. Secara matematis, XGBoost bertujuan meminimalkan fungsi objektif gabungan antara fungsi loss (mengukur kesalahan prediksi) dan fungsi regularisasi (mengendalikan kompleksitas model), sehingga tidak hanya fokus pada fitting data tetapi juga menghindari overfitting.

      Model prediksi dibangun secara aditif, di mana pada setiap iterasi $t$, model baru $f_t(x)$ ditambahkan untuk mengurangi error dari prediksi sebelumnya. Fungsi objektif pada iterasi ke- $t$ dirumuskan sebagai:
      
      $\mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)$

      dengan $\Omega(f) = \gamma T + \frac{1}{2} \lambda \|w\|^2$ di mana $T$ adalah jumlah daun dalam pohon, $w_j$ adalah bobot nilai di daun ke- $j$, dan $\gamma$, $\lambda$ adalah parameter regularisasi.

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

      Support Vector Regression (SVR) adalah algoritma regresi berbasis prinsip Support Vector Machines (SVM) yang bertujuan untuk menemukan fungsi $f(x)$ yang menyimpang maksimal sebesar $\varepsilon$ dari nilai aktual target $y_i$, sekaligus menjaga fungsi tetap sesederhana mungkin. SVR berupaya meminimalkan kompleksitas model dengan menjaga vektor bobot $w$ tetap kecil, menggunakan konsep $ε$-insensitive loss function. Artinya, prediksi yang meleset kurang dari $\varepsilon$ diabaikan, sementara deviasi lebih besar dari $\varepsilon$ dihukum secara proporsional. Pendekatan ini sangat sesuai untuk aplikasi prediksi harga saham seperti Bank BCA (BBCA), di mana toleransi terhadap deviasi kecil penting dalam memperhalus model terhadap fluktuasi minor pasar.

      Secara matematis, SVR memformulasikan masalah optimasi sebagai berikut:

      $\text{minimize} \quad \frac{1}{2} \lVert w \rVert^2 + C \sum_{i=1}^{\ell} (\xi_i + \xi_i^*)$

      dengan kendala :

      $y_i - \langle w, x_i \rangle - b \leq \varepsilon + \xi_i$

      $\langle w, x_i \rangle + b - y_i \leq \varepsilon + \xi_i^*$

      $\xi_i, \xi_i^* \geq 0$

      di mana C adalah parameter trade-off antara kompleksitas model dan toleransi error ￼. Solusi optimal w dan b diperoleh melalui teknik optimasi dual menggunakan kernel k(x, x{\prime}), sehingga dalam prediksi non-linear, fungsi regresi diekspresikan sebagai:
$f(x) = \sum_{i=1}^{\ell} (\alpha_i - \alpha_i^*) k(x_i, x) + b.$

      Penggunaan kernel seperti Radial Basis Function (RBF) memungkinkan SVR menangkap hubungan non-linear antara variabel-variabel fundamental atau teknikal saham BBCA. Dengan pendekatan ini, SVR dapat membangun model prediktif yang robust, efisien, dan adaptif terhadap dinamika pasar saham Indonesia.

   - Algoritma Long Short-Term Memory

     

      
      

## Referensi
[1] Yahoo Finance, BBCA Historical Data, diakses dari https://finance.yahoo.com/quote/BBCA.JK/history

[2] Bursa Efek Indonesia, Profil Emiten BBCA, diakses dari https://www.idx.co.id/id/data-pasar/profil-emiten/bbca/

[3] Goodfellow, Ian, Bengio, Yoshua, Courville, Aaron. Deep Learning. MIT Press, 2016.

[4] Fischer, Thomas, dan Krauss, Christopher. Deep learning with long short-term memory networks for financial market predictions. Neurocomputing, 2018.

[5] Fama, Eugene F. Efficient Capital Markets: A Review of Theory and Empirical Work. Journal of Finance, 1970.
