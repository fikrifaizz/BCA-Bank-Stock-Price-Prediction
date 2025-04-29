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
    - Melakukan pembagian data menjadi 2, yaitu data latih (training data) dan data uji (testing data) dengan perbandingan rasio sebesar 90 : 10 yang akan digunakan ketika membangun model machine learning.
    - Melakukan standarisasi nilai pada data fitur numerik untuk mencegah terjadinya penyimpangan nilai data yang cukup besar.
2. Tahap pembuatan model machine learning akan digunakan 3 model dengan algoritma machine learning yang berbeda. Algoritma yang akan digunakan adalah K-Nearest Neighbor Algorithm, Random Forest Algorithm, dan Adaptive Boosting Algorithm. Dari ketiga model tersebut akan dilakukan evaluasi performa dan kinerja masing-masing algoritma dan akan dipilih satu algoritma yang memberikan hasil prediksi yang terbaik.

## Referensi
[1] Yahoo Finance, BBCA Historical Data, diakses dari https://finance.yahoo.com/quote/BBCA.JK/history

[2] Bursa Efek Indonesia, Profil Emiten BBCA, diakses dari https://www.idx.co.id/id/data-pasar/profil-emiten/bbca/

[3] Goodfellow, Ian, Bengio, Yoshua, Courville, Aaron. Deep Learning. MIT Press, 2016.

[4] Fischer, Thomas, dan Krauss, Christopher. Deep learning with long short-term memory networks for financial market predictions. Neurocomputing, 2018.

[5] Fama, Eugene F. Efficient Capital Markets: A Review of Theory and Empirical Work. Journal of Finance, 1970.
