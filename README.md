# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Maju adalah perusahaan multinasional yang telah berdiri sejak tahun 2000 dan memiliki lebih dari 1.000 karyawan yang tersebar di seluruh Indonesia. Sebagai entitas bisnis berskala besar, perusahaan ini terus berkembang dan menjalankan operasi di berbagai wilayah serta sektor strategis. Namun, di tengah pertumbuhan tersebut, perusahaan menghadapi tantangan serius dalam manajemen sumber daya manusia, khususnya terkait dengan tingginya tingkat keluar-masuk karyawan (attrition rate) yang telah melebihi 10%. Tingkat attrition yang tinggi ini menimbulkan kekhawatiran akan meningkatnya biaya rekrutmen, hilangnya talenta berpengalaman, serta potensi penurunan produktivitas dan stabilitas tim kerja.

Menanggapi permasalahan ini, manajemen HR Jaya Jaya Maju berinisiatif untuk mengidentifikasi akar penyebab dari tingginya attrition melalui pendekatan analisis data. Dengan menganalisis berbagai faktor seperti karakteristik demografis, kepuasan kerja, kompensasi, dan pengalaman karyawan, perusahaan berharap dapat memahami pola-pola yang mendorong keputusan resign. Informasi ini akan menjadi dasar dalam merumuskan strategi retensi karyawan yang lebih tepat sasaran, efisien, dan berdampak positif terhadap kinerja jangka panjang perusahaan.

### Permasalahan Bisnis

1. Tingginya attrition rate yang melebihi 10% 
2. Kesulitan dalam mengidentifikasi faktor-faktor penyebab tingginya attrition karyawan 
3. Kurangnya insight mengenai pola pengunduran diri karyawan berdasarkan usia, jabatan, atau kompensasi 
4. Belum adanya strategi HR yang efektif untuk menekan tingkat keluar masuk karyawan

### Cakupan Proyek

Proyek ini bertujuan untuk mengidentifikasi faktor-faktor utama yang memengaruhi tingginya tingkat attrition karyawan di perusahaan menggunakan pendekatan data science. Proyek ini akan memberikan gambaran visual, analisis statistik, dan rekomendasi berbasis data kepada pihak manajemen HR.
Ruang Lingkup Pekerjaan:
1. Persiapan dan Pemuatan Data 
   - Mengunduh data dari Dicoding Academy
   - Memuat Data hasil dari unduhan sebelumnya
2. Pembersihan dan Pemahaman Data
   - Menangani nilai kosong (missing values), terutama pada variabel target attrition. 
   - Mengidentifikasi kolom numerik dan kategorikal. 
   - Mengecek data duplikat.
3. Eksplorasi Data (Exploratory Data Analysis / EDA)
   - Visualisasi distribusi dan outlier menggunakan boxplot untuk setiap fitur numerik. 
   - Analisis tingkat attrition berdasarkan:
     - Departemen (department)
     - Jabatan (jobrole)
     - Kelompok umur (age)
     - Kepuasan kerja (jobsatisfaction dan kolom terkait)
   - Menggunakan grafik batang (bar chart) untuk memudahkan interpretasi.
4. Visualisasi dan Interpretasi
   - Menyimpan dan menyusun visualisasi grafik sebagai bahan presentasi atau laporan. 
   - Menyampaikan insight berdasarkan korelasi visual antara variabel dan tingkat attrition.

### Persiapan

Dataset yang digunakan dalam proyek ini adalah data karyawan dari perusahaan Jaya Jaya Maju yang berisi informasi demografis, data pekerjaan, dan status attrition karyawan. Dataset ini diperoleh dari [Dicoding Academy](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee).


#### Setup environment - Looker Studio:

1. Siapkan Data di Google Sheets
2. Buka Looker Studio & Hubungkan Data
3. Atur Tipe Data
4. Rancang Komponen Dashboard
5. Tambahkan Judul & Desain
6. Selesai

#### Setup environment - Python dengan Virtualenv

1. Membuat virtual environment

    ```python -m venv venv```
2. Mengaktifkan virtual environment

    - Untuk Windows
        ```venv\Scripts\activate```        

    - Untuk MacOS/Linux
        ```source venv/bin/activate```
3. Menginstall dependencies

    ```pip install -r requirements.txt```

#### Menjalankan File Prediksi
```python prediction.py```

## Business Dashboard
<img src="https://github.com/user-attachments/assets/81d51743-a18c-4bad-8fde-004ebe2cf19f" alt="Dashboard" title="Dashboard">

Dashboard ini dirancang untuk membantu tim HR Jaya Jaya Maju:
- Mengidentifikasi pola dan faktor utama yang memengaruhi tingkat attrition karyawan
- Mendeteksi kelompok rentan yang memiliki kemungkinan lebih tinggi untuk keluar
- Menyediakan dasar data bagi pengambilan kebijakan retensi karyawan yang lebih tepat

Berbasis Looker Studio, dashboard ini memiliki fitur interaktif seperti filter dan grafik yang memudahkan eksplorasi data. Elemen-elemen utama meliputi:

- KPI Overview:
   - Total Karyawan: 1.470
   - Tingkat Attrition (%): 13,40%
   - Rata-rata Penghasilan Bulanan: 6.502,93
   - Rata-rata Masa Kerja: 7,01 tahun
   - Rata-rata Usia Karyawan: 36,92 tahun

- Visualisasi Insight:
   1. Attrition berdasarkan Jarak Rumah ke Kantor (Distance From Home)
   Menunjukkan bahwa jarak lebih jauh memiliki kecenderungan attrition lebih tinggi.
   2. Attrition berdasarkan Work Life Balance & Job Satisfaction
   Kombinasi WLB rendah dan kepuasan kerja rendah menghasilkan attrition rate tertinggi hingga 23%.
   3. Attrition berdasarkan Overtime
   Karyawan yang sering lembur memiliki attrition rate lebih tinggi (27,7%).
   4. Attrition berdasarkan Status Pernikahan dan Gender
   Karyawan single memiliki kecenderungan keluar yang lebih tinggi dibanding married atau divorced, dengan variasi antara laki-laki dan perempuan.
   5. Attrition berdasarkan Usia
   Rentang usia muda (18–30 tahun) menunjukkan rasio attrition lebih tinggi dari usia lainnya.
   6. Attrition berdasarkan Job Role dan Level Jabatan
   Level jabatan rendah (misal Sales Rep, Lab Technician) memiliki tingkat attrition yang lebih tinggi dibanding level manajerial.

🔗 Link Dashboard:
[Buka di Looker Studio](https://lookerstudio.google.com/reporting/584bf7ee-3d16-4fad-858f-44116fdd8316/page/azeLF)
## Conclusion
Proyek analisis ini berhasil mengidentifikasi berbagai faktor yang berkontribusi terhadap tingginya attrition rate di perusahaan Jaya Jaya Maju, yang tercatat mencapai 13,4%. Melalui pemanfaatan dashboard interaktif di Looker Studio, ditemukan bahwa attrition lebih banyak terjadi pada karyawan dengan karakteristik tertentu seperti usia muda, status lajang, jarak rumah jauh dari kantor, sering lembur, serta tingkat kepuasan kerja dan work-life balance yang rendah.

Temuan ini memberikan wawasan yang dapat ditindaklanjuti oleh departemen HR untuk merancang strategi retensi yang lebih tepat sasaran. Beberapa tindakan yang dapat dipertimbangkan antara lain: peninjauan ulang kebijakan lembur, peningkatan fasilitas kesejahteraan karyawan, penguatan hubungan kerja dengan manajer, serta pengembangan jalur karier yang lebih jelas bagi karyawan pada level jabatan rendah. Dengan berbasis data, perusahaan dapat membuat keputusan yang lebih cermat untuk menurunkan angka attrition dan mempertahankan talenta terbaik dalam jangka panjang.

### Rekomendasi Action Items (Optional)

Berikut adalah beberapa rekomendasi action items yang dapat diambil oleh perusahaan Jaya Jaya Maju guna menurunkan tingkat attrition dan mencapai target retensi karyawan:
1. Tingkatkan Work-Life Balance untuk Karyawan Lembur
2. Program Peningkatan Kepuasan Kerja
3. Fasilitas Penunjang untuk Karyawan yang Tinggal Jauh
4. Program Retensi untuk Usia Muda dan Entry-Level
5. Kebijakan Promosi dan Kenaikan Gaji yang Transparan
6. Bangun Hubungan Kerja yang Lebih Baik dengan Atasan Langsung
7. Pantau dan Evaluasi Secara Berkala dengan Dashboard
