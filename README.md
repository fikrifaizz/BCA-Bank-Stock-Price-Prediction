# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Maju adalah sebuah perusahaan multinasional yang telah beroperasi sejak tahun 2000 dan memiliki lebih dari 1.000 karyawan yang tersebar di berbagai wilayah di Indonesia. Dengan skala operasional yang besar dan cakupan bisnis yang luas, perusahaan ini telah menunjukkan daya tahan dan pertumbuhan yang signifikan di industrinya.

Meskipun telah tumbuh menjadi perusahaan besar, Jaya Jaya Maju mengalami tantangan serius dalam mengelola karyawan, yang tercermin dari tingginya angka attrition rate (tingkat pengunduran diri atau keluar). Data terakhir menunjukkan attrition rate telah melebihi 10%, angka yang cukup mengkhawatirkan bagi perusahaan sebesar ini.

### Permasalahan Bisnis

1. Faktor apa saja yang paling memengaruhi attrition karyawan?
2. Apa rekomendasi strategis yang dapat diambil HR untuk menekan tingkat keluar masuk karyawan?

### Cakupan Proyek

Proyek ini bertujuan untuk mengidentifikasi faktor-faktor utama yang memengaruhi tingginya tingkat attrition karyawan di perusahaan menggunakan pendekatan data science. Proyek ini akan memberikan gambaran visual, analisis statistik, dan rekomendasi berbasis data kepada pihak manajemen HR.
Ruang Lingkup Pekerjaan:
1. Persiapan dan Pemuatan Data 
   - Mengakses data karyawan dari database PostgreSQL. 
   - Memuat data ke dalam environment Python menggunakan SQLAlchemy.
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

Sumber data: [Dicoding Academy](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee)

Setup environment:

1. Install Docker
2. Jalankan perintah berikut pada Terminal/Command Prompt/PowerShell guna memanggil (pull) Docker image untuk menjalankan Metabase. 

    ```docker pull metabase/metabase:v0.46.4```
3. Apabila proses pembuatan docker image telah selesai, Anda dapat menjalankan image tersebut menggunakan perintah berikut.

    ```docker run -p 3000:3000 --name metabase metabase/metabase```
4. Jalankan Metabase di lokal server dengan tautan berikut.

    ```http://localhost:3000/setup```
5. Setelah melakukan setup metabase, langkah selanjutnya adalah menambahkan data dan pada kasus ini menggunakan PostgreSQL.
6. Selesai

## Business Dashboard

<img src="https://github.com/user-attachments/assets/90fc9f0a-df00-4d0d-8beb-f350ccd32a76" alt="Dashboard" title="Dashboard">

Dashboard ini dirancang untuk menyajikan insight visual yang ringkas dan informatif terkait tingkat attrition (pengunduran diri karyawan) di perusahaan Jaya Jaya Maju. Fungsinya adalah sebagai alat bantu pengambilan keputusan strategis bagi tim Human Resources (HR), khususnya dalam merumuskan strategi retensi karyawan.

Tujuan Dashboard
- Mendeteksi pola dan tren attrition berdasarkan demografi, jabatan, dan kepuasan kerja.
- Mengidentifikasi area-area kritis seperti departemen atau kelompok umur yang memiliki risiko tinggi kehilangan karyawan.
- Menyediakan visualisasi yang mudah dimengerti oleh pihak manajerial tanpa perlu interpretasi teknis mendalam.

Komponen Kunci Dashboard
1. Attrition by Department

   Salah satu departemen (kemungkinan Sales atau Human Resources) menunjukkan tingkat attrition tertinggi dibanding departemen lain seperti Research & Development.
2. Attrition by Job Role

   Beberapa posisi (mungkin Sales Executive atau Laboratory Technician) memiliki tingkat pengunduran diri yang sangat tinggi dibanding peran lainnya.
3. Attrition by Age Group

   Kelompok usia muda (misalnya usia 18–25 atau 26–35) memiliki tingkat attrition lebih tinggi dibanding kelompok usia yang lebih tua.
4. Attrition by Job Satisfaction

   Terdapat korelasi kuat: semakin rendah job satisfaction, semakin tinggi kemungkinan karyawan keluar.

## Conclusion

Berdasarkan hasil eksplorasi data dan dashboard yang telah dibuat, ditemukan beberapa faktor utama yang secara konsisten berkorelasi dengan tingginya tingkat attrition:
- Departemen Tertentu: Misalnya, departemen seperti Sales atau Human Resources menunjukkan tingkat keluar masuk karyawan yang lebih tinggi dibandingkan departemen lainnya seperti Research & Development. Hal ini mengindikasikan adanya tekanan kerja, kurangnya dukungan, atau budaya kerja yang tidak optimal.
- Posisi atau Jabatan Spesifik: Jabatan tertentu seperti Sales Executive atau Laboratory Technician menunjukkan tingkat attrition tinggi. Hal ini dapat mencerminkan ekspektasi kerja yang tidak terpenuhi atau ketidakjelasan jenjang karier.
- Kelompok Usia Muda (18–35 tahun): Karyawan berusia muda cenderung lebih mudah meninggalkan perusahaan, kemungkinan karena mereka masih dalam tahap eksplorasi karier atau merasa kurang mendapat ruang pengembangan.
- Tingkat Kepuasan Kerja Rendah: Karyawan dengan skor kepuasan kerja yang rendah memiliki kecenderungan tinggi untuk keluar. Ini adalah indikator penting yang tidak boleh diabaikan.

### Rekomendasi Action Items (Optional)

Berdasarkan insight di atas, berikut adalah beberapa strategi konkret yang dapat diambil oleh departemen HR:

1. Fokuskan Intervensi pada Departemen Rawan
   - Lakukan audit internal terhadap beban kerja, gaya kepemimpinan, dan budaya tim di departemen yang memiliki attrition tinggi.
   - Terapkan program HR khusus seperti sesi coaching, penguatan teamwork, dan rotasi kerja jika diperlukan.
2. Evaluasi dan Perbaiki Desain Jabatan
   - Untuk posisi dengan attrition tinggi, lakukan review deskripsi pekerjaan, sistem insentif, dan jalur pengembangan karier.
   - Pertimbangkan redesign tanggung jawab agar lebih sesuai dengan harapan dan kapasitas karyawan.
3. Bangun Retensi untuk Karyawan Muda
   - Buat program onboarding yang lebih engaging, career mentoring, dan peluang inovasi untuk generasi muda.
   - Sediakan fleksibilitas kerja (work from anywhere, jam kerja fleksibel) agar sesuai dengan preferensi generasi milenial/Gen Z.
4. Tingkatkan Kepuasan Kerja Secara Sistemik
   - Gunakan hasil survei internal secara berkala untuk mengukur kepuasan dan menyusun kebijakan berbasis feedback karyawan.
   - Perkuat pengakuan terhadap kontribusi karyawan, baik secara finansial (bonus) maupun non-finansial (penghargaan, jenjang karier).
