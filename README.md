# 🎬 Movie Recommendation System

## 📌 Project Overview

Seiring meningkatnya jumlah pengguna internet dan volume informasi yang tersedia secara daring sejak akhir abad ke-20, pengguna kerap mengalami kesulitan dalam menemukan informasi yang benar-benar relevan. Dalam konteks hiburan digital seperti film, hal ini memunculkan kebutuhan akan sistem rekomendasi yang mampu membantu pengguna menemukan film sesuai preferensi mereka tanpa harus melakukan pencarian berulang. Rekomendasi film menjadi solusi yang populer untuk mempermudah proses pencarian dan pengambilan keputusan pengguna terhadap konten hiburan yang ingin mereka konsumsi Choi et al. (2012).

Menurut Choi et al. (2012), sistem rekomendasi sangat penting untuk mengurangi usaha pencarian informasi yang berulang dengan cara menyarankan konten yang relevan berdasarkan pola perilaku atau preferensi pengguna sebelumnya. Sistem ini secara signifikan dapat meningkatkan efisiensi dan pengalaman pengguna dalam menjelajahi konten digital, seperti film.

Lebih lanjut, Goyani & Chaurasiya (2020) menjelaskan bahwa sistem rekomendasi film dapat dikembangkan melalui dua pendekatan utama, yaitu Collaborative Filtering, yang merekomendasikan item berdasarkan kemiripan antar pengguna, dan Content-Based Filtering, yang memanfaatkan preferensi eksplisit dari pengguna untuk menyarankan konten serupa. Keduanya memiliki kelebihan masing-masing, dan penerapan gabungan dari kedua pendekatan ini dapat meningkatkan akurasi dan personalisasi rekomendasi.

Hal-hal tersebut mendorong saya untuk mengangkat topik ini dalam proyek yang akan saya kerjakan. Saya tertarik untuk membangun sistem rekomendasi film yang mengombinasikan metode collaborative dan content-based filtering, dengan tujuan untuk meningkatkan relevansi rekomendasi dan menciptakan pengalaman pengguna yang lebih personal. Selain itu, dunia hiburan adalah bidang yang sangat dinamis dan dekat dengan kehidupan sehari-hari, sehingga sistem seperti ini akan memiliki nilai guna praktis yang tinggi serta tantangan teknis yang menarik untuk dipecahkan.

Referensi:

Choi, S. M., Ko, S. K., & Han, Y. S. (2012). A movie recommendation algorithm based on genre correlations. Expert Systems with Applications, 39(9), 8079-8085.
Goyani, M., & Chaurasiya, N. (2020). A review of movie recommendation system: Limitations, Survey and Challenges. ELCVIA. Electronic Letters on Computer Vision and Image Analysis, 19(3), 0018-37.



## 💼 Business Understanding

### 🧩 Problem Statements

- Bagaimana cara membangun sistem rekomendasi yang dapat memberikan saran film atau acara TV yang relevan untuk pengguna berdasarkan konten dari film atau acara yang pernah mereka sukai sebelumnya?
- Dengan memanfaatkan informasi dari pengguna lain yang memiliki ketertarikan serupa, bagaimana sistem dapat merekomendasikan film atau acara TV yang mungkin disukai pengguna namun belum pernah ditonton sebelumnya?

### 🎯 Goals

- Menghasilkan rekomendasi film atau acara TV yang dipersonalisasi menggunakan teknik *Content-Based Filtering*.
- Menghasilkan rekomendasi film atau acara TV berdasarkan preferensi pengguna lain yang serupa menggunakan teknik *Collaborative Filtering*.

### 🛠️ Solution Approach

- **Content-Based Filtering**
  - Menggunakan fitur konten seperti genres, deskripsi, actors, dan directors.
  - Menggunakan TF-IDF + Cosine Similarity untuk menemukan kemiripan antar konten.

- **Collaborative Filtering**
  - Menggunakan pendekatan Neural Collaborative Filtering berbasis interaksi dummy.
  - Memanfaatkan embedding untuk memetakan pengguna dan item ke dalam vektor berdimensi rendah.

## 📊 Data Understanding

Dataset diambil dari Kaggle: [`netflix-tv-shows-and-movies`.](https://www.kaggle.com/datasets/victorsoeiro/netflix-tv-shows-and-movies?select=titles.csv)

### Jumlah & Fitur

- **titles.csv**: 14 fitur, termasuk title, genres, description, imdb_score, tmdb_score.
- **credits.csv**: informasi aktor dan sutradara.

<!-- IMAGE HERE: visualisasi distribusi konten -->

**Insight:**
- Mayoritas konten berupa film.
- Genre dominan: Drama, Comedy, Thriller.
- Negara produksi terbanyak: US, India, UK.

<!-- IMAGE HERE: visualisasi genre & negara -->

**Distribusi Skor:**
IMDb & TMDb rata-rata sekitar 6.5–6.8.

<!-- IMAGE HERE: distribusi skor -->

**Top Aktor & Sutradara:**
- Aktor: Boman Irani, Kareena Kapoor Khan
- Sutradara: Raúl Campos, Jan Suter

## 🧹 Data Preparation

### Duplicate Handling

- Tidak ditemukan duplikat.

### Missing Value Handling

- Kolom kategorikal seperti `age_certification` diisi dengan "Unknown".
- Kolom numerik (`imdb_score`, `tmdb_score`) diisi dengan mean.
- Deskripsi kosong diisi dengan string kosong.

### Feature Engineering

- Aktor & sutradara digabung berdasarkan `id` dan ditambahkan sebagai fitur teks.
- Kolom baru `content` digabung dari title, deskripsi, genre, aktor, sutradara.

## 🤖 Modeling and Result

### Content-Based Filtering

- Algoritma: TF-IDF + Cosine Similarity
- Fungsi `recommend_content_based(title, n=5)` mengembalikan top-N film mirip.

**Kelebihan:**
- Cepat, tidak butuh interaksi pengguna lain.

**Kekurangan:**
- Over-specialization, cold-start problem.

### Collaborative Filtering

- Algoritma: Neural Collaborative Filtering (dengan dummy rating)
- Fungsi `recommend_collaborative(user_name, n=5)` untuk prediksi berdasarkan pengguna.

**Kelebihan:**
- Personalized, menangkap pola kompleks.

**Kekurangan:**
- Cold start & membutuhkan interaksi data yang cukup.

<!-- IMAGE HERE: plot RMSE per epoch -->

## 🧪 Evaluation

### 📈 Content-Based Filtering

- **Metrik:** Cosine Similarity  
- **Formula:**  
  \[
  \text{CosSim} = \frac{A \cdot B}{\|A\| \|B\|}
  \]

- **Interpretasi:** Semakin dekat ke 1, makin mirip.
- **Kaitan bisnis:** Memahami kesukaan user berdasarkan konten, meningkatkan kepuasan dan engagement.

### 📉 Collaborative Filtering

- **Metrik:** Root Mean Squared Error (RMSE)  
- **Formula:**  
  \[
  RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
  \]

- **Interpretasi:** Semakin kecil RMSE, semakin akurat model.
- **Kaitan bisnis:** Model mampu menyarankan film yang belum diketahui user tapi disukai oleh komunitas serupa.

## 🧠 Business Impact Revisited

- **Problem:** Kesulitan menemukan film yang relevan.
- **Goals:** Memberikan rekomendasi personal dan luas.
- **Solution:** Kombinasi CBF & CF menyasar dua tipe user:
  - User baru → CBF.
  - User aktif → CF.
- Evaluasi berdasarkan kemiripan dan prediksi performa rekomendasi yang diharapkan memperkuat engagement pengguna dan eksplorasi konten.

---

> 📌 *Catatan: Tambahkan gambar visualisasi di bagian yang ditandai dengan `<!-- IMAGE HERE -->`.*

