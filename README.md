#  Movie Recommendation System

## Project Overview

Seiring meningkatnya jumlah pengguna internet dan volume informasi yang tersedia secara daring sejak akhir abad ke-20, pengguna kerap mengalami kesulitan dalam menemukan informasi yang benar-benar relevan. Dalam konteks hiburan digital seperti film, hal ini memunculkan kebutuhan akan sistem rekomendasi yang mampu membantu pengguna menemukan film sesuai preferensi mereka tanpa harus melakukan pencarian berulang. Rekomendasi film menjadi solusi yang populer untuk mempermudah proses pencarian dan pengambilan keputusan pengguna terhadap konten hiburan yang ingin mereka konsumsi Choi et al. (2012).

Menurut Choi et al. (2012), sistem rekomendasi sangat penting untuk mengurangi usaha pencarian informasi yang berulang dengan cara menyarankan konten yang relevan berdasarkan pola perilaku atau preferensi pengguna sebelumnya. Sistem ini secara signifikan dapat meningkatkan efisiensi dan pengalaman pengguna dalam menjelajahi konten digital, seperti film.

Lebih lanjut, Goyani & Chaurasiya (2020) menjelaskan bahwa sistem rekomendasi film dapat dikembangkan melalui dua pendekatan utama, yaitu Collaborative Filtering, yang merekomendasikan item berdasarkan kemiripan antar pengguna, dan Content-Based Filtering, yang memanfaatkan preferensi eksplisit dari pengguna untuk menyarankan konten serupa. Keduanya memiliki kelebihan masing-masing, dan penerapan gabungan dari kedua pendekatan ini dapat meningkatkan akurasi dan personalisasi rekomendasi.

Hal-hal tersebut mendorong saya untuk mengangkat topik ini dalam proyek yang akan saya kerjakan. Saya tertarik untuk membangun sistem rekomendasi film yang mengombinasikan metode collaborative dan content-based filtering, dengan tujuan untuk meningkatkan relevansi rekomendasi dan menciptakan pengalaman pengguna yang lebih personal. Selain itu, dunia hiburan adalah bidang yang sangat dinamis dan dekat dengan kehidupan sehari-hari, sehingga sistem seperti ini akan memiliki nilai guna praktis yang tinggi serta tantangan teknis yang menarik untuk dipecahkan.

Referensi:

- Choi, S. M., Ko, S. K., & Han, Y. S. (2012). A movie recommendation algorithm based on genre correlations. *Expert Systems with Applications, 39(9)*, 8079-8085.  
- Goyani, M., & Chaurasiya, N. (2020). A review of movie recommendation system: Limitations, Survey and Challenges. *ELCVIA. Electronic Letters on Computer Vision and Image Analysis, 19(3)*, 0018-37.



## Business Understanding

### Problem Statements

- Bagaimana cara membangun sistem rekomendasi yang dapat memberikan saran film atau acara TV yang relevan untuk pengguna berdasarkan konten dari film atau acara yang pernah mereka sukai sebelumnya?
- Dengan memanfaatkan informasi dari pengguna lain yang memiliki ketertarikan serupa, bagaimana sistem dapat merekomendasikan film atau acara TV yang mungkin disukai pengguna namun belum pernah ditonton sebelumnya?

### Goals

- Menghasilkan rekomendasi film atau acara TV yang dipersonalisasi menggunakan teknik *Content-Based Filtering*.
- Menghasilkan rekomendasi film atau acara TV berdasarkan preferensi pengguna lain yang serupa menggunakan teknik *Collaborative Filtering*.

###  Solution Approach

- **Content-Based Filtering**
  - Menggunakan fitur konten seperti genres, deskripsi, actors, dan directors.
  - Menggunakan TF-IDF + Cosine Similarity untuk menemukan kemiripan antar konten.

- **Collaborative Filtering**
  - Menggunakan pendekatan Neural Collaborative Filtering berbasis interaksi dummy.
  - Memanfaatkan embedding untuk memetakan pengguna dan item ke dalam vektor berdimensi rendah.

## Data Understanding

Dataset diambil dari Kaggle: [`netflix-tv-shows-and-movies`.](https://www.kaggle.com/datasets/victorsoeiro/netflix-tv-shows-and-movies?select=titles.csv)

Dataset ini dibuat untuk mencantumkan semua tayangan yang tersedia di layanan streaming Netflix, serta menganalisis data tersebut untuk menemukan fakta-fakta menarik. Data ini dikumpulkan pada bulan Juli 2022 dan mencakup tayangan yang tersedia di wilayah Amerika Serikat.

### Jumlah & Fitur

#### Deskripsi Dataset

##### 1. `credits.csv` (5 kolom)
Dataset ini berisi lebih dari 50.000 data kredit dari aktor dan sutradara yang terlibat dalam judul-judul Netflix. Terdiri dari 5 kolom informasi sebagai berikut:

- `person_id`: ID individu berdasarkan JustWatch.
- `id`: ID judul berdasarkan JustWatch.
- `name`: Nama aktor atau sutradara.
- `character_name`: Nama karakter (jika tersedia).
- `role` Peran sebagai ACTOR atau DIRECTOR.

##### 2. `titles.csv` (15 kolom)
Dataset ini berisi lebih dari 5.000 judul unik yang tersedia di Netflix, dengan 15 kolom informasi sebagai berikut:

- `id`: ID judul berdasarkan JustWatch.
- `title`: Nama judul film atau acara.
- `show_type`: Tipe tayangan, bisa berupa MOVIE atau SHOW.
- `description`: Deskripsi singkat tentang judul.
- `release_year`: Tahun rilis.
- `age_certification`: Sertifikasi usia (rating umur).
- `runtime`: Durasi film atau episode (jika berupa SHOW).
- `genres`: Daftar genre dari judul tersebut.
- `production_countries`: Daftar negara yang memproduksi judul.
- `seasons`: Jumlah musim (jika merupakan SHOW).
- `imdb_id`: ID judul pada situs IMDb.
- `imdb_score`: Skor rating dari IMDb.
- `imdb_votes`: Jumlah voting pada IMDb.
- `tmdb_popularity`: Skor popularitas dari TMDB.
- `tmdb_score`: Skor rating dari TMDB.


<p align='center'>
      <img src ="https://github.com/faRIZky/sistem-rekomendasi_achmad-fariz/blob/main/images/tipe%20konten.png?raw=true" alt="konten"> 
</p>

<p align='center'>
      <img src ="https://github.com/faRIZky/sistem-rekomendasi_achmad-fariz/blob/main/images/genre%20terbanyak.png?raw=true" alt="genre"> 
</p>

<p align='center'>
      <img src ="https://github.com/faRIZky/sistem-rekomendasi_achmad-fariz/blob/main/images/negara%20produksi%20terbanyak.png?raw=true" alt="negara"> 
</p>

**Insight:**
- Mayoritas konten berupa film.
- Genre dominan: Drama, Comedy, Thriller.
- Negara produksi terbanyak: US, India, UK.

<p align='center'>
      <img src ="https://github.com/faRIZky/sistem-rekomendasi_achmad-fariz/blob/main/images/imdb%20score.png?raw=true" alt="skor"> 
</p>

**Distribusi Skor:**
IMDb & TMDb rata-rata sekitar 6.5–6.8.

```
top_actors = credits_df[credits_df['role'] == 'ACTOR']['name'].value_counts().head(10)
top_directors = credits_df[credits_df['role'] == 'DIRECTOR']['name'].value_counts().head(10)
```

**Top Aktor & Sutradara:**
- Aktor: Boman Irani, Kareena Kapoor Khan
- Sutradara: Raúl Campos, Jan Suter

## Data Preparation
## Data Preparation (Content-based Filtering)

### Duplicate Handling
```
# Cek dan hapus duplikat di titles_df
print("Duplikat di titles_df:", titles_df.duplicated().sum())
titles_df.drop_duplicates(inplace=True)

# Cek dan hapus duplikat di credits_df
print("Duplikat di credits_df:", credits_df.duplicated().sum())
credits_df.drop_duplicates(inplace=True)
```
- Insight: Tidak ditemukan duplikat.
- Alasan langkah: Mengecek duplikat penting untuk memastikan tidak ada entri ganda yang bisa mengganggu proses analisis dan model training. Meski tidak ada, ini langkah preventif yang wajib dalam EDA.

### Missing Value Handling

```
titles_df['age_certification'].fillna("Unknown", inplace=True)
titles_df['imdb_score'].fillna(titles_df['imdb_score'].mean(), inplace=True)
titles_df['tmdb_score'].fillna(titles_df['tmdb_score'].mean(), inplace=True)
titles_df['seasons'].fillna(0, inplace=True)
titles_df['genres'].fillna("[]", inplace=True)
titles_df['description'].fillna("", inplace=True)

credits_df['character'].fillna("Unknown", inplace=True)
```

- Insight: Banyak nilai hilang terutama di kolom age_certification, imdb_score, tmdb_score, dan description.

- Alasan langkah:

Unknown untuk kolom kategorikal menjaga struktur data tanpa membuang baris.

Rata-rata digunakan untuk mengisi skor agar distribusi data tidak berubah drastis.

description, genres, dan seasons diisi default agar tidak error saat diproses string.

```
# Gabung aktor
actors = credits_df[credits_df['role'] == 'ACTOR'].groupby('id')['name'].apply(lambda x: ' '.join(x)).reset_index()
actors.columns = ['id', 'actors']

# Gabung sutradara
directors = credits_df[credits_df['role'] == 'DIRECTOR'].groupby('id')['name'].apply(lambda x: ' '.join(x)).reset_index()
directors.columns = ['id', 'directors']

# Merge ke titles_df
titles_df = titles_df.merge(actors, on='id', how='left')
titles_df = titles_df.merge(directors, on='id', how='left')

# Isi NaN hasil gabungan
titles_df['actors'].fillna("", inplace=True)
titles_df['directors'].fillna("", inplace=True)

def combine_features(row):
    genres = " ".join(ast.literal_eval(row['genres'])) if row['genres'] != "[]" else ""
    return f"{row['title']} {row['description']} {genres} {row['age_certification']} {row['actors']} {row['directors']}"

titles_df['content'] = titles_df.apply(combine_features, axis=1)
```

- Aktor & sutradara digabung berdasarkan `id` dan ditambahkan sebagai fitur teks.
- Kolom baru `content` digabung dari title, deskripsi, genre, aktor, sutradara.

```
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(titles_df['content'])

# Cosine Similarity antar judul
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index mapping berdasarkan title
indices = pd.Series(titles_df.index, index=titles_df['title']).drop_duplicates()
```
Langkah ini bertujuan untuk membangun sistem Content-Based Filtering. Dengan menggunakan TfidfVectorizer, kita mengubah teks pada kolom content menjadi representasi numerik berbasis frekuensi kata, lalu menghitung kemiripan antar film menggunakan cosine similarity. Hasilnya adalah matriks yang menunjukkan seberapa mirip satu film dengan yang lain berdasarkan kontennya, sehingga sistem bisa merekomendasikan film serupa.

### Data Preparation (Collaborative Filtering)
- Encode user, item dan berikan dummy rating
Pada bagian ini, dilakukan proses encoding data supaya bisa digunakan dalam model Collaborative Filtering.
Karena model butuh input numerik, maka:

  - Nama aktor/sutradara (name) diubah menjadi user_id.

  - ID film (id) diubah menjadi item_id.

  - Ditambahkan kolom rating dummy, karena model matrix factorization memerlukan data interaksi user-item.

Selanjutnya, karna kita akan menggunakan 2 skema pelatihan dengan split data yang berbeda, kita akan membuat 2 variable splitting data.

```
# Data Splitting untuk Skema 1: 80:20
x_train_1, x_val_1, y_train_1, y_val_1 = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Splitting untuk Skema 2: 70:30
x_train_2, x_val_2, y_train_2, y_val_2 = train_test_split(X, y, test_size=0.3, random_state=42)
```
Sekarang data dengan 2 skala pembagian yang berbeda telah siap untuk modeling collaborative filtering.
## Modeling and Result

### Content-Based Filtering
Content-Based Filtering adalah pendekatan dalam sistem rekomendasi yang menyarankan item, seperti film, berdasarkan kemiripan konten — dalam kasus ini, genre film. Sistem ini menggunakan teknik TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengekstraksi fitur penting dari kolom genre dan mengubahnya menjadi representasi vektor numerik. Setelah itu, sistem menghitung cosine similarity antara film yang sudah ditonton pengguna dengan film lainnya untuk menemukan rekomendasi yang paling mirip secara konten.

Prediksi hasil rekomendasi kemudian dievaluasi menggunakan metrik precision.
Metode ini sangat efektif terutama ketika tidak tersedia data interaksi antar pengguna (cold start problem), serta cocok untuk pengguna baru karena hanya bergantung pada preferensi konten mereka sendiri, tanpa membutuhkan data pengguna lain.

Algoritma:
Sistem rekomendasi ini menggunakan pendekatan Content-Based Filtering dengan menghitung kesamaan antar film berdasarkan representasi kontennya menggunakan cosine similarity. Film yang paling mirip dengan film input akan direkomendasikan.

Fungsi:

- `recommend_content_based(title, n=5)`
Mengembalikan daftar top-N film yang paling mirip dengan film input berdasarkan skor cosine similarity. Jika judul tidak ditemukan dalam dataset, fungsi akan mengembalikan pesan error.

- `get_genres(title)`
Mengembalikan daftar genre dari film tertentu dalam bentuk list. Fungsi ini membaca data genre yang disimpan dalam format teks list.

- `precision_at_k_detail_df(title, k=5)`
Mengembalikan dataframe berisi daftar rekomendasi film, genre mereka, dan apakah mereka relevan (minimal ada satu genre yang cocok dengan film target). Selain itu, fungsi ini juga menghitung nilai Precision@k, yaitu persentase film relevan dari seluruh rekomendasi yang diberikan.
  
---

Dilakukan evaluasi menggunakan fungsi `precision_at_k_detail_df` untuk film **Breaking Bad** dengan k=5. Berikut hasil rekomendasi dan evaluasinya:

| Recommended Title | Genres | Relevant (Genre Match) |
|:------------------|:-------|:-----------------------|
| El Camino: A Breaking Bad Movie | action, crime, drama, thriller | ✅ |
| Linewatch | crime, thriller, drama | ✅ |
| W/ Bob & David | comedy | ❌ |
| Better Call Saul | crime, drama | ✅ |
| The Lovebirds | action, thriller, comedy, romance, crime | ✅ |

**Hasil Evaluasi:**  
- Nilai **Precision@5** untuk *Breaking Bad* adalah **0.80** atau **80%**.
- Ini berarti 4 dari 5 film yang direkomendasikan memiliki kesamaan genre dengan film *Breaking Bad*.
- Model menunjukkan performa cukup baik, walaupun masih ada 1 film (*W/ Bob & David*) yang kurang relevan secara genre.

---
- Kelebihan dan Kekurangan Content-Based Filtering (TF-IDF + Cosine Similarity):

- Kelebihan:
  - Sederhana dan cepat: Menggunakan TF-IDF dan Cosine Similarity membuat perhitungannya efisien, bahkan untuk dataset menengah.

  - Interpretable: Sistem rekomendasi mudah dijelaskan karena hanya mencari kemiripan berdasarkan fitur konten (genre, deskripsi, dsb).

  - Tidak butuh data user lain: Cocok kalau user masih sedikit atau belum ada interaksi.

- Kekurangan:
  - Over-specialization: Rekomendasi cenderung terbatas pada film yang mirip, kurang eksploratif (kurang diverse).

  - Tidak bisa tangani cold start item: Film baru tanpa deskripsi yang cukup sulit untuk direkomendasikan secara akurat.

  - Tidak mempertimbangkan preferensi komunitas: Tidak tahu kalau film yang kurang mirip secara konten, tapi banyak disukai orang dengan selera mirip user.



### Collaborative Filtering

Collaborative Filtering dalam proyek ini menggunakan pendekatan Neural Collaborative Filtering (NCF), yaitu memanfaatkan model neural network untuk mempelajari pola interaksi antara pengguna (aktor/sutradara) dan film yang pernah mereka bintangi atau sutradarai. Karena tidak tersedia rating eksplisit, sistem membuat dummy rating (nilai 1) untuk mensimulasikan interaksi positif.
Model dilatih untuk memprediksi kemungkinan ketertarikan pengguna terhadap film lain dan merekomendasikan berdasarkan skor tertinggi.
Pendekatan ini menghasilkan rekomendasi yang lebih personal, dengan memperhitungkan pola perilaku tersembunyi antar pengguna yang tidak dapat ditangkap hanya dari konten film.

- Algoritma:
Collaborative Filtering berbasis model Neural Network.
Pendekatannya menggunakan embedding vector untuk memetakan user dan item ke dalam ruang dimensi yang lebih kecil, lalu menghitung prediksi rating berdasarkan dot product antara vektor user dan item. Model ini mempelajari pola preferensi pengguna dari data interaksi user-item (seperti ratings atau credits).

Model yang digunakan adalah RecommenderNet, dengan struktur sebagai berikut:

- Embedding untuk user dan item.

- Bias untuk user dan item.

- Dot product antara user embedding dan item embedding ditambah bias, lalu dilalui fungsi aktivasi sigmoid untuk prediksi rating antara 0 dan 1.

Fungsi:
- `RecommenderNet`

  - Fungsi ini membangun model neural network sederhana untuk collaborative filtering.

  - Mengembalikan output prediksi rating untuk pasangan (user_id, item_id) berdasarkan hasil pembelajaran dari data training.

- `recommend_collaborative(user_name, model, n=10)`

  - Fungsi ini digunakan untuk menghasilkan rekomendasi film untuk user tertentu.

  - Langkah-langkahnya:

    - Encode nama user menjadi user ID.

    - Identifikasi item yang belum ditonton oleh user.

    - Prediksi rating untuk semua item tersebut menggunakan model.

    - Pilih top-N item dengan skor prediksi tertinggi.

    - Mengembalikan DataFrame berisi judul film, tipe, dan tahun rilis.


Dalam proyek ini, dilakukan dua skema pelatihan model RecommenderNet untuk eksperimen:

- Skema 1 menggunakan ukuran embedding sebesar 50, dengan learning rate 0.001, dan data dibagi dengan rasio 80:20 (80% untuk training, 20% untuk validation).

- Skema 2 menggunakan ukuran embedding yang lebih besar, yaitu 100, dengan learning rate yang lebih kecil 0.0005, serta pembagian data 70:30 (70% untuk training, 30% untuk validation).

Perbedaan embedding size dan learning rate ini bertujuan untuk mengamati pengaruh kapasitas representasi dan kecepatan pembelajaran model terhadap performa rekomendasi. Sementara, variasi data split digunakan untuk mengevaluasi stabilitas generalisasi model dengan proporsi data training yang berbeda.

| Skema | Embedding Size | Learning Rate | Data Split 
|------|----------------|---------------|------------|
| Skema 1 | 50 | 0.001 | 80:20 |
| Skema 2 | 100 | 0.0005 | 70:30 | 

Berikut adalah hasil training dari model collaborative filtering Skema 1 dan Skema 2 menggunakan metrik **Root Mean Squared Error (RMSE)**.
Nilai RMSE yang lebih rendah menunjukkan model semakin baik dalam memprediksi interaksi user-item.

### Hasil Training Skema 1

| Epoch | Loss  | Train RMSE | Val Loss | Val RMSE |
|------|-------|------------|----------|----------|
| 1    | 0.6715 | 0.4889     | 0.6000   | 0.4507   |
| 2    | 0.5510 | 0.4227     | 0.5104   | 0.3971   |
| 3    | 0.3841 | 0.3179     | 0.4444   | 0.3510   |
| 4    | 0.2391 | 0.2112     | 0.4050   | 0.3175   |
| 5    | 0.1611 | 0.1395     | 0.3781   | 0.2920   |
| 6    | 0.1275 | 0.0985     | 0.3508   | 0.2684   |
| 7    | 0.1135 | 0.0779     | 0.3185   | 0.2444   |
| 8    | 0.1050 | 0.0663     | 0.2848   | 0.2214   |
| 9    | 0.0981 | 0.0594     | 0.2528   | 0.2005   |
| 10   | 0.0918 | 0.0555     | 0.2239   | 0.1820   |


### Hasil Training Skema 2

| Epoch | Loss  | Train RMSE | Val Loss | Val RMSE |
|------|-------|------------|----------|----------|
| 1    | 0.6838 | 0.4951     | 0.6516   | 0.4785   |
| 2    | 0.6230 | 0.4631     | 0.6082   | 0.4539   |
| 3    | 0.5142 | 0.3999     | 0.5691   | 0.4286   |
| 4    | 0.3814 | 0.3139     | 0.5384   | 0.4060   |
| 5    | 0.2712 | 0.2330     | 0.5161   | 0.3870   |
| 6    | 0.1971 | 0.1692     | 0.4992   | 0.3708   |
| 7    | 0.1547 | 0.1254     | 0.4840   | 0.3563   |
| 8    | 0.1325 | 0.0966     | 0.4674   | 0.3419   |
| 9    | 0.1205 | 0.0781     | 0.4482   | 0.3274   |
| 10   | 0.1141 | 0.0670     | 0.4271   | 0.3127   |



Berikut adalah hasil rekomendasi collaborative filtering untuk setiap skema dengan input user "Robert De Niro":
- `recommend_collaborative("Robert De Niro", model=model_1, n=5)`

| No | Title                   | Type  | Release Year |
|----|--------------------------|-------|--------------|
| 1  | Titanic                  | MOVIE | 1997         |
| 2  | Donnie Brasco            | MOVIE | 1997         |
| 3  | Catch Me If You Can      | MOVIE | 2002         |
| 4  | Les Misérables           | MOVIE | 2012         |
| 5  | tick, tick... BOOM!      | MOVIE | 2021         |

Model Skema 1 (embedding size 50, learning rate 0.001) digunakan untuk menghasilkan rekomendasi di atas.
Film-film yang direkomendasikan banyak berasal dari genre drama dan kriminal, serta didominasi oleh film populer.
Ini menunjukkan model mampu menemukan hubungan user-item berdasarkan pola keterkaitan umum di data.

- `recommend_collaborative("Robert De Niro", model=model_1, n=5)`

| No | Title                                                   | Type  | Release Year |
|----|---------------------------------------------------------|-------|--------------|
| 1  | The Departed                                             | MOVIE | 2006         |
| 2  | The Blind Side                                           | MOVIE | 2009         |
| 3  | Starsky & Hutch                                          | MOVIE | 2004         |
| 4  | Argo                                                     | MOVIE | 2012         |
| 5  | Rolling Thunder Revue: A Bob Dylan Story by Martin Scorsese | MOVIE | 2019         |

Model Skema 2 (embedding size 100, learning rate 0.0005) digunakan untuk menghasilkan daftar rekomendasi ini.
Film yang direkomendasikan tetap memiliki keterkaitan dengan genre drama, crime, dan thriller.
Pilihan film juga mencakup beberapa karya yang lebih spesifik, seperti dokumenter musik, tanpa mengambil kesimpulan performa model pada tahap ini.

## Evaluation
## Evaluation: Content-Based Filtering

### 1. Metrik Evaluasi yang Digunakan

- **Precision@K**:  
  Precision@K digunakan untuk mengukur seberapa relevan item-item yang direkomendasikan terhadap preferensi pengguna.  
  Formula:  
>  Precision@K = Jumlah item relevan yang direkomendasi/ K
  
- **Definisi Relevansi**:  
  Dalam konteks ini, item dikatakan *relevan* jika memiliki **genre** yang sama dengan genre film yang menjadi dasar rekomendasi.

---

### 2. Hasil Evaluasi

Evaluasi dilakukan pada tiga contoh judul: **Narcos**, **Breaking Bad**, dan **Peaky Blinders**, masing-masing dengan **K=5** (5 rekomendasi teratas).

| Judul              | Precision@5 | Interpretasi |
|--------------------|-------------|--------------|
| Narcos             | 1.00 (100%) | Semua film rekomendasi memiliki genre yang relevan. |
| Breaking Bad       | 0.80 (80%)  | 4 dari 5 film rekomendasi memiliki genre yang relevan. |
| Peaky Blinders     | 0.60 (60%)  | 3 dari 5 film rekomendasi memiliki genre yang relevan. |


### 3. Interpretasi Hasil

- **Precision tinggi** (seperti pada *Narcos*) menunjukkan bahwa sistem berhasil merekomendasikan film yang sesuai dengan preferensi konten pengguna.
- **Precision lebih rendah** (seperti *Peaky Blinders*) menunjukkan ada potensi perbaikan, misalnya dengan mempertimbangkan fitur tambahan seperti deskripsi, aktor, atau director.
- **Secara keseluruhan**, sistem Content-Based Filtering berhasil menunjukkan relevansi yang baik, mendukung tujuan **personalized recommendation** berbasis konten.

---
## Evaluation: Collaborative Filtering
### 1. Metrik Evaluasi yang Digunakan

- **Root Mean Squared Error (RMSE)**:  
  RMSE digunakan untuk mengukur seberapa besar rata-rata error (kesalahan prediksi) antara rating sebenarnya dan rating yang diprediksi oleh model.  
  Formula:  
>  RMSE = sqrt( (1/n) * Σ (y_true - y_pred)² )

  
- **Interpretasi RMSE**:
  - Semakin kecil nilai RMSE, semakin baik model dalam melakukan prediksi.
  - RMSE mendeteksi error besar dengan penalti lebih berat dibandingkan MAE, cocok untuk evaluasi sistem rekomendasi.

---

### 2. Hasil Evaluasi

Evaluasi dilakukan pada 2 skema berbeda:

| Skema | Embedding Size | Learning Rate | Data Split | Validation RMSE Akhir | Interpretasi |
|------|----------------|---------------|------------|-----------------------|--------------|
| Skema 1 | 50 | 0.001 | 80:20 | **0.1820** | Performa sangat baik, error kecil. |
| Skema 2 | 100 | 0.0005 | 70:30 | 0.3127 | Error lebih besar dibanding Skema 1. |

---

### 3. Interpretasi Hasil

- **Skema 1** menghasilkan RMSE validasi lebih rendah (**0.1820**) dibandingkan Skema 2 (**0.3127**).
- Ini menunjukkan bahwa:
  - **Skema 1 lebih baik** dalam mempelajari hubungan user-item dibandingkan Skema 2.
  - Meskipun embedding size Skema 2 lebih besar (100 vs 50), learning rate yang lebih kecil (0.0005) dan data split (70:30) mungkin membuat model lebih lambat belajar/mengalami underfitting.
- **Secara keseluruhan**, **Skema 1** dipilih sebagai model terbaik untuk rekomendasi berbasis Collaborative Filtering pada proyek ini.

---
### Hubungan Model Terhadap Business Understanding

### 1. Apakah sudah menjawab setiap problem statement?

- **Problem Statement 1:**  
  > "Bagaimana cara membangun sistem rekomendasi yang dapat memberikan saran film atau acara TV yang relevan untuk pengguna berdasarkan konten dari film atau acara yang pernah mereka sukai sebelumnya?"  
  **Sudah dijawab** melalui implementasi **Content-Based Filtering** yang memanfaatkan genre, deskripsi, dan skor film untuk mengukur kemiripan antar film dan memberikan rekomendasi berdasarkan konten yang mirip.

- **Problem Statement 2:**  
  > "Dengan memanfaatkan informasi dari pengguna lain yang memiliki ketertarikan serupa, bagaimana sistem dapat merekomendasikan film atau acara TV yang mungkin disukai pengguna namun belum pernah ditonton sebelumnya?"
   **Sudah dijawab** melalui implementasi **Collaborative Filtering** berbasis model (menggunakan arsitektur RecommenderNet) yang mempelajari hubungan antar pengguna dan item tanpa memerlukan explicit rating.

---

### 2. Apakah berhasil mencapai setiap goals yang diharapkan?

- **Goal 1:**  
  >*Menghasilkan rekomendasi film atau acara TV yang dipersonalisasi menggunakan teknik Content-Based Filtering. **Tercapai** , sistem mampu merekomendasikan film yang relevan dengan film yang sebelumnya disukai pengguna berdasarkan fitur konten.

- **Goal 2:**  
  > Menghasilkan rekomendasi film atau acara TV menggunakan teknik Collaborative Filtering berdasarkan kemiripan pengguna.  
  **Tercapai**, model RecommenderNet berhasil melatih embedding pengguna dan item, dan menghasilkan rekomendasi personalized untuk user yang diuji.

---

### 3. Apakah setiap solusi statement yang direncanakan berdampak?

- **Content-Based Filtering:**
  - Pendekatan dengan menggunakan TF-IDF dan cosine similarity terhadap genres, description, dan tmdb_score terbukti mampu menghasilkan rekomendasi relevan dengan precision yang cukup tinggi (berdasarkan hasil evaluasi Precision@K).
  - Dampaknya adalah sistem mampu tetap merekomendasikan film meskipun pengguna baru (*cold start*) karena berbasis konten yang melekat pada film.

- **Collaborative Filtering:**
  - Pendekatan dengan RecommenderNet dan embedding matrix berhasil menemukan pola latent (tersembunyi) antar pengguna dan item, yang tercermin dari nilai RMSE validasi yang semakin menurun dari epoch ke epoch, terutama pada **Skema 1** yang lebih optimal.
  - Dampaknya, sistem dapat memberikan rekomendasi yang personalized berdasarkan perilaku pengguna lain, memperkaya rekomendasi yang tidak hanya berdasarkan konten.

---

### 4. Kesimpulan

Secara keseluruhan, **model dan pendekatan yang digunakan telah berhasil menjawab problem statement, mencapai goals yang diharapkan, serta memberikan dampak positif terhadap sistem rekomendasi**.  
Sistem ini tidak hanya mampu memberikan rekomendasi berdasarkan konten yang mirip (content-based), tetapi juga mampu memperkaya rekomendasi dengan pola preferensi antar pengguna (collaborative filtering).

Kedua model ini melengkapi satu sama lain dan membuat sistem rekomendasi menjadi lebih robust dan efektif dalam memenuhi kebutuhan bisnis.

