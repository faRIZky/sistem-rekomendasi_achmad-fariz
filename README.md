#  Movie Recommendation System

## Project Overview

Seiring meningkatnya jumlah pengguna internet dan volume informasi yang tersedia secara daring sejak akhir abad ke-20, pengguna kerap mengalami kesulitan dalam menemukan informasi yang benar-benar relevan. Dalam konteks hiburan digital seperti film, hal ini memunculkan kebutuhan akan sistem rekomendasi yang mampu membantu pengguna menemukan film sesuai preferensi mereka tanpa harus melakukan pencarian berulang. Rekomendasi film menjadi solusi yang populer untuk mempermudah proses pencarian dan pengambilan keputusan pengguna terhadap konten hiburan yang ingin mereka konsumsi Choi et al. (2012).

Menurut Choi et al. (2012), sistem rekomendasi sangat penting untuk mengurangi usaha pencarian informasi yang berulang dengan cara menyarankan konten yang relevan berdasarkan pola perilaku atau preferensi pengguna sebelumnya. Sistem ini secara signifikan dapat meningkatkan efisiensi dan pengalaman pengguna dalam menjelajahi konten digital, seperti film.

Lebih lanjut, Goyani & Chaurasiya (2020) menjelaskan bahwa sistem rekomendasi film dapat dikembangkan melalui dua pendekatan utama, yaitu Collaborative Filtering, yang merekomendasikan item berdasarkan kemiripan antar pengguna, dan Content-Based Filtering, yang memanfaatkan preferensi eksplisit dari pengguna untuk menyarankan konten serupa. Keduanya memiliki kelebihan masing-masing, dan penerapan gabungan dari kedua pendekatan ini dapat meningkatkan akurasi dan personalisasi rekomendasi.

Hal-hal tersebut mendorong saya untuk mengangkat topik ini dalam proyek yang akan saya kerjakan. Saya tertarik untuk membangun sistem rekomendasi film yang mengombinasikan metode collaborative dan content-based filtering, dengan tujuan untuk meningkatkan relevansi rekomendasi dan menciptakan pengalaman pengguna yang lebih personal. Selain itu, dunia hiburan adalah bidang yang sangat dinamis dan dekat dengan kehidupan sehari-hari, sehingga sistem seperti ini akan memiliki nilai guna praktis yang tinggi serta tantangan teknis yang menarik untuk dipecahkan.

Referensi:

Choi, S. M., Ko, S. K., & Han, Y. S. (2012). A movie recommendation algorithm based on genre correlations. Expert Systems with Applications, 39(9), 8079-8085.
Goyani, M., & Chaurasiya, N. (2020). A review of movie recommendation system: Limitations, Survey and Challenges. ELCVIA. Electronic Letters on Computer Vision and Image Analysis, 19(3), 0018-37.



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

## 📊 Data Understanding

Dataset diambil dari Kaggle: [`netflix-tv-shows-and-movies`.](https://www.kaggle.com/datasets/victorsoeiro/netflix-tv-shows-and-movies?select=titles.csv)

### Jumlah & Fitur

- **titles.csv**: 14 fitur, termasuk title, genres, description, imdb_score, tmdb_score.
- **credits.csv**: informasi aktor dan sutradara.

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

## 🧹 Data Preparation

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

## 🤖 Modeling and Result

### Content-Based Filtering

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

# Fungsi rekomendasi
def recommend_content_based(title, n=5):
    idx = indices.get(title)
    if idx is None:
        return f"Judul '{title}' tidak ditemukan dalam dataset."
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]  # skip yang 0 (judul itu sendiri)
    movie_indices = [i[0] for i in sim_scores]
    
    return titles_df[['title', 'type', 'release_year']].iloc[movie_indices]

```
- Algoritma: TF-IDF + Cosine Similarity
- Fungsi `recommend_content_based(title, n=5)` mengembalikan top-N film mirip.

```
recommend_content_based("Narcos", n=5)
```

| Title                                | Type | Release Year |
|--------------------------------------|------|---------------|
| El cartel 2 - La guerra total        | SHOW | 2010          |
| Narcos: Mexico                       | SHOW | 2018          |
| Tijuana                              | SHOW | 2019          |
| Apaches                              | SHOW | 2018          |
| El Escamoso                          | SHOW | 2001          |

Sistem merekomendasikan 5 film di atas sebagai Top-5 kepada user.

Content-Based Filtering (TF-IDF + Cosine Similarity)

- Kelebihan:
  - Sederhana dan cepat: Menggunakan TF-IDF dan Cosine Similarity membuat perhitungannya efisien, bahkan untuk dataset menengah.

  - Interpretable: Sistem rekomendasi mudah dijelaskan karena hanya mencari kemiripan berdasarkan fitur konten (genre, deskripsi, dsb).

  - Tidak butuh data user lain: Cocok kalau user masih sedikit atau belum ada interaksi.

- Kekurangan:
  - Over-specialization: Rekomendasi cenderung terbatas pada film yang mirip, kurang eksploratif (kurang diverse).

  - Tidak bisa tangani cold start item: Film baru tanpa deskripsi yang cukup sulit untuk direkomendasikan secara akurat.

  - Tidak mempertimbangkan preferensi komunitas: Tidak tahu kalau film yang kurang mirip secara konten, tapi banyak disukai orang dengan selera mirip user.



### Collaborative Filtering

Sebelum memulai, kita memerlukan re-prep data terlebih dahulu.
```
titles_df = pd.read_csv('/content/dataset/titles.csv')
credits_df = pd.read_csv('/content/dataset/credits.csv')

from sklearn.preprocessing import LabelEncoder

# Encode user (name → user_id)
user_encoder = LabelEncoder()
credits_df['user_id'] = user_encoder.fit_transform(credits_df['name'])
user2user_encoded = dict(zip(credits_df['name'], credits_df['user_id']))
user_encoded2user = dict(zip(credits_df['user_id'], credits_df['name']))

# Encode item (movie ID → item_id)
item_encoder = LabelEncoder()
credits_df['item_id'] = item_encoder.fit_transform(credits_df['id'])
item2item_encoded = dict(zip(credits_df['id'], credits_df['item_id']))
item_encoded2item = dict(zip(credits_df['item_id'], credits_df['id']))

# Mapping nama aktor/sutradara jadi user_id
credits_df['user_id'] = user_encoder.fit_transform(credits_df['name'])

# Mapping ID film jadi item_id
credits_df['item_id'] = item_encoder.fit_transform(credits_df['id'])

# Tambahkan dummy rating
credits_df['rating'] = 1
```
- Encode nama pengguna (aktor/sutradara) menjadi user_id
Menggunakan LabelEncoder untuk mengubah nama menjadi angka unik, karena model machine learning hanya bisa bekerja dengan data numerik.

- Encode ID film menjadi item_id
Sama seperti di atas, ID film diubah menjadi angka agar bisa diproses oleh model.

- Buat mapping dictionary (user ↔ user_id, item ↔ item_id)
Mapping ini mempermudah konversi antara nama dan ID saat membuat rekomendasi.

- Tambahkan rating dummy (bernilai 1)
Karena tidak tersedia data rating asli, kita mensimulasikan bahwa seorang aktor/sutradara “menyukai” film yang mereka terlibat di dalamnya. Ini memungkinkan kita membuat model collaborative filtering berbasis interaksi yang ada.

```
X = credits_df[['user_id', 'item_id']].values
y = credits_df['rating'].values
```
Langkah ini menyiapkan data untuk pelatihan model, di mana X berisi pasangan user dan item, dan y adalah dummy rating sebagai target prediksi.

```
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```
Split data menjadi 80/20

```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size=50, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)
        
        self.item_embedding = layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.item_bias = layers.Embedding(num_items, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        
        dot_product = tf.reduce_sum(user_vector * item_vector, axis=1, keepdims=True)
        x = dot_product + user_bias + item_bias
        return tf.nn.sigmoid(x)
```
Kelas `RecommenderNet` ini membangun model collaborative filtering berbasis neural network. Model menggunakan embedding layer untuk merepresentasikan user dan item dalam vektor berdimensi rendah. Output prediksi adalah hasil perkalian dot product antara vektor user dan item, ditambah bias masing-masing, dan dilalui fungsi aktivasi sigmoid untuk menghasilkan skor antara 0 dan 1.

```
num_users = credits_df['user_id'].nunique()
num_items = credits_df['item_id'].nunique()

model = RecommenderNet(num_users, num_items, embedding_size=50)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```
Langkah ini menginisialisasi dan meng-compile model `RecommenderNet`. Jumlah user dan item ditentukan dari data unik. Model di-compile dengan loss function `BinaryCrossentropy` karena rating bersifat biner (dummy = 1), dan metrik evaluasinya menggunakan `RootMeanSquaredError (RMSE)` untuk menilai seberapa baik prediksi model terhadap data aktual.

```
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_val, y_val)
)

plt.plot(history.history['root_mean_squared_error'], label='Train RMSE', color='blue')
plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE', color='red')
plt.title('Model RMSE Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()

```

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
( A · B) / (||A|| * ||B||)

- **Interpretasi:** Semakin dekat ke 1, makin mirip.
- **Kaitan bisnis:** Memahami kesukaan user berdasarkan konten, meningkatkan kepuasan dan engagement.

### 📉 Collaborative Filtering

- **Metrik:** Root Mean Squared Error (RMSE)  
- **Formula:**  
√((1/n) * Σ(yi - ŷi)²)

- **Interpretasi:** Semakin kecil RMSE, semakin akurat model.
- **Kaitan bisnis:** Model mampu menyarankan film yang belum diketahui user tapi disukai oleh komunitas serupa.

## 🧠 Business Impact Revisited

- **Problem:** Kesulitan menemukan film yang relevan.
- **Goals:** Memberikan rekomendasi personal dan luas.
- **Solution:** Kombinasi CBF & CF menyasar dua tipe user:
  - User baru → CBF.
  - User aktif → CF.
- Evaluasi berdasarkan kemiripan dan prediksi performa rekomendasi yang diharapkan memperkuat engagement pengguna dan eksplorasi konten.


## Selengkapnya harap periksa ipynb saya.
