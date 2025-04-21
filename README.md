# Sistem Rekomendasi Film - Achmad Fariz

## Project Overview

Seiring meningkatnya jumlah pengguna internet dan volume informasi yang tersedia secara daring sejak akhir abad ke-20, pengguna kerap mengalami kesulitan dalam menemukan informasi yang benar-benar relevan. Dalam konteks hiburan digital seperti film, hal ini memunculkan kebutuhan akan sistem rekomendasi yang mampu membantu pengguna menemukan film sesuai preferensi mereka tanpa harus melakukan pencarian berulang. 

Rekomendasi film menjadi solusi yang populer untuk mempermudah proses pencarian dan pengambilan keputusan pengguna terhadap konten hiburan yang ingin mereka konsumsi (Choi et al., 2012).

Menurut Choi et al. (2012), sistem rekomendasi sangat penting untuk mengurangi usaha pencarian informasi yang berulang dengan cara menyarankan konten yang relevan berdasarkan pola perilaku atau preferensi pengguna sebelumnya. Sistem ini secara signifikan dapat meningkatkan efisiensi dan pengalaman pengguna dalam menjelajahi konten digital, seperti film.

Lebih lanjut, Goyani & Chaurasiya (2020) menjelaskan bahwa sistem rekomendasi film dapat dikembangkan melalui dua pendekatan utama:

- **Collaborative Filtering**, yang merekomendasikan item berdasarkan kemiripan antar pengguna
- **Content-Based Filtering**, yang memanfaatkan preferensi eksplisit dari pengguna untuk menyarankan konten serupa.

Keduanya memiliki kelebihan masing-masing, dan penerapan gabungan dari kedua pendekatan ini dapat meningkatkan akurasi dan personalisasi rekomendasi.

Topik ini diangkat karena dunia hiburan adalah bidang yang sangat dinamis dan dekat dengan kehidupan sehari-hari, sehingga sistem seperti ini akan memiliki nilai guna praktis yang tinggi serta tantangan teknis yang menarik untuk dipecahkan.

**Referensi:**

- Choi, S. M., Ko, S. K., & Han, Y. S. (2012). A movie recommendation algorithm based on genre correlations. *Expert Systems with Applications, 39(9)*, 8079–8085.  
- Goyani, M., & Chaurasiya, N. (2020). A review of movie recommendation system: Limitations, Survey and Challenges. *ELCVIA, 19(3)*, 0018–37.

---

## Business Understanding

### Problem Statements

- Bagaimana cara membangun sistem rekomendasi yang dapat memberikan saran film atau acara TV yang relevan untuk pengguna berdasarkan konten dari film atau acara yang pernah mereka sukai sebelumnya?
- Dengan memanfaatkan informasi dari pengguna lain yang memiliki ketertarikan serupa, bagaimana sistem dapat merekomendasikan film atau acara TV yang mungkin disukai pengguna namun belum pernah ditonton sebelumnya?

### Goals

- Menghasilkan rekomendasi film atau acara TV yang dipersonalisasi untuk setiap pengguna menggunakan teknik Content-Based Filtering berdasarkan genre, deskripsi, dan skor film.
- Menghasilkan rekomendasi film atau acara TV dengan teknik Collaborative Filtering berdasarkan pola kesukaan pengguna lain.

### Solution Approach

Pendekatan utama:

- **Content-Based Filtering**:
  - Gunakan `titles.csv` untuk membangun profil konten (genres, deskripsi, skor).
  - Hitung kemiripan antar film (TF-IDF + cosine similarity).

- **Collaborative Filtering**:
  - Simulasikan matriks user-item karena tidak tersedia data rating eksplisit.
  - Gunakan item-based collaborative filtering atau matrix factorization (SVD).

---

## Data Understanding

Dataset yang digunakan:

- `titles.csv` — berisi informasi tentang film dan acara
- `credits.csv` — berisi informasi tentang aktor dan sutradara

**Contoh struktur data:**

- Jumlah fitur: 14
- Banyak missing values yang perlu ditangani

**Distribusi tipe konten:**

> _[Masukkan gambar: Distribusi Tipe Konten (Movie vs Show)]_

**Insight:** Mayoritas konten berupa film (MOVIE) sebanyak 3744 judul, dibandingkan dengan 2106 untuk acara (SHOW).

**Distribusi genre terbanyak:**

> _[Masukkan gambar: 10 Genre Terbanyak]_

**Insight:** Drama dan komedi mendominasi, diikuti oleh thriller, action, dan romance.

**Negara produksi terbanyak:**

> _[Masukkan gambar: 10 Negara Produksi Terbanyak]_

**Insight:** US, India, Inggris, dan Jepang jadi negara produksi paling dominan.

**Distribusi skor film:**

> _[Masukkan gambar: Distribusi IMDb dan TMDb Score]_

**Insight:** Skor rata-rata film berkisar antara 6.5–6.8.

**Peran dan popularitas:**

> _[Masukkan gambar: Jumlah Peran Berdasarkan Role]_  
> _[Masukkan tabel: Top Aktor & Sutradara]_

**Insight:** Boman Irani dan Kareena Kapoor Khan mendominasi sebagai aktor. Raúl Campos dan Jan Suter adalah sutradara paling produktif.

---

## Data Preparation

- Duplikat dicek dan dihapus dari `titles_df` dan `credits_df`
- Missing values diisi:
  - Kategorikal → "Unknown"
  - Numerik → nilai rata-rata
  - Teks → string kosong untuk mencegah error saat pemrosesan
- Gabungkan aktor & sutradara per film
- Buat fitur gabungan `content` berisi:
  - Judul, deskripsi, genre, sertifikasi umur, aktor, dan sutradara

Contoh isi `content` (per baris):

Berikut adalah **markdown lengkap** untuk bagian _Modelling_ dan _Evaluation_ laporan kamu. Kamu tinggal **copy-paste** saja ke file `.md` atau ke Google Colab (Markdown cell):

---

```markdown
# Modelling

## Content-Based Filtering

```python
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

Langkah ini bertujuan untuk membangun sistem **Content-Based Filtering**. Dengan menggunakan `TfidfVectorizer`, kita mengubah teks pada kolom `content` menjadi representasi numerik berbasis frekuensi kata, lalu menghitung kemiripan antar film menggunakan **cosine similarity**. Hasilnya adalah matriks yang menunjukkan seberapa mirip satu film dengan yang lain berdasarkan kontennya, sehingga sistem bisa merekomendasikan film serupa.

```python
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

Fungsi ini digunakan untuk **memberikan rekomendasi film** berbasis konten. Saat pengguna memasukkan judul film, sistem akan:

1. Mencari indeks film tersebut dalam dataset.
2. Mengambil skor kemiripan (cosine similarity) dengan semua film lainnya.
3. Mengurutkan berdasarkan skor tertinggi dan **mengabaikan film itu sendiri**.
4. Mengembalikan daftar top-`n` film paling mirip berdasarkan konten seperti deskripsi, genre, aktor, dll.

```python
recommend_content_based("Narcos", n=5)
```

### Content-Based Filtering (TF-IDF + Cosine Similarity)

**Kelebihan:**
- Sederhana dan cepat.
- Interpretasi mudah.
- Tidak butuh data user lain.

**Kekurangan:**
- Over-specialization.
- Tidak bisa menangani cold start item.
- Tidak mempertimbangkan preferensi komunitas.

---

## Collaborative Filtering

### Data Preparation

```python
titles_df = pd.read_csv('/content/dataset/titles.csv')
credits_df = pd.read_csv('/content/dataset/credits.csv')
```

```python
from sklearn.preprocessing import LabelEncoder

# Encode user dan item
user_encoder = LabelEncoder()
credits_df['user_id'] = user_encoder.fit_transform(credits_df['name'])

item_encoder = LabelEncoder()
credits_df['item_id'] = item_encoder.fit_transform(credits_df['id'])

# Tambahkan dummy rating
credits_df['rating'] = 1
```

```python
X = credits_df[['user_id', 'item_id']].values
y = credits_df['rating'].values
```

```python
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Model Building

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size=50, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = layers.Embedding(input_dim=num_users, output_dim=embedding_size,
                                               embeddings_initializer='he_normal',
                                               embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.user_bias = layers.Embedding(num_users, 1)

        self.item_embedding = layers.Embedding(input_dim=num_items, output_dim=embedding_size,
                                               embeddings_initializer='he_normal',
                                               embeddings_regularizer=keras.regularizers.l2(1e-6))
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

```python
num_users = credits_df['user_id'].nunique()
num_items = credits_df['item_id'].nunique()

model = RecommenderNet(num_users, num_items, embedding_size=50)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```

```python
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_val, y_val)
)
```

```python
plt.plot(history.history['root_mean_squared_error'], label='Train RMSE', color='blue')
plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE', color='red')
plt.title('Model RMSE Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()
```

### Fungsi Rekomendasi

```python
def recommend_collaborative(user_name, n=10):
    if user_name not in user2user_encoded:
        return f"User '{user_name}' tidak ditemukan."

    user_id = user2user_encoded[user_name]
    all_item_ids = credits_df['item_id'].unique()
    items_watched_by_user = credits_df[credits_df['user_id'] == user_id]['item_id'].tolist()
    items_not_watched = np.setdiff1d(all_item_ids, items_watched_by_user)

    user_input = np.array([[user_id, item_id] for item_id in items_not_watched])
    predicted_ratings = model.predict(user_input).flatten()

    top_indices = predicted_ratings.argsort()[-n:][::-1]
    top_item_ids = items_not_watched[top_indices]
    recommended_titles = titles_df[titles_df['id'].isin([item_encoded2item[item] for item in top_item_ids])]

    return recommended_titles[['title', 'type', 'release_year']].reset_index(drop=True)
```

```python
recommend_collaborative("Robert De Niro", n=5)
```

### Collaborative Filtering (Neural Collaborative Filtering)

**Kelebihan:**
- Menangkap pola kompleks user-item.
- Rekomendasi sangat personal.
- Mendukung hubungan non-linear.

**Kekurangan:**
- Membutuhkan banyak data interaksi.
- Cold start problem.
- Proses training lebih berat.

---

# Evaluation

## 1. Content-Based Filtering (CBF)

**Metric Used:** Cosine Similarity  
**Formula:**  
```
( A · B ) / ( ||A|| * ||B|| )
```

**Interpretasi:** Nilai antara 0 dan 1. Semakin tinggi, semakin mirip.  
**Business Relevance:** Menjawab kebutuhan pengguna menemukan film mirip dengan yang mereka sukai. Meningkatkan kepuasan dan loyalitas.

---

## 2. Collaborative Filtering (CF)

**Metric Used:** Root Mean Squared Error (RMSE)  
**Formula:**  
```
√((1/n) * Σ(yi - ŷi)²)
```

**Interpretasi:** Nilai lebih rendah = prediksi lebih akurat.  
**Business Relevance:** Memberikan rekomendasi yang lebih "tersembunyi" namun relevan bagi pengguna.

---

## 3. Relevance to Business Understanding

**Problem Statement:**
- CBF: Untuk user yang ingin film dengan gaya serupa.
- CF: Untuk user yang ingin eksplorasi film baru lewat selera user lain.

**Goals:**
- CBF → Personal berdasarkan konten.
- CF → Personal berdasarkan pola pengguna lain.

**Combined Approach:**
Menggunakan keduanya untuk menciptakan sistem rekomendasi yang:
- Kuat untuk berbagai tipe user.
- Adaptif terhadap cold start dan eksplorasi.
- Memaksimalkan engagement dan waktu tonton.
