# 🎬 Movie Recommender System: Multi-Model Approach [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/95karthik4/Movie-Recommender-)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Library](https://img.shields.io/badge/Lib-Scikit--Surprise-orange) ![Library](https://img.shields.io/badge/Lib-TuriCreate-green) ![Topic](https://img.shields.io/badge/Topic-Recommender%20Systems-red)

### 🍿 Project Overview
In the age of information overload, personalized recommendations are essential for user retention. This project builds and compares **five distinct recommendation engines** using the famous **MovieLens dataset**.

From simple popularity metrics to advanced matrix factorization and hybrid techniques, this project explores the full spectrum of how AI predicts user preferences.

---

### 🧠 Algorithms Implemented
I engineered five different systems to solve the recommendation problem from different angles:

1.  **🔥 Popularity-Based Recommender:**
    * *Logic:* "Everyone likes these, so you might too."
    * *Method:* Ranks movies by vote count and average rating. Solves the "Cold Start" problem for new users.

2.  **🏷️ Content-Based Filtering:**
    * *Logic:* "You liked *Toy Story*, so you'll like *Finding Nemo*."
    * *Method:* Uses **TF-IDF Vectorization** on movie genres and metadata to find items with similar characteristics.

3.  **🤝 Collaborative Filtering (SVD):**
    * *Logic:* "Users similar to you liked this movie."
    * *Method:* Utilizes **Singular Value Decomposition (SVD)** from the `Surprise` library to uncover latent factors in user-item interactions.

4.  **⚡ TuriCreate Recommender:**
    * *Method:* Leverages Apple's `TuriCreate` framework for highly optimized, scalable item-similarity models.

5.  **🧬 Hybrid Recommender:**
    * *Logic:* The best of both worlds.
    * *Method:* Combines SVD (Collaborative) and TF-IDF (Content) scores to produce a weighted, more robust prediction.

---

### 📊 Performance & Insights
The models were evaluated on the MovieLens dataset containing 100,000 ratings.

* **SVD (Collaborative Filtering)** achieved the lowest error rate (RMSE), making it the most accurate for existing users.
* **Content-Based** successfully recommended niche movies that popular models missed.
* **Hybrid Approach** provided the most balanced suggestions, mitigating the weaknesses of single-model systems.

---

### 🛠️ Setup & Usage
1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-surprise turicreate
    ```
    *(Note: TuriCreate requires a specific Python environment, usually Linux/Mac or WSL on Windows)*

2.  **Run the Notebook:**
    Open `Karthik_Task4_Movie_Recommenders.ipynb` in Jupyter Notebook.

3.  **Data:**
    The project uses the MovieLens 100k dataset (`u.data`, `u.item`), which is loaded directly within the notebook.

---

### 👨‍💻 About the Author
**Karthik Kunnamkumarath**
*Aerospace Engineer | Project Management Professional (PMP) | AI Solutions Developer*

I combine engineering precision with data science to solve complex problems.
* 📍 Toronto, ON
* 💼 [LinkedIn Profile](https://linkedin.com/in/4karthik95)
* 📧 Aero13027@gmail.com


---

### 💻 Code Snippet: Building a Hybrid Engine
Here is the logic used to blend Collaborative and Content-based predictions:

```python
def hybrid_prediction(user_id, movie_id, alpha=0.5):
    # Get score from Collaborative Filtering (SVD)
    svd_score = svd_model.predict(user_id, movie_id).est
    
    # Get score from Content-Based Filtering (Cosine Similarity)
    # (Assuming we have a similarity matrix 'sim_matrix')
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    content_score = calculate_weighted_score(sim_scores)
    
    # Weighted average
    final_score = (alpha * svd_score) + ((1 - alpha) * content_score)
    return final_score
