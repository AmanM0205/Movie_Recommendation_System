# 🎬 Movie Recommendation System

An end-to-end machine learning project that recommends movies using **content-based filtering**, **collaborative filtering** (matrix factorization and EASE) and a **tuned hybrid** of them. It's trained and evaluated on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset and served through a **Streamlit** web app and a CLI.

**Headline result:** the hybrid reaches **Precision@10 = 0.223** and **NDCG@10 = 0.302**, about **3× the popularity baseline**.

## Features

| Feature | How it works |
|---|---|
| **Similar movies** | TF-IDF over genres + user tags → cosine similarity |
| **"Fans also liked"** | EASE item-item weights (movies watched together) |
| **Personalised picks** | Hybrid of EASE, content and popularity scores |
| **New-user recommendations** | Rate a few movies → instant picks with no retraining |
| **Rating prediction** | Biased matrix factorization (Funk SVD) from scratch, RMSE 0.876 |
| **Cold start** | Bayesian-weighted popularity ranking |
| **Evaluation** | RMSE/MAE + Precision@K, Recall@K, NDCG@K, catalogue coverage |

## Project structure

```
Movie_Recommendation_System/
├── app.py                 # Streamlit web app
├── train.py               # evaluate, tune hybrid weight, train & save models
├── recommend.py           # command-line recommendations
├── requirements.txt
├── src/
│   ├── config.py          # paths & hyperparameters
│   ├── data_loader.py     # download, preprocess, train/test split
│   ├── models.py          # Popularity, Content-based, MatrixFactorization, EASE, Hybrid
│   ├── evaluate.py        # rating + ranking metrics
│   └── pipeline.py        # build / save / load model bundle
├── tests/test_models.py   # unit tests (synthetic data, no download)
├── data/raw/              # MovieLens is downloaded here automatically
├── models/                # trained model bundle (recommender.joblib)
└── reports/metrics.json   # evaluation results
```

## Quick start

```powershell
# 1. Create a virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. Download data, evaluate models and train the final model (~1 minute)
python train.py

# 3. Launch the web app
streamlit run app.py
```

CLI examples:

```powershell
python recommend.py --movie "matrix"     # similar movies
python recommend.py --user 1             # personalised for MovieLens user 1
python recommend.py --top                # most popular
pytest -q                                # run tests
```

Training options: `python train.py --factors 100 --epochs 60 --reg 0.08 --alpha 0.8 --beta 0.3` (or `--skip-eval` to only train).

## How it works

### 1. Data
MovieLens *latest-small*: 100,836 ratings (0.5–5★) from 610 users on 9,742 movies, plus free-text tags. The user–movie matrix is **98.3% empty**, which is the core challenge.

Preprocessing extracts the release year, normalises genres (`Sci-Fi` → `scifi`), and joins each movie's user tags. It then builds a text "soup" in which genres are repeated so they outweigh sparse tags.

**Train/test split:** 20% of *each user's* ratings are held out, so every test user has history in train. A further validation split is carved from train to tune the hybrid weight, so the test set is never used for tuning.

### 2. Models

**Popularity (baseline):** IMDb-style weighted rating
`score = v/(v+m)·R + m/(v+m)·C`, which shrinks movies with few votes toward the global mean.

**Content-based:** each movie is a TF-IDF vector (unigrams + bigrams). A user profile is the sum of their rated movies' vectors weighted by *(rating − user mean)*, so disliked movies push the profile away. Recommendations are the movies with the highest cosine similarity to the profile.

**Matrix factorization (rating prediction):** biased MF (Funk SVD), implemented from scratch in NumPy and trained with mini-batch SGD:

```
r̂(u,i) = μ + b_u + b_i + p_u · q_i
loss   = Σ (r − r̂)² + λ (b_u² + b_i² + ‖p_u‖² + ‖q_i‖²)
```

New users are handled by *folding in*: with the item factors fixed, `b_u` and `p_u` are solved in closed form with ridge regression.

**EASE (top-N ranking):** [Embarrassingly Shallow Autoencoder](https://arxiv.org/abs/1905.03375) (Steck, 2019). It learns an item-item matrix `B` from the binary *user-watched-movie* matrix `X` in closed form:

```
P = (XᵀX + λI)⁻¹,   B_ij = −P_ij / P_jj,   diag(B) = 0
scores(u) = x_u · B
```

**Hybrid:**
```
personal = α · minmax(EASE) + (1 − α) · minmax(content)
score    = (1 − β) · personal + β · minmax(popularity)
```
α and β are tuned in sequence by NDCG@10 on the validation split (best: α = 0.8, β = 0.3). Users with no ratings get the popularity ranking.

### 3. Results

Test set: 20,164 held-out ratings. A movie counts as *relevant* if the user rated it ≥ 4★. Candidates are all movies the user hasn't rated in train with ≥ 10 ratings (the same filter the app uses).

**Rating prediction** (lower is better)

| Model | RMSE | MAE |
|---|---|---|
| Global mean baseline | 1.050 | 0.831 |
| Matrix factorization | **0.876** | **0.666** |

**Top-10 recommendation** (higher is better)

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage |
|---|---|---|---|---|
| Popularity | 0.077 | 0.053 | 0.096 | 0.4% |
| Content-based | 0.018 | 0.017 | 0.023 | 11.6% |
| Matrix factorization | 0.038 | 0.023 | 0.045 | 5.4% |
| EASE | 0.218 | 0.204 | 0.292 | 4.7% |
| **Hybrid** | **0.223** | **0.211** | **0.302** | 4.2% |

### 4. What the results show

- **A good RMSE doesn't make a good ranker.** MF predicts ratings well, yet it scores *worse* than popularity at top-10 ranking. Optimising squared error rewards getting every rating roughly right, including obscure movies with a few 5★ ratings. Ranking only cares about getting the top 10 right.
- **Modelling *what* people watch beats modelling *how* they rate it.** EASE ignores the rating values and still almost triples popularity's precision.
- **Popularity is a strong baseline.** An early hybrid built on MF needed so much popularity weight that it almost collapsed into the popularity list (0.9% coverage). Swapping in EASE fixed this.
- **Content features add a little on top.** They give a modest lift over pure EASE (NDCG 0.292 → 0.302) and solve the item cold start: a brand-new movie with tags or genres can still be recommended.
- **Accuracy vs. diversity:** coverage stays low (around 4–5% of the catalogue). The app exposes α, β and the minimum-ratings filter as sliders so you can trade accuracy for discovery.

## Ideas for extending the project

- Add movie posters & overviews via the [TMDB API](https://developer.themoviedb.org/) (`links.csv` has `tmdbId`)
- Use sentence embeddings of plot summaries for richer content features
- Try other implicit-feedback models (ALS, BPR) or a neural recommender (two-tower / NCF)
- Tune EASE's λ and MF hyperparameters with a proper grid or Optuna search
- Re-rank for diversity and novelty, not just accuracy
- Scale up to MovieLens 25M; serve via a FastAPI endpoint; containerise with Docker

## Dataset licence

MovieLens data © GroupLens Research, University of Minnesota. Used for research/education under the [GroupLens terms](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html); the data is downloaded at runtime and not redistributed in this repo.
