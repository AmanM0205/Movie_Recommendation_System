"""Recommendation models.

Every recommender exposes the same scoring interface:

    score(movie_ids, user_id=None, user_ratings=None) -> np.ndarray

which returns one score per entry of ``movie_ids`` (higher = better). That shared
interface lets the hybrid model and the evaluation code treat all models alike.
``user_ratings`` is a DataFrame with ``movieId`` and ``rating`` columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import EASE_PARAMS, HYBRID_ALPHA, HYBRID_BETA, MF_PARAMS, RANDOM_STATE


def _lookup(index: pd.Index, values, default=-1) -> np.ndarray:
    pos = index.get_indexer(pd.Index(values))
    return np.where(pos >= 0, pos, default)


# --------------------------------------------------------------------------- #
# 1. Popularity baseline
# --------------------------------------------------------------------------- #
class PopularityRecommender:
    """Ranks movies by an IMDb-style weighted (Bayesian) average rating.

    weighted = v/(v+m) * R + m/(v+m) * C
    where v = #ratings, R = mean rating, C = global mean, m = minimum votes.
    """

    def __init__(self, min_votes_quantile: float = 0.8):
        self.min_votes_quantile = min_votes_quantile

    def fit(self, ratings: pd.DataFrame) -> "PopularityRecommender":
        stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
        c = ratings["rating"].mean()
        m = stats["count"].quantile(self.min_votes_quantile)
        v, r = stats["count"], stats["mean"]
        self.scores_ = (v / (v + m) * r + m / (v + m) * c).rename("score")
        self.index_ = pd.Index(self.scores_.index)
        return self

    def score(self, movie_ids, user_id=None, user_ratings=None) -> np.ndarray:
        pos = _lookup(self.index_, movie_ids)
        values = self.scores_.to_numpy()
        return np.where(pos >= 0, values[pos], 0.0)


# --------------------------------------------------------------------------- #
# 2. Content-based filtering
# --------------------------------------------------------------------------- #
class ContentBasedRecommender:
    """TF-IDF over genres + user tags; cosine similarity between movies.

    For a user, a profile vector is built as the rating-weighted sum of the
    TF-IDF vectors of the movies they rated (ratings centred on the user's
    mean, so disliked movies push the profile away).
    """

    def __init__(self, max_features: int = 20000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range

    def fit(self, movies: pd.DataFrame) -> "ContentBasedRecommender":
        self.vectorizer_ = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
        )
        # Rows are L2-normalised, so a dot product equals cosine similarity.
        self.matrix_ = self.vectorizer_.fit_transform(movies["soup"].fillna("")).tocsr()
        self.index_ = pd.Index(movies["movieId"])
        self.popularity_ = (movies["num_ratings"].to_numpy() if "num_ratings" in movies
                            else np.zeros(len(movies)))
        return self

    def similar_movies(self, movie_id: int, n: int = 10) -> pd.Series:
        """Return the n most similar movie ids with their cosine similarity."""
        if movie_id not in self.index_:
            return pd.Series(dtype=float)
        idx = self.index_.get_loc(movie_id)
        sims = (self.matrix_ @ self.matrix_[idx].T).toarray().ravel()
        sims[idx] = -np.inf
        # Many movies share identical genre vectors; break ties by popularity.
        top = np.lexsort((-self.popularity_, -np.round(sims, 6)))[:n]
        return pd.Series(sims[top], index=self.index_[top], name="score")

    def user_profile(self, user_ratings: pd.DataFrame):
        if user_ratings is None or len(user_ratings) == 0:
            return None
        pos = self.index_.get_indexer(user_ratings["movieId"])
        mask = pos >= 0
        if not mask.any():
            return None
        r = user_ratings["rating"].to_numpy(dtype=float)[mask]
        w = r - r.mean()
        if np.allclose(w, 0):  # user gave every movie the same rating
            w = np.ones_like(r)
        weights = np.zeros(self.matrix_.shape[0])
        np.add.at(weights, pos[mask], w)
        profile = self.matrix_.T @ weights  # dense vector over TF-IDF features
        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else None

    def score(self, movie_ids, user_id=None, user_ratings=None) -> np.ndarray:
        profile = self.user_profile(user_ratings)
        if profile is None:
            return np.zeros(len(movie_ids))
        sims = self.matrix_ @ profile
        pos = _lookup(self.index_, movie_ids)
        return np.where(pos >= 0, sims[pos], 0.0)


# --------------------------------------------------------------------------- #
# 3. Collaborative filtering for rating prediction: biased MF (Funk SVD)
# --------------------------------------------------------------------------- #
class MatrixFactorization:
    """Biased matrix factorization trained with mini-batch SGD.

    r_hat(u, i) = mu + b_u + b_i + p_u . q_i
    Loss: squared error + reg * (b_u^2 + b_i^2 + |p_u|^2 + |q_i|^2)
    """

    def __init__(self, n_factors=50, n_epochs=40, lr=0.01, reg=0.05, batch_size=256,
                 random_state=RANDOM_STATE, verbose=False):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

    @classmethod
    def default(cls, **overrides) -> "MatrixFactorization":
        return cls(**{**MF_PARAMS, **overrides})

    def fit(self, ratings: pd.DataFrame) -> "MatrixFactorization":
        rng = np.random.default_rng(self.random_state)
        self.user_index_ = pd.Index(np.sort(ratings["userId"].unique()))
        self.item_index_ = pd.Index(np.sort(ratings["movieId"].unique()))
        u = self.user_index_.get_indexer(ratings["userId"])
        i = self.item_index_.get_indexer(ratings["movieId"])
        r = ratings["rating"].to_numpy(dtype=float)

        n_users, n_items, k = len(self.user_index_), len(self.item_index_), self.n_factors
        self.mu_ = r.mean()
        self.bu_ = np.zeros(n_users)
        self.bi_ = np.zeros(n_items)
        self.P_ = rng.normal(0, 0.1, (n_users, k))
        self.Q_ = rng.normal(0, 0.1, (n_items, k))
        self.item_counts_ = np.bincount(i, minlength=n_items)

        lr, reg = self.lr, self.reg
        self.history_ = []
        for epoch in range(self.n_epochs):
            order = rng.permutation(len(r))
            for start in range(0, len(r), self.batch_size):
                b = order[start:start + self.batch_size]
                ub, ib = u[b], i[b]
                pu, qi = self.P_[ub], self.Q_[ib]
                err = r[b] - (self.mu_ + self.bu_[ub] + self.bi_[ib] + np.einsum("ij,ij->i", pu, qi))
                np.add.at(self.bu_, ub, lr * (err - reg * self.bu_[ub]))
                np.add.at(self.bi_, ib, lr * (err - reg * self.bi_[ib]))
                np.add.at(self.P_, ub, lr * (err[:, None] * qi - reg * pu))
                np.add.at(self.Q_, ib, lr * (err[:, None] * pu - reg * qi))
            train_rmse = float(np.sqrt(np.mean((r - self._predict_idx(u, i)) ** 2)))
            self.history_.append(train_rmse)
            if self.verbose:
                print(f"  epoch {epoch + 1:2d}/{self.n_epochs}  train RMSE {train_rmse:.4f}")
        return self

    def _predict_idx(self, u, i):
        return self.mu_ + self.bu_[u] + self.bi_[i] + np.einsum("ij,ij->i", self.P_[u], self.Q_[i])

    def predict(self, user_ids, movie_ids) -> np.ndarray:
        """Predict ratings; unknown users/movies fall back to the available biases."""
        u = self.user_index_.get_indexer(pd.Index(user_ids))
        i = self.item_index_.get_indexer(pd.Index(movie_ids))
        ku, ki = u >= 0, i >= 0
        bu = np.where(ku, self.bu_[u], 0.0)
        bi = np.where(ki, self.bi_[i], 0.0)
        pu = np.where(ku[:, None], self.P_[u], 0.0)
        qi = np.where(ki[:, None], self.Q_[i], 0.0)
        return np.clip(self.mu_ + bu + bi + np.einsum("ij,ij->i", pu, qi), 0.5, 5.0)

    def user_factors(self, user_id=None, user_ratings=None):
        """Return (b_u, p_u). Known users use learned factors; new users are folded in."""
        if user_id is not None and user_id in self.user_index_:
            idx = self.user_index_.get_loc(user_id)
            return self.bu_[idx], self.P_[idx]
        return self.fold_in(user_ratings)

    def fold_in(self, user_ratings: pd.DataFrame | None):
        """Solve for a new user's bias and factors by ridge regression with Q fixed."""
        if user_ratings is None or len(user_ratings) == 0:
            return 0.0, np.zeros(self.n_factors)
        i = self.item_index_.get_indexer(user_ratings["movieId"])
        mask = i >= 0
        if not mask.any():
            return 0.0, np.zeros(self.n_factors)
        i = i[mask]
        y = user_ratings["rating"].to_numpy(dtype=float)[mask] - self.mu_ - self.bi_[i]
        X = np.hstack([np.ones((len(i), 1)), self.Q_[i]])
        A = X.T @ X + self.reg * len(i) * np.eye(X.shape[1]) + 1e-6 * np.eye(X.shape[1])
        w = np.linalg.solve(A, X.T @ y)
        return w[0], w[1:]

    def score(self, movie_ids, user_id=None, user_ratings=None) -> np.ndarray:
        bu, pu = self.user_factors(user_id, user_ratings)
        i = self.item_index_.get_indexer(pd.Index(movie_ids))
        known = i >= 0
        scores = np.full(len(i), self.mu_ + bu)
        scores[known] += self.bi_[i[known]] + self.Q_[i[known]] @ pu
        return scores

    def similar_movies(self, movie_id: int, n: int = 10, min_ratings: int = 20) -> pd.Series:
        """Movies whose latent factors are closest (cosine) to the given movie."""
        if movie_id not in self.item_index_:
            return pd.Series(dtype=float)
        idx = self.item_index_.get_loc(movie_id)
        Qn = normalize(self.Q_)
        sims = Qn @ Qn[idx]
        sims[self.item_counts_ < min_ratings] = -np.inf
        sims[idx] = -np.inf
        top = np.argsort(-sims)[:n]
        return pd.Series(sims[top], index=self.item_index_[top], name="score")


# --------------------------------------------------------------------------- #
# 4. Collaborative filtering for ranking: EASE (implicit feedback)
# --------------------------------------------------------------------------- #
class EASERecommender:
    """Embarrassingly Shallow Autoencoder (Steck, 2019).

    Learns an item-item weight matrix B from the binary "user watched movie"
    matrix X by solving, in closed form,

        min_B ||X - XB||^2 + lambda ||B||^2   subject to diag(B) = 0
        =>  P = (X^T X + lambda I)^-1,   B_ij = -P_ij / P_jj

    A user's scores are x_u @ B. Unlike the rating-prediction MF model, EASE is
    optimised to reconstruct *which* movies a user watches, which makes it a far
    stronger top-N ranker. New users need no retraining: just their movie list.
    """

    def __init__(self, lambda_: float = 300.0, min_item_ratings: int = 5, like_threshold: float = 0.0):
        self.lambda_ = lambda_
        self.min_item_ratings = min_item_ratings
        self.like_threshold = like_threshold  # ratings below this don't count as interactions

    def fit(self, ratings: pd.DataFrame) -> "EASERecommender":
        counts = ratings["movieId"].value_counts()
        self.item_index_ = pd.Index(np.sort(counts[counts >= self.min_item_ratings].index))
        r = ratings[ratings["movieId"].isin(self.item_index_)]
        users = pd.Index(r["userId"].unique())
        X = sparse.csr_matrix(
            (np.ones(len(r)), (users.get_indexer(r["userId"]), self.item_index_.get_indexer(r["movieId"]))),
            shape=(len(users), len(self.item_index_)))
        G = (X.T @ X).toarray()
        G[np.diag_indices_from(G)] += self.lambda_
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        np.fill_diagonal(B, 0.0)
        self.B_ = B.astype(np.float32)
        return self

    def score(self, movie_ids, user_id=None, user_ratings=None) -> np.ndarray:
        x = np.zeros(len(self.item_index_), dtype=np.float32)
        if user_ratings is not None and len(user_ratings):
            liked = user_ratings[user_ratings["rating"] >= self.like_threshold]
            pos = self.item_index_.get_indexer(liked["movieId"])
            x[pos[pos >= 0]] = 1.0
        scores = x @ self.B_
        pos = self.item_index_.get_indexer(pd.Index(movie_ids))
        floor = scores.min() if len(scores) else 0.0  # movies outside the model rank last
        return np.where(pos >= 0, scores[pos], floor).astype(float)

    def similar_movies(self, movie_id: int, n: int = 10) -> pd.Series:
        """'People who watched this also watched': row of B for a single movie."""
        if movie_id not in self.item_index_:
            return pd.Series(dtype=float)
        row = self.B_[self.item_index_.get_loc(movie_id)].astype(float)
        top = np.argsort(-row)[:n]
        return pd.Series(row[top], index=self.item_index_[top], name="score")


# --------------------------------------------------------------------------- #
# 5. Hybrid
# --------------------------------------------------------------------------- #
def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = np.min(x), np.max(x)
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


class HybridRecommender:
    """Weighted blend of signals, each min-max scaled to [0, 1]:

        personal = alpha * collaborative + (1 - alpha) * content
        score    = (1 - beta) * personal + beta * popularity

    `collaborative` is any model with the shared `score` interface (EASE by
    default). Users with no ratings get the popularity ranking (cold start).
    """

    def __init__(self, collaborative, content: ContentBasedRecommender,
                 popularity: PopularityRecommender, alpha: float = HYBRID_ALPHA,
                 beta: float = HYBRID_BETA):
        self.collaborative = collaborative
        self.content = content
        self.popularity = popularity
        self.alpha = alpha
        self.beta = beta

    def score(self, movie_ids, user_id=None, user_ratings=None) -> np.ndarray:
        pop = self.popularity.score(movie_ids)
        if user_ratings is None or len(user_ratings) == 0:
            return pop
        cf = _minmax(self.collaborative.score(movie_ids, user_id, user_ratings))
        cb = _minmax(self.content.score(movie_ids, user_id, user_ratings))
        personal = self.alpha * cf + (1 - self.alpha) * cb
        return (1 - self.beta) * personal + self.beta * _minmax(pop)


def top_n(model, movie_ids, n=10, user_id=None, user_ratings=None, exclude=None,
          allowed_mask=None) -> pd.Series:
    """Generic top-N helper for any model with the shared `score` interface."""
    movie_ids = np.asarray(movie_ids)
    scores = np.asarray(model.score(movie_ids, user_id=user_id, user_ratings=user_ratings), dtype=float)
    if exclude is not None and len(exclude):
        scores[np.isin(movie_ids, list(exclude))] = -np.inf
    if allowed_mask is not None:
        scores[~allowed_mask] = -np.inf
    n = min(n, int(np.isfinite(scores).sum()))
    top = np.argpartition(-scores, n - 1)[:n] if n > 0 else np.array([], dtype=int)
    top = top[np.argsort(-scores[top])]
    return pd.Series(scores[top], index=movie_ids[top], name="score")
