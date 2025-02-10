"""Unit tests on a tiny synthetic dataset (no download needed).  Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import preprocess_movies, train_test_split_by_user
from src.evaluate import ndcg_at_k, precision_at_k, recall_at_k, evaluate_ranking
from src.models import (ContentBasedRecommender, EASERecommender, HybridRecommender,
                        MatrixFactorization, PopularityRecommender, top_n)


@pytest.fixture
def data():
    movies = pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5, 6],
        "title": ["Space War (1990)", "Star Battle (1992)", "Love Story (1985)",
                  "Romance Again (1999)", "Space Love (2001)", "Cartoon Fun (2005)"],
        "genres": ["Action|Sci-Fi", "Action|Sci-Fi", "Romance|Drama",
                   "Romance|Drama", "Sci-Fi|Romance", "Animation|Children"],
    })
    rng = np.random.default_rng(0)
    rows = []
    for u in range(1, 31):
        likes_scifi = u % 2 == 0
        for m in movies.movieId:
            scifi = m in (1, 2, 5)
            base = 4.5 if scifi == likes_scifi else 1.5
            rows.append((u, m, float(np.clip(base + rng.normal(0, 0.3), 0.5, 5))))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    tags = pd.DataFrame({"userId": [1], "movieId": [1], "tag": ["Spaceships"]})
    return preprocess_movies(movies, ratings, tags), ratings


def test_preprocess(data):
    movies, _ = data
    assert movies.loc[0, "year"] == 1990
    assert movies.loc[0, "clean_title"] == "Space War"
    assert "scifi" in movies.loc[0, "soup"] and "spaceships" in movies.loc[0, "soup"]


def test_split_keeps_every_user(data):
    _, ratings = data
    train, test = train_test_split_by_user(ratings, test_size=0.3)
    assert len(train) + len(test) == len(ratings)
    assert set(test.userId) <= set(train.userId)


def test_content_similarity(data):
    movies, _ = data
    cb = ContentBasedRecommender().fit(movies)
    assert cb.similar_movies(1, n=1).index[0] == 2  # both Action|Sci-Fi


def test_matrix_factorization_learns(data):
    _, ratings = data
    mf = MatrixFactorization(n_factors=4, n_epochs=60, lr=0.02, batch_size=32).fit(ratings)
    pred = mf.predict(ratings.userId, ratings.movieId)
    assert np.sqrt(np.mean((pred - ratings.rating) ** 2)) < 0.6
    # unknown user / movie must not crash
    assert mf.predict([999], [999])[0] == pytest.approx(np.clip(mf.mu_, 0.5, 5))


def test_fold_in_new_user(data):
    _, ratings = data
    mf = MatrixFactorization(n_factors=4, n_epochs=60, lr=0.02, batch_size=32).fit(ratings)
    new_user = pd.DataFrame({"movieId": [1, 2, 3], "rating": [5.0, 5.0, 1.0]})
    scores = mf.score([4, 5], user_ratings=new_user)
    assert scores[1] > scores[0]  # sci-fi fan should prefer movie 5 over romance 4


def test_ease_ranks_same_taste_movies(data):
    movies, ratings = data
    liked = ratings[ratings.rating >= 3]  # watched = liked, so tastes are learnable
    ease = EASERecommender(lambda_=1.0, min_item_ratings=1).fit(liked)
    scifi_fan = pd.DataFrame({"movieId": [1, 2], "rating": [5.0, 5.0]})
    s = ease.score([3, 4, 5, 99], user_ratings=scifi_fan)
    assert s[2] > s[0] and s[2] > s[1]  # sci-fi movie 5 beats romance movies 3, 4
    assert s[3] == s.min()               # unknown movie ranks last


def test_hybrid_and_top_n(data):
    movies, ratings = data
    hybrid = HybridRecommender(
        EASERecommender(lambda_=1.0, min_item_ratings=1).fit(ratings),
        ContentBasedRecommender().fit(movies), PopularityRecommender().fit(ratings),
        alpha=0.5, beta=0.2)
    ur = ratings[ratings.userId == 2]
    recs = top_n(hybrid, movies.movieId, n=3, user_id=2, user_ratings=ur, exclude={1})
    assert len(recs) == 3 and 1 not in recs.index
    # cold start falls back to popularity
    assert np.allclose(hybrid.score(movies.movieId), hybrid.popularity.score(movies.movieId))


def test_ranking_metrics():
    recs, rel = [1, 2, 3, 4], {2, 4, 9}
    assert precision_at_k(recs, rel, 4) == 0.5
    assert recall_at_k(recs, rel, 4) == pytest.approx(2 / 3)
    assert ndcg_at_k([2, 4, 9], rel, 3) == pytest.approx(1.0)
    assert ndcg_at_k([], rel, 3) == 0.0


def test_evaluate_ranking_runs(data):
    movies, ratings = data
    train, test = train_test_split_by_user(ratings, test_size=0.5)
    res = evaluate_ranking(PopularityRecommender().fit(train), train, test, movies.movieId, k=2)
    assert 0 <= res["precision@2"] <= 1 and res["users_evaluated"] > 0
