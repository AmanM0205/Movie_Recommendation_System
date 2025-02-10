"""Evaluation metrics for rating prediction and top-K recommendation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RELEVANCE_THRESHOLD, TOP_K


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def precision_at_k(recommended, relevant: set, k: int) -> float:
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended, relevant: set, k: int) -> float:
    return len(set(recommended[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended, relevant: set, k: int) -> float:
    gains = [1.0 if m in relevant else 0.0 for m in recommended[:k]]
    dcg = sum(g / np.log2(rank + 2) for rank, g in enumerate(gains))
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_ranking(model, train: pd.DataFrame, test: pd.DataFrame, movie_ids,
                     k: int = TOP_K, threshold: float = RELEVANCE_THRESHOLD,
                     min_ratings: int = 0) -> dict[str, float]:
    """Top-K metrics. For each user: rank every movie they did not rate in train,
    and check how many of their highly-rated test movies land in the top K.

    `min_ratings` restricts candidates to movies with at least that many ratings
    in train (the same filter the app applies to hide obscure titles)."""
    movie_ids = np.asarray(movie_ids)
    index = pd.Index(movie_ids)
    counts = train["movieId"].value_counts()
    blocked = pd.Series(movie_ids).map(counts).fillna(0).to_numpy() < min_ratings
    relevant_by_user = test[test["rating"] >= threshold].groupby("userId")["movieId"].apply(set)
    train_by_user = {u: g for u, g in train.groupby("userId")}

    precisions, recalls, ndcgs, recommended_all = [], [], [], set()
    for user_id, relevant in relevant_by_user.items():
        user_train = train_by_user.get(user_id)
        if user_train is None:
            continue
        scores = np.asarray(model.score(movie_ids, user_id=user_id, user_ratings=user_train), dtype=float)
        seen = index.get_indexer(user_train["movieId"])
        scores[seen[seen >= 0]] = -np.inf
        scores[blocked] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        recs = list(movie_ids[top[np.argsort(-scores[top])]])
        recommended_all.update(recs)
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))

    return {
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "coverage": len(recommended_all) / len(movie_ids),
        "users_evaluated": len(precisions),
    }
