"""Train, evaluate and save the movie recommenders.

    python train.py                 # evaluate, tune hybrid alpha, train on all data, save
    python train.py --skip-eval     # just train on all data and save
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from src.config import (HYBRID_ALPHA, HYBRID_BETA, METRICS_PATH, MF_PARAMS, MIN_RATINGS,
                        REPORTS_DIR, TOP_K)
from src.data_loader import load_raw_data, preprocess_movies, train_test_split_by_user
from src.evaluate import evaluate_ranking, mae, rmse
from src.models import HybridRecommender
from src.pipeline import build_models, train_full_and_save

ALPHA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
BETA_GRID = [0.0, 0.1, 0.2, 0.3, 0.5]


def tune_hybrid(train: pd.DataFrame, movies: pd.DataFrame, mf_overrides: dict) -> tuple[float, float]:
    """Pick alpha, then beta, by NDCG on a validation split carved out of train."""
    inner, val = train_test_split_by_user(train, test_size=0.2, random_state=7)
    m = build_models(inner, movies, **mf_overrides)
    ids = movies["movieId"]

    def val_ndcg(alpha, beta):
        h = HybridRecommender(m["ease"], m["content"], m["popularity"], alpha, beta)
        return evaluate_ranking(h, inner, val, ids, min_ratings=MIN_RATINGS)[f"ndcg@{TOP_K}"]

    def search(candidates, make_params):
        scores = {}
        for c in candidates:
            alpha, beta = make_params(c)
            scores[c] = val_ndcg(alpha, beta)
            print(f"  alpha={alpha:.1f} beta={beta:.1f}  val NDCG@{TOP_K}={scores[c]:.4f}")
        return max(scores, key=scores.get)

    best_alpha = search(ALPHA_GRID, lambda a: (a, 0.0))            # 1) personal blend
    best_beta = search(BETA_GRID, lambda b: (best_alpha, b))      # 2) popularity prior
    print(f"  -> best alpha = {best_alpha}, beta = {best_beta}")
    return best_alpha, best_beta


def evaluate(mf_overrides: dict) -> float:
    raw = load_raw_data()
    ratings = raw["ratings"]
    movies = preprocess_movies(raw["movies"], ratings, raw["tags"], raw["links"])
    print(f"Ratings: {len(ratings):,} | users: {ratings.userId.nunique():,} | "
          f"movies: {len(movies):,} | sparsity: "
          f"{1 - len(ratings) / (ratings.userId.nunique() * len(movies)):.2%}")

    train, test = train_test_split_by_user(ratings)
    print(f"Train: {len(train):,}  Test: {len(test):,}\n")

    print("Tuning hybrid weights on validation split...")
    alpha, beta = tune_hybrid(train, movies, mf_overrides)

    print("\nTraining models on train split...")
    t0 = time.time()
    models = build_models(train, movies, alpha=alpha, beta=beta, **mf_overrides)
    print(f"  done in {time.time() - t0:.1f}s")

    mf = models["collaborative"]
    results = {"rating_prediction": {}, "ranking": {}, "hybrid_alpha": alpha,
               "hybrid_beta": beta, "min_ratings_filter": MIN_RATINGS}
    global_mean = train["rating"].mean()
    results["rating_prediction"]["global_mean_baseline"] = {
        "rmse": rmse(test["rating"], [global_mean] * len(test)),
        "mae": mae(test["rating"], [global_mean] * len(test)),
    }
    preds = mf.predict(test["userId"], test["movieId"])
    results["rating_prediction"]["matrix_factorization"] = {
        "rmse": rmse(test["rating"], preds), "mae": mae(test["rating"], preds)}

    for name in ["popularity", "content", "collaborative", "ease", "hybrid"]:
        results["ranking"][name] = evaluate_ranking(models[name], train, test, movies["movieId"],
                                                    min_ratings=MIN_RATINGS)

    print("\n=== Rating prediction (lower is better) ===")
    print(pd.DataFrame(results["rating_prediction"]).T.round(4).to_string())
    print(f"\n=== Top-{TOP_K} ranking, movies with >= {MIN_RATINGS} ratings (higher is better) ===")
    print(pd.DataFrame(results["ranking"]).T.drop(columns="users_evaluated").round(4).to_string())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nMetrics saved to {METRICS_PATH}")
    return alpha, beta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-eval", action="store_true", help="skip evaluation and alpha tuning")
    p.add_argument("--alpha", type=float, default=None, help="force hybrid CF weight (0-1)")
    p.add_argument("--beta", type=float, default=None, help="force hybrid popularity weight (0-1)")
    p.add_argument("--factors", type=int, default=MF_PARAMS["n_factors"])
    p.add_argument("--epochs", type=int, default=MF_PARAMS["n_epochs"])
    p.add_argument("--lr", type=float, default=MF_PARAMS["lr"])
    p.add_argument("--reg", type=float, default=MF_PARAMS["reg"])
    args = p.parse_args()
    mf_overrides = dict(n_factors=args.factors, n_epochs=args.epochs, lr=args.lr, reg=args.reg)

    alpha, beta = (HYBRID_ALPHA, HYBRID_BETA) if args.skip_eval else evaluate(mf_overrides)
    alpha = alpha if args.alpha is None else args.alpha
    beta = beta if args.beta is None else args.beta

    print(f"\nTraining final models on the full dataset (alpha={alpha}, beta={beta})...")
    train_full_and_save(alpha=alpha, beta=beta, verbose=False, **mf_overrides)


if __name__ == "__main__":
    main()
