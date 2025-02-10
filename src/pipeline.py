"""Build, save and load the full set of trained recommenders."""
from __future__ import annotations

import joblib
import pandas as pd

from .config import EASE_PARAMS, HYBRID_ALPHA, HYBRID_BETA, MODEL_PATH, MODELS_DIR
from .data_loader import load_raw_data, preprocess_movies
from .models import (ContentBasedRecommender, EASERecommender, HybridRecommender,
                     MatrixFactorization, PopularityRecommender)


def build_models(ratings: pd.DataFrame, movies: pd.DataFrame, alpha: float = HYBRID_ALPHA,
                 beta: float = HYBRID_BETA, verbose: bool = False, **mf_overrides) -> dict:
    popularity = PopularityRecommender().fit(ratings)
    content = ContentBasedRecommender().fit(movies)
    collaborative = MatrixFactorization.default(verbose=verbose, **mf_overrides).fit(ratings)
    ease = EASERecommender(**EASE_PARAMS).fit(ratings)
    hybrid = HybridRecommender(ease, content, popularity, alpha=alpha, beta=beta)
    return {"popularity": popularity, "content": content, "collaborative": collaborative,
            "ease": ease, "hybrid": hybrid}


def train_full_and_save(alpha: float = HYBRID_ALPHA, beta: float = HYBRID_BETA,
                        verbose: bool = True, **mf_overrides) -> dict:
    """Train every model on the full dataset and save one artifact bundle."""
    raw = load_raw_data()
    movies = preprocess_movies(raw["movies"], raw["ratings"], raw["tags"], raw["links"])
    models = build_models(raw["ratings"], movies, alpha=alpha, beta=beta, verbose=verbose,
                          **mf_overrides)
    artifacts = {
        "movies": movies.drop(columns=["soup"]),
        "ratings": raw["ratings"][["userId", "movieId", "rating"]],
        **models,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, MODEL_PATH, compress=3)
    print(f"Saved models to {MODEL_PATH}")
    return artifacts


def load_artifacts() -> dict:
    return joblib.load(MODEL_PATH)


def load_or_train() -> dict:
    return load_artifacts() if MODEL_PATH.exists() else train_full_and_save()
