"""Download, load and preprocess the MovieLens dataset."""
from __future__ import annotations

import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .config import DATASET_NAME, MOVIELENS_URL, RAW_DIR, RANDOM_STATE, TEST_SIZE


def download_movielens(force: bool = False) -> Path:
    """Download and extract MovieLens into data/raw. Returns the dataset folder."""
    target = RAW_DIR / DATASET_NAME
    if target.exists() and not force:
        return target

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / f"{DATASET_NAME}.zip"
    print(f"Downloading {MOVIELENS_URL} ...")
    # Use Mozilla's CA bundle (certifi): the Windows store may lack the root CA
    # that files.grouplens.org chains to. Certificate verification stays on.
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    with urllib.request.urlopen(MOVIELENS_URL, context=context) as resp, open(zip_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()
    print(f"Dataset extracted to {target}")
    return target


def load_raw_data() -> dict[str, pd.DataFrame]:
    """Return the raw MovieLens tables: movies, ratings, tags, links."""
    path = download_movielens()
    return {
        "movies": pd.read_csv(path / "movies.csv"),
        "ratings": pd.read_csv(path / "ratings.csv"),
        "tags": pd.read_csv(path / "tags.csv"),
        "links": pd.read_csv(path / "links.csv"),
    }


def preprocess_movies(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    tags: pd.DataFrame | None = None,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Clean the movie table and build the text 'soup' used by the content model."""
    movies = movies.copy()

    # "Toy Story (1995)" -> year 1995, clean_title "Toy Story"
    movies["year"] = pd.to_numeric(
        movies["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce"
    ).astype("Int64")
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)

    genres = movies["genres"].replace("(no genres listed)", "")
    movies["genres"] = genres.str.replace("|", ", ", regex=False)
    # "Sci-Fi" -> "scifi" so each genre stays a single token
    genre_tokens = genres.str.replace("-", "", regex=False).str.lower().str.replace("|", " ", regex=False)

    if tags is not None and len(tags):
        tag_text = (
            tags.assign(tag=tags["tag"].astype(str).str.lower().str.strip())
            .groupby("movieId")["tag"]
            .agg(lambda s: " ".join(s.unique()))
        )
        movies["tags"] = movies["movieId"].map(tag_text).fillna("")
    else:
        movies["tags"] = ""

    # Genres are repeated so they carry more weight than sparse user tags.
    movies["soup"] = (genre_tokens + " " + genre_tokens + " " + movies["tags"]).str.strip()

    stats = ratings.groupby("movieId")["rating"].agg(num_ratings="count", avg_rating="mean")
    movies = movies.merge(stats, on="movieId", how="left")
    movies["num_ratings"] = movies["num_ratings"].fillna(0).astype(int)
    movies["avg_rating"] = movies["avg_rating"].round(2)

    if links is not None:
        movies = movies.merge(links[["movieId", "imdbId"]], on="movieId", how="left")
        movies["imdb_url"] = movies["imdbId"].map(
            lambda x: f"https://www.imdb.com/title/tt{int(x):07d}/" if pd.notna(x) else None
        )
        movies = movies.drop(columns="imdbId")

    return movies.reset_index(drop=True)


def train_test_split_by_user(
    ratings: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a fraction of every user's ratings, so every test user also appears in train."""
    test = ratings.groupby("userId", group_keys=False).sample(frac=test_size, random_state=random_state)
    train = ratings.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)
