"""Command-line recommendations.

    python recommend.py --movie "toy story"      # movies similar to a title
    python recommend.py --user 1                 # personalised picks for a MovieLens user
    python recommend.py --top                    # most popular (cold start)
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.models import top_n
from src.pipeline import load_or_train


def find_movie(movies: pd.DataFrame, query: str) -> pd.Series | None:
    matches = movies[movies["title"].str.contains(query, case=False, regex=False)]
    if matches.empty:
        return None
    return matches.sort_values("num_ratings", ascending=False).iloc[0]


def show(movies: pd.DataFrame, recs: pd.Series, title: str) -> None:
    table = movies.set_index("movieId").loc[recs.index, ["title", "genres", "avg_rating"]]
    table["score"] = recs.round(3).values
    print(f"\n{title}\n" + table.reset_index(drop=True).to_string(index=True))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--movie", help="title (or part of it)")
    g.add_argument("--user", type=int, help="MovieLens userId")
    g.add_argument("--top", action="store_true", help="popular movies")
    p.add_argument("-n", type=int, default=10)
    args = p.parse_args()

    a = load_or_train()
    movies, ratings = a["movies"], a["ratings"]
    ids = movies["movieId"].to_numpy()
    popular_enough = (movies["num_ratings"] >= 10).to_numpy()

    if args.movie:
        movie = find_movie(movies, args.movie)
        if movie is None:
            sys.exit(f"No movie matching '{args.movie}'")
        show(movies, a["content"].similar_movies(movie.movieId, args.n),
             f"Similar content to {movie.title}:")
        show(movies, a["ease"].similar_movies(movie.movieId, args.n),
             f"Fans of {movie.title} also liked:")
    elif args.user is not None:
        user_ratings = ratings[ratings["userId"] == args.user]
        if user_ratings.empty:
            sys.exit(f"User {args.user} not found")
        recs = top_n(a["hybrid"], ids, args.n, user_id=args.user, user_ratings=user_ratings,
                     exclude=set(user_ratings["movieId"]), allowed_mask=popular_enough)
        show(movies, recs, f"Recommendations for user {args.user}:")
    else:
        show(movies, top_n(a["popularity"], ids, args.n), "Top rated movies:")


if __name__ == "__main__":
    main()
