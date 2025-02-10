"""Streamlit web app.  Run with:  streamlit run app.py"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import METRICS_PATH
from src.models import top_n
from src.pipeline import load_or_train

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run downloads data and trains, ~1 min)...")
def get_artifacts():
    return load_or_train()


a = get_artifacts()
movies: pd.DataFrame = a["movies"]
ratings: pd.DataFrame = a["ratings"]
movie_ids = movies["movieId"].to_numpy()
by_id = movies.set_index("movieId")
titles = movies.sort_values("num_ratings", ascending=False)
title_to_id = dict(zip(titles["title"], titles["movieId"]))


def render(recs: pd.Series, score_label: str = "Score"):
    if recs.empty:
        st.info("No recommendations available.")
        return
    df = by_id.loc[recs.index, ["title", "genres", "year", "avg_rating", "num_ratings", "imdb_url"]].copy()
    df.insert(0, score_label, recs.values)
    st.dataframe(
        df.reset_index(drop=True),
        hide_index=True,
        width="stretch",
        column_config={
            score_label: st.column_config.ProgressColumn(score_label, min_value=0.0,
                                                         max_value=float(max(recs.max(), 1e-9)), format="%.2f"),
            "title": "Title", "genres": "Genres",
            "year": st.column_config.NumberColumn("Year", format="%d"),
            "avg_rating": st.column_config.NumberColumn("Avg ★", format="%.2f"),
            "num_ratings": "# Ratings",
            "imdb_url": st.column_config.LinkColumn("IMDb", display_text="Open"),
        },
    )


# ---------------- Sidebar ----------------
st.sidebar.title("🎬 Movie Recommender")
n = st.sidebar.slider("Number of recommendations", 5, 30, 10)
min_votes = st.sidebar.slider("Minimum #ratings per movie", 0, 100, 10,
                              help="Hide obscure movies with very few ratings")
alpha = st.sidebar.slider("Hybrid weight: collaborative ↔ content", 0.0, 1.0,
                          float(a["hybrid"].alpha), 0.05,
                          help="1.0 = pure collaborative filtering, 0.0 = pure content-based")
beta = st.sidebar.slider("Popularity boost", 0.0, 1.0, float(a["hybrid"].beta), 0.05,
                         help="Blend in overall popularity; 0 = fully personalised")
a["hybrid"].alpha, a["hybrid"].beta = alpha, beta
allowed = (movies["num_ratings"] >= min_votes).to_numpy()

if METRICS_PATH.exists():
    with st.sidebar.expander("Model evaluation"):
        m = json.loads(METRICS_PATH.read_text())
        st.caption("Rating prediction")
        st.dataframe(pd.DataFrame(m["rating_prediction"]).T.round(4))
        st.caption("Top-K ranking")
        st.dataframe(pd.DataFrame(m["ranking"]).T.drop(columns="users_evaluated").round(4))

st.title("Movie Recommendation System")
st.caption(f"MovieLens · {len(movies):,} movies · {len(ratings):,} ratings · "
           f"{ratings.userId.nunique():,} users")

tab_similar, tab_rate, tab_user, tab_top = st.tabs(
    ["🔍 Similar movies", "⭐ Rate & get picks", "👤 Existing user", "🏆 Top rated"])

with tab_similar:
    title = st.selectbox("Pick a movie you like", list(title_to_id), index=0)
    mid = title_to_id[title]
    st.write(f"**{title}** — {by_id.at[mid, 'genres']}")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Similar content")
        st.caption("Content-based: genres + user tags (TF-IDF cosine)")
        render(a["content"].similar_movies(mid, n), "Similarity")
    with c2:
        st.subheader("Fans also liked")
        st.caption("Collaborative: EASE item-item weights (watched together)")
        recs = a["ease"].similar_movies(mid, n * 3)
        recs = recs[by_id.loc[recs.index, "num_ratings"].to_numpy() >= min_votes].head(n)
        render(recs, "Strength")

with tab_rate:
    st.write("Rate a few movies and get personalised recommendations — "
             "new users are folded into the trained model on the fly.")
    picked = st.multiselect("Movies you've seen", list(title_to_id), max_selections=30)
    user_rows = []
    cols = st.columns(3)
    for k, t in enumerate(picked):
        with cols[k % 3]:
            user_rows.append({"movieId": title_to_id[t],
                              "rating": st.slider(t, 0.5, 5.0, 4.0, 0.5, key=f"r_{t}")})
    if st.button("Recommend", type="primary", disabled=not picked):
        ur = pd.DataFrame(user_rows)
        render(top_n(a["hybrid"], movie_ids, n, user_ratings=ur,
                     exclude=set(ur["movieId"]), allowed_mask=allowed), "Match")

with tab_user:
    uid = st.selectbox("MovieLens user ID", sorted(ratings["userId"].unique()))
    ur = ratings[ratings["userId"] == uid]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"User {uid}'s favourites")
        fav = ur.sort_values("rating", ascending=False).head(n)
        render(pd.Series(fav["rating"].values, index=fav["movieId"].values), "Their rating")
    with c2:
        st.subheader("Recommended for them")
        render(top_n(a["hybrid"], movie_ids, n, user_id=uid, user_ratings=ur,
                     exclude=set(ur["movieId"]), allowed_mask=allowed), "Match")

with tab_top:
    st.write("Bayesian-weighted average rating (used for cold-start users).")
    render(top_n(a["popularity"], movie_ids, n, allowed_mask=allowed), "Weighted ★")
