import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_model(df):

    tfidf = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = tfidf.fit_transform(
        df["content"]
    )

    cosine_sim = cosine_similarity(
        tfidf_matrix,
        tfidf_matrix
    )

    return cosine_sim


def get_recommendations(
    place_name,
    df,
    cosine_sim,
    top_n=5
):

    indices = pd.Series(
        df.index,
        index=df["Place_Name"]
    ).drop_duplicates()

    idx = indices[place_name]

    sim_scores = list(
        enumerate(cosine_sim[idx])
    )

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:top_n+1]

    place_indices = [
        i[0]
        for i in sim_scores
    ]

    return df.iloc[
        place_indices
    ][
        [
            "Place_Name",
            "Category",
            "City",
            "Rating"
        ]
    ]