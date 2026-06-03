from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


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

    indices = pd.Series(
        df.index,
        index=df["Place_Name"]
    ).drop_duplicates()

    return cosine_sim, indices


def get_recommendations(
    place_name,
    cosine_sim,
    indices,
    df,
    top_n=5
):

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

    result = df.iloc[
        place_indices
    ].copy()

    result["Similarity"] = [
        i[1]
        for i in sim_scores
    ]

    return result