def create_content_feature(df):

    df = df.copy()

    df["content"] = (
        df["Category"].astype(str)
        + " "
        + df["City"].astype(str)
        + " "
        + df["Description"].astype(str)
    )

    return df