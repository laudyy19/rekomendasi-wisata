def create_content_feature(df):

    df = df.copy()

    df["content"] = (
        df["Category"] + " " +
        df["City"] + " " +
        df["Description"]
    )

    return df