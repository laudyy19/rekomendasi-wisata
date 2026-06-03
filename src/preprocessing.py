def preprocess_data(df):

    df = df.copy()

    text_cols = [
        "Place_Name",
        "Description",
        "Category",
        "City"
    ]

    for col in text_cols:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.lower()
        )

    return df