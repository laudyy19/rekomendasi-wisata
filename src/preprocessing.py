def preprocess_data(df):

    df = df.copy()

    df = df.drop_duplicates()

    df = df.fillna("")

    return df