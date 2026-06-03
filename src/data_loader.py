import pandas as pd
from pathlib import Path

def load_data():

    project_root = Path(__file__).resolve().parent.parent

    file_path = project_root / "data" / "tourism_with_id.csv"

    df = pd.read_csv(file_path)

    df = df.drop(
        columns=["Unnamed: 11", "Unnamed: 12"],
        errors="ignore"
    )

    return df