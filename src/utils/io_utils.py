import os
import pandas as pd


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path) or '.')
    df.to_csv(path, index=False)
