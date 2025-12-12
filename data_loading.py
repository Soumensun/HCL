# data_loading.py
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load CSV into DataFrame."""
    df = pd.read_csv(path)
    return df

def preview(df, n=5):
    """Quick preview helper."""
    print(df.head(n))
