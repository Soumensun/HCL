"""data_loading.py
Simple helpers to load dataset used in HCL hackathon project.
Usage:
    from data_loading import load_data
    df = load_data('data/telecom_churn.csv')
"""
import pandas as pd
from typing import Tuple

def load_data(path: str) -> pd.DataFrame:
    """Load CSV file into a pandas DataFrame."""
    df = pd.read_csv(path)
    return df

def preview(df, n=5):
    print(df.head(n))
