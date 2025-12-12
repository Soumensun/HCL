"""eda.py
Basic exploratory data analysis functions: summary, distributions and correlation.
Saves simple plots to outputs/ for inclusion in reports.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def summary(df: pd.DataFrame):
    print('\n--- Dataset info ---\n')
    print(df.info())
    print('\n--- Missing values per column ---\n')
    print(df.isnull().sum())
    print('\n--- Basic statistics ---\n')
    print(df.describe(include='all').T)

def plot_numeric_distributions(df: pd.DataFrame, outdir='outputs'):
    os.makedirs(outdir, exist_ok=True)
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        plt.hist(df[col].dropna(), bins=30)
        plt.title(f'Distribution: {col}')
        plt.xlabel(col); plt.ylabel('count')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'dist_{col}.png'))
        plt.close()

def plot_correlation(df: pd.DataFrame, outdir='outputs'):
    os.makedirs(outdir, exist_ok=True)
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if len(numeric_cols) < 2:
        return
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(8,6))
    plt.imshow(corr, cmap='RdYlBu', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.title('Correlation matrix (numeric features)')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'correlation_matrix.png'))
    plt.close()
