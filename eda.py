# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(df: pd.DataFrame, output_dir="plots"):
    """Plot histograms for each numeric feature to inspect distributions."""
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    for col in numeric_cols:
        if col == 'Churn':
            continue  # skip target
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)  # histogram with density curve
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.savefig(f"{output_dir}/dist_{col}.png")
        plt.close()
        
def plot_correlation_heatmap(df: pd.DataFrame, output_path="plots/correlation_heatmap.png"):
    """Plot a heatmap of the correlation matrix for numeric features."""
    numeric_df = df.select_dtypes(include=['int64','float64']).copy()
    corr = numeric_df.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig(output_path)
    plt.close()

def detect_skew_outliers(df: pd.DataFrame):
    """Compute skewness of numeric features and detect outliers using IQR rule."""
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    skew_info = {}
    outlier_info = {}
    for col in numeric_cols:
        if col == 'Churn':
            continue
        # Skewness
        skew_val = df[col].skew()
        skew_info[col] = round(skew_val, 2)
        # Outliers via IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_count = outliers.count()
        if outlier_count > 0:
            outlier_info[col] = outlier_count
    # Print skewness results
    print("Skewness of features:", skew_info)
    # Print outlier counts
    if outlier_info:
        print("Detected outliers (count) in features:", outlier_info)
    else:
        print("No significant outliers detected by IQR rule.")
    return skew_info, outlier_info

# If run as script, perform EDA on the dataset
if __name__ == "__main__":
    from data_loading import load_data
    df = load_data()  # load dataset
    plot_distributions(df)
    plot_correlation_heatmap(df)
    detect_skew_outliers(df)
