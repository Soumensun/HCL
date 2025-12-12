"""preprocessing.py
Functions for cleaning, encoding and scaling data.
Exports: preprocess(df, target_column)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_preprocessing_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return preprocessor

def preprocess(df: pd.DataFrame, target_column: str, test_size=0.2, random_state=42):
    """Return X_train, X_test, y_train, y_test and fitted pipeline."""
    df = df.copy()
    # Basic drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Identify features
    y = df[target_column]
    X = df.drop(columns=[target_column])

    numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object','category']).columns.tolist()

    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if len(y.unique())>1 else None
    )

    # Fit transform training, transform test
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return X_train_proc, X_test_proc, y_train.values, y_test.values, preprocessor, numeric_features, categorical_features
