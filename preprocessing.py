# preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        self.scaler = None
        self.numeric_cols = None
        self.cat_cols = None
    
    def fit(self, X: pd.DataFrame):
        """Fit preprocessing steps: identify columns, fit scaler on numeric features."""
        # Identify numeric and categorical feature columns
        self.numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Exclude target or any non-feature columns if present (ensure 'Churn' not included)
        if 'Churn' in self.numeric_cols:
            self.numeric_cols.remove('Churn')
        # Identify categorical (object type) columns, if any
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Initialize scaler for numeric features
        self.scaler = StandardScaler()
        # Fit scaler on numeric features
        if self.numeric_cols:
            self.scaler.fit(X[self.numeric_cols])
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the preprocessing (impute, encode, scale) to a dataset."""
        X_proc = X.copy()
        # Handle missing numeric values by imputing with median (if any)
        for col in X_proc.select_dtypes(include=['int64', 'float64']).columns:
            if X_proc[col].isnull().any():
                median_val = X_proc[col].median()
                X_proc[col].fillna(median_val, inplace=True)
        # Handle missing categorical by imputing with mode
        for col in X_proc.select_dtypes(include=['object', 'category']).columns:
            if X_proc[col].isnull().any():
                mode_val = X_proc[col].mode(dropna=True)
                if not mode_val.empty:
                    X_proc[col].fillna(mode_val[0], inplace=True)
        
        # One-hot encode categorical columns (if any exist in training data)
        if self.cat_cols:
            # Use pandas get_dummies for one-hot encoding
            X_proc = pd.get_dummies(X_proc, columns=self.cat_cols, drop_first=True)
            # Ensure any categorical dummy columns missing (because they weren't in this set) are added with 0
            for col in self.cat_cols:
                # For each category dummy from training, ensure exists in X_proc
                train_dummy_cols = [c for c in self.fitted_feature_names if c.startswith(col + "_")]
                for dummy in train_dummy_cols:
                    if dummy not in X_proc.columns:
                        X_proc[dummy] = 0
        # Scale numeric features
        if self.scaler and self.numeric_cols:
            X_proc[self.numeric_cols] = self.scaler.transform(X_proc[self.numeric_cols])
        return X_proc
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit the preprocessing and transform X in one go (used on training set)."""
        # Record original categorical dummy columns after fitting for consistency
        X_proc = X.copy()
        self.fit(X_proc)
        # After fitting scaler, we also prepare dummy columns list for categorical features
        if self.cat_cols:
            # One-hot encode on the training set
            X_proc = pd.get_dummies(X_proc, columns=self.cat_cols, drop_first=True)
            # Save the full list of feature names after encoding, to use for aligning test data
            self.fitted_feature_names = X_proc.columns.tolist()
            # Fit scaler on numeric columns of X_proc (since X_proc now has dummy columns too)
            if self.numeric_cols:
                self.scaler.fit(X_proc[self.numeric_cols])
            # Scale numeric features
            if self.numeric_cols:
                X_proc[self.numeric_cols] = self.scaler.transform(X_proc[self.numeric_cols])
        else:
            # No categorical features, just fit scaler on numeric and transform
            self.fitted_feature_names = X_proc.columns.tolist()
            if self.numeric_cols:
                self.scaler.fit(X_proc[self.numeric_cols])
                X_proc[self.numeric_cols] = self.scaler.transform(X_proc[self.numeric_cols])
        return X_proc
