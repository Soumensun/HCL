# data_loading.py

import pandas as pd

# Define the expected column names and dtypes for validation
EXPECTED_COLUMNS = [
    "Churn", "AccountWeeks", "ContractRenewal", "DataPlan", "DataUsage", 
    "CustServCalls", "DayMins", "DayCalls", "MonthlyCharge", "OverageFee", "RoamMins"
]
EXPECTED_TYPES = {
    "Churn": "int64",            # binary target (0 or 1)
    "AccountWeeks": "int64",     # numeric
    "ContractRenewal": "int64",  # binary (0 or 1)
    "DataPlan": "int64",         # binary (0 or 1)
    "DataUsage": "float64",      # numeric (GB of data used, for example)
    "CustServCalls": "int64",    # numeric (count of customer service calls)
    "DayMins": "float64",        # numeric (minutes of day-time calls)
    "DayCalls": "int64",         # numeric (number of day-time calls)
    "MonthlyCharge": "float64",  # numeric (monthly bill amount)
    "OverageFee": "float64",     # numeric (fee for exceeding plan limits)
    "RoamMins": "float64"        # numeric (roaming minutes)
}

def load_data(csv_path="telecom_churn.csv"):
    """Load the dataset and validate its schema."""
    df = pd.read_csv(csv_path)
    # Check that all expected columns are present
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")
    # (Optional) Check data types
    for col, expected_type in EXPECTED_TYPES.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if expected_type not in actual_type:
                print(f"Warning: Column {col} expected type {expected_type} but got {actual_type}")
    # Basic info and preview
    print("Data loaded successfully. Shape:", df.shape)
    print(df.info())           # Print schema info
    print(df.head(5))          # Print first 5 rows as sample
    return df

# If run as script, load the data
if __name__ == "__main__":
    df = load_data()
