# Telecom Customer Churn Prediction (Logistic Regression + Streamlit)

This project builds an end-to-end Machine Learning pipeline to predict **customer churn** (binary classification) using **Logistic Regression**, including preprocessing, EDA, class-imbalance handling, hyperparameter tuning, evaluation, and deployment using **Streamlit**.

---

## 1. Dataset Details (telecom_churn.csv)

### What is churn?
**Churn** means a customer is likely to stop using the telecom service.  
The goal is to predict whether a customer will churn (`1`) or not (`0`).

### Dataset Size
- Rows: **3333**
- Columns: **11**

### Target Column
- `Churn` (binary)
  - `0` = Not churned
  - `1` = Churned

### Class Distribution (Imbalance)
- `Churn = 0`: 2850 samples
- `Churn = 1`: 483 samples  
So churn rate is about **14.5%**, which is **imbalanced**.  
This matters because a naive model can predict "no churn" for everyone and still get high accuracy.

### Feature Columns and Meaning
| Column | Type | Meaning |
|--------|------|---------|
| AccountWeeks | int | How many weeks the customer has been with the service |
| ContractRenewal | int (0/1) | Whether the customer recently renewed contract |
| DataPlan | int (0/1) | Whether the customer has a data plan |
| DataUsage | float | Amount of data used (e.g., GB/month) |
| CustServCalls | int | Number of customer service calls |
| DayMins | float | Daytime minutes used |
| DayCalls | int | Daytime calls made |
| MonthlyCharge | float | Monthly bill amount |
| OverageFee | float | Extra fee due to exceeding plan limits |
| RoamMins | float | Roaming minutes used |

**No missing values** were found in this dataset.

---

## 2. Why Logistic Regression (Not Linear Regression)

- **Linear Regression** is for continuous numeric prediction (regression).
- **Logistic Regression** is designed for **binary classification**, modeling:

\[
P(y=1|x) = \sigma(w^Tx + b) = \frac{1}{1 + e^{-(w^Tx + b)}}
\]

It outputs probabilities and supports decision thresholds (default 0.5), making it correct for churn prediction.

---

## 3. Workflow (Pipeline)

The pipeline follows a standard ML workflow:

### Step 1: Data Collection
- Load `telecom_churn.csv`
- Validate required columns exist

File: `data_loading.py`

---

### Step 2: Data Preprocessing
- Separate features `X` and target `y`
- Train/test split (stratified)
- Scale numeric features using `StandardScaler`

Why scaling?
Logistic regression depends on gradient-based optimization; features with very different scales can dominate the optimization and distort coefficients.

File: `preprocessing.py`

---

### Step 3: Exploratory Data Analysis (EDA)
- Distribution plots for numeric columns
- Correlation heatmap
- Skewness + outlier inspection (IQR-based)

EDA helps identify:
- highly correlated features
- skewed distributions
- extreme values/outliers that may affect learning

File: `eda.py`

---

### Step 4: Model Training (Logistic Regression)
- Train Logistic Regression
- Uses regularization (L1/L2 based on tuning)

File: `training.py`

---

### Step 5: Handling Class Imbalance (SMOTE)

Because churn rate is ~14.5%, the dataset is imbalanced.

We use **SMOTE** on the **training set only**:
- It synthetically creates minority-class samples instead of just duplicating them.
- Often improves recall for minority class (churn=1).

Why SMOTE over only class-weighting?
- **Class weights** change penalty but do not add new minority patterns.
- **SMOTE** expands the minority region in feature space, helping the decision boundary learn minority structure better.
- In churn problems, recall is usually important (catch churners early), and SMOTE often helps that.

(We still keep evaluation on the untouched test set to remain honest.)

File: `training.py`

---

### Step 6: Hyperparameter Tuning (GridSearchCV)
We tune:
- regularization strength `C`
- solver
- penalty type (L1 vs L2)

Scoring used: **ROC-AUC** (better for imbalanced classification than raw accuracy)

File: `training.py`

---

### Step 7: Model Evaluation
Metrics included:
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

These metrics matter because accuracy alone is misleading on imbalanced datasets.

File: `evaluation.py`

---

### Step 8: Deployment (Streamlit)
A Streamlit web app:
- takes feature inputs from a user
- applies scaling
- outputs churn probability and prediction

File: `app.py`

---

## 4. Project Structure

```bash
customer_churn_prediction/
├── telecom_churn.csv
├── data_loading.py
├── preprocessing.py
├── eda.py
├── training.py
├── evaluation.py
├── app.py
├── requirements.txt


## 5.Open Notebook in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Soumensun/HCL/blob/main/FINALHackathon_HCLTECH.ipynb)

## Outputs
![Prediction Comparison](images/All%20Model%20Prediction.png)

