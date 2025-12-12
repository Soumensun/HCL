# Telecom Customer Churn Prediction  
_Logistic Regression • Random Forest • XGBoost • Optuna • Stacking Ensemble • Streamlit App_

This project builds a complete Machine Learning system to predict **customer churn** for a telecom company.  
It includes **EDA → preprocessing → imbalance handling → model training → hyperparameter tuning → ensemble learning → evaluation → deployment via Streamlit**.

---

## 📌 1. Dataset Details (telecom_churn.csv)

### What is churn?
A customer is considered "churned" when they stop using the service.  
Goal: Predict whether a customer will churn (`1`) or stay (`0`).

### Dataset Size
- **Rows:** 3333  
- **Columns:** 11  

### Target Column: `Churn`
- `0` → Not churned  
- `1` → Churned  

### Class Imbalance
- Churn = 0 → 2850  
- Churn = 1 → 483  
⮕ Only **14.5% churners** → dataset is **imbalanced**.

---

## 📌 2. Why Logistic Regression (Not Linear Regression)
Logistic Regression models the probability:

\[
P(y=1|x) = \frac{1}{1 + e^{-(w^Tx + b)}}
\]

It is ideal for **binary classification**, unlike Linear Regression which predicts continuous values.

---

## 📌 3. ML Workflow (Pipeline)

### **Step 1 — Data Loading**
Handled by `data_loading.py`  
Ensures correct columns and clean structure.

---

### **Step 2 — Preprocessing**
Performed in `preprocessing.py`:
- Train/Test split (stratified)
- Scaling using `StandardScaler`
- Outlier & skewness checks

Scaling is crucial for algorithms like Logistic Regression.

---

### **Step 3 — Exploratory Data Analysis (EDA)**
In `eda.py`, we analyze:
- Feature distributions  
- Correlations  
- Outliers  
- Customer behavior patterns  

---

### **Step 4 — Handling Class Imbalance (SMOTE)**  
Why SMOTE?
- Prevents model from being biased toward majority class  
- Generates synthetic minority samples  
- Improves recall for churners  

---

## 📌 4. Ensemble Learning (Very Important)

To improve model robustness, we trained multiple models:

### **Base Models**
- Logistic Regression  
- Random Forest  
- XGBoost  

### **Why Ensemble?**
Each model captures different patterns:
- Logistic Regression → linear structure  
- Random Forest → non-linear interactions  
- XGBoost → gradient-boosted precision  

### **Voting Classifier**
- Combines predictions by averaging (soft vote)  
Useful to reduce variance.

### **Stacking Ensemble (Best Model)**
Uses:
- Level-0 models: `Logistic Regression + Random Forest + XGBoost`
- Meta-model: `Logistic Regression`

The meta-model **learns how much to trust each base model**, giving the stacking model superior performance.

---

## 📌 5. Hyperparameter Tuning (Optuna)

Optuna was used for:
- Searching best `max_depth`, `learning_rate`, `n_estimators`
- Pruning bad trials early  
- Achieving better performance than grid search  

### **Final Optuna-tuned XGBoost Results**

RMSE = 0.2873
MAE = 0.1380
R2 = 0.3360


---

## 📌 6. Baseline Model Performance

Logistic Regression: RMSE=0.3884, MAE=0.2926, R2=-0.2139
Random Forest: RMSE=0.2929, MAE=0.1759, R2=0.3099
XGBoost: RMSE=0.2982, MAE=0.1251, R2=0.2846
Lasso: RMSE=0.5000
Ridge: RMSE=0.4054


---

## 📌 7. Model Comparison (Visual Outputs)

### **Predictions vs Actuals**
![Prediction Comparison](images/All%20Model%20Prediction.png)

---

### **RMSE Comparison**
![RMSE Comparison](images/All%20Model%20Losses.png)

---

### **Confusion Matrix & Metrics**
![Confusion Matrix](images/Matrices.png)

---

## 📌 8. Streamlit App (Deployment Preview)

A simple UI for entering customer features and getting a churn prediction.

![Streamlit UI](images/Predict1.png)

---

## 📌 9. Project Structure
HCL/
├── data_loading.py
├── preprocessing.py
├── eda.py
├── training.py
├── evaluation.py
├── app.py
├── images/
│ ├── All_Model_Prediction.png
│ ├── All_Model_Losses.png
│ ├── Matrices.png
│ ├── Predict1.png
├── FINALHackathon_HCLTECH.ipynb
└── requirements.txt


---

## 📌 10. Open Notebook in Google Colab

Click below to view the full notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Soumensun/HCL/blob/main/FINALHackathon_HCLTECH.ipynb)

---

## 📌 11. Summary

✔ Addressed data imbalance with SMOTE  
✔ Performed comprehensive EDA  
✔ Tuned models using Optuna  
✔ Compared Logistic Regression, RF, XGB, Stacking  
✔ Best performance from **Stacking Ensemble**  
✔ Deployed model using **Streamlit**  

This project demonstrates expertise in:
- Machine Learning  
- Feature engineering  
- Imbalanced learning  
- Hyperparameter tuning  
- Ensemble methods  
- Real-world deployment  

---

