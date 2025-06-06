# Telco Customer Churn — XGBoost & SHAP

An end-to-end, **explainable machine-learning** project that predicts telecom customer churn with **XGBoost** and interprets the model with **SHAP**.  
The goal is to build a high-performance classifier *and* understand **why** it makes each prediction.

---

## Table of Contents
1. [Project Motivation](#project-motivation)  
2. [Dataset](#dataset)  
3. [Project Structure](#project-structure)  
4. [Methodology](#methodology)  
5. [Results](#results)  
6. [Key SHAP Insights](#key-shap-insights)  
7. [How to Reproduce](#how-to-reproduce)  
8. [Next Steps](#next-steps)  
9. [License](#license)  
10. [Acknowledgements](#acknowledgements)  

---

## Project Motivation
Combine a strong gradient-boosting model with SHAP explanations to balance predictive power and interpretability.

---

## Dataset
| Source | Size | Features | Target |
|--------|------|----------|--------|
| Original Kaggle dataset: **`WA_Fn-UseC_-Telco-Customer-Churn.csv`** | 7,043 rows × 21 columns | Demographics, account & service details | `Churn` (Yes/No) |

**Cleaning highlights**

| Step | Details | Δ Rows |
|------|---------|-------|
| Convert `TotalCharges` to numeric | Empty strings → `NaN`, 11 rows dropped | 7,043 → 7,032 |
| Drop exact duplicates | 22 rows removed | 7,032 → 7,010 |
| Final dataset | No missing values; 20 predictors + target | **7,010 rows** |

> See `data_cleaning_eda.ipynb` for full code & visuals.

---

## Project Structure
XGBoost_SHAP_Telco_Churn
- data
  1. WA_Fn-UseC_-Telco-Customer-Churn.csv # raw
  2. Telco_Customer_Churn_clean.csv # cleaned
- notebooks
  1. data_cleaning_and_eda.ipynb
  2. xgboost_modeling.ipynb
  3. model_interpretation_shap.ipynb

## Methodology

| Stage | Key Actions |
|-------|-------------|
| **EDA** | Univariate & bivariate plots; target imbalance check (~26 % churn). |
| **Pre-processing** | Binary columns → label encoding; multi-class categoricals → one-hot (dropping first level); numeric scaling kept native for XGBoost. |
| **Baseline model** | `XGBClassifier` (1000 trees, default params, `eval_metric='auc'`). |
| **Hyper-parameter tuning** | Randomized search over (gamma, learning_rate, max_depth, subsample). |
| **Evaluation metric** | ROC AUC |
| **Interpretability** | Use `shap.TreeExplainer` to generate global (summary plot) and local plot for feature with highest attribution |
