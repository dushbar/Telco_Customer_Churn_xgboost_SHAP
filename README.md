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
7. [Acknowledgements](#acknowledgements)  

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
  1. data_cleaning_eda.ipynb
  2. xgboost_modeling.ipynb
  3. model_interpretation_shap.ipynb

## Methodology

| Stage | Key Actions |
|-------|-------------|
| **EDA** | Univariate & bivariate plots|
| **Pre-processing** | Binary columns → label encoding; multi-class categoricals → one-hot (dropping first level); no scaling was done. |
| **Baseline model** | `XGBClassifier` (1000 trees, default params, `eval_metric='auc'`). |
| **Hyper-parameter tuning** | Randomized search over (gamma, learning_rate, max_depth, subsample). |
| **Evaluation metric** | ROC AUC |
| **Interpretability** | Use `shap.TreeExplainer` to generate global (summary plot) and local plot for feature with highest attribution |

## Results

| Model | ROC AUC |
|-------|--------:|
| **Baseline XGBoost** | **0.82758** |
| **Tuned XGBoost** | **0.856549** |


## Key SHAP Insights

- `tenure, InternetService_Fiber optic, Contract_Two year, PaymentMethod_Electronic check` are the top 4 features by attribution.
- These 4 features are also the top 4 features by F statistic that we found by applying ANOVA F-Test
- Visual explanations are in `model_interpretation_shap.ipynb`.

## Acknowledgements

- **[Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)**
- Guidance from *Data Science Projects with Python: A case study approach* by Stephen Klosterman
