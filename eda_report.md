# Exploratory Data Analysis (EDA) Report

**Dataset:** IBM Telco Customer Churn  
**Total Records:** 7,043  
**Total Features:** 21 (20 predictors + 1 target variable `Churn`)  

---

## 📌 Executive Summary of Findings

1. **Target Distribution (`Churn`)**:
   - **No (Retained):** 5,174 customers (**73.46%**)
   - **Yes (Churned):** 1,869 customers (**26.54%**)
   - *Insight:* Significant class imbalance (~3:1 ratio). We must use appropriate evaluation metrics (Precision, Recall, F1-Score, ROC-AUC) and class weighting or resampling (SMOTE) during model training.

2. **Numerical Feature Insights**:
   - **Tenure:**
     - Non-churned average tenure: **37.5 months** (median: 38.0).
     - Churned average tenure: **17.9 months** (median: 10.0).
     - *Insight:* New customers (tenure < 12 months) have the highest risk of churn. Churn rate drops significantly after 2 years of tenure.
   - **Monthly Charges:**
     - Non-churned mean: **$61.27**
     - Churned mean: **$74.44**
     - *Insight:* Higher monthly bills directly correlate with higher churn probability.
   - **Total Charges:**
     - 11 blank space strings (`' '`) identified in new customers with `tenure = 0`. These will be imputed with 0.0 or median during preprocessing.

3. **Key Categorical Drivers of Churn**:
   - **Contract Type (Highest Impact Feature)**:
     - Month-to-month: **42.71% churn rate** (Very high risk)
     - One year: **11.27% churn rate**
     - Two year: **2.83% churn rate** (Extremely loyal)
   - **Internet Service**:
     - Fiber Optic: **41.89% churn rate** (High churn due to price sensitivity or competition)
     - DSL: **18.96% churn rate**
     - No Internet: **7.40% churn rate**
   - **Security & Support Add-ons**:
     - No Tech Support: **41.64% churn rate** (vs 15.17% with Tech Support)
     - No Online Security: **41.77% churn rate** (vs 14.61% with Online Security)
   - **Payment Method**:
     - Electronic Check: **45.29% churn rate** (Significantly higher than Automatic Bank Transfer/Credit Card at ~16-19%).

---

## 🖼️ Generated Plots (Saved in `eda_plots/`)
- `churn_distribution.png`: Bar chart of overall churn count.
- `tenure_distribution.png`: Density plot showing high churn among early tenure customers.
- `MonthlyCharges_distribution.png`: Density plot comparing spending habits.
- `categorical_churn_rates.png`: Comparison of churn rates across Contract, Internet Service, Payment Method, Tech Support, etc.

---

## 🎯 Recommended Next Steps for Preprocessing (Step 2)
1. Drop `customerID` (non-predictive identifier).
2. Clean `TotalCharges` (convert blank strings to `0.0`).
3. Encode target variable `Churn` (`Yes` -> 1, `No` -> 0).
4. One-Hot Encode binary and multi-class categorical features.
5. Scale numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler`.
6. Split data into 80% Train and 20% Test sets.
