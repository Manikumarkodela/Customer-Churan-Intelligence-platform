# Customer Churn Prediction Project Roadmap

This document details the step-by-step workflow to build the Customer Churn Prediction ML model and Streamlit web application. We will complete this project together step-by-step.

---

## 📋 Task Checklist

### Step 1: Data Inspection & Exploratory Data Analysis (EDA)
- [x] Inspect dataset structure (`Telco-Customer-Churn.csv`), column data types, missing values, and summary statistics.
- [x] Conduct univariate and bivariate analysis on numerical features (e.g., Tenure, MonthlyCharges, TotalCharges) vs. Churn.
- [x] Analyze categorical features (e.g., Contract type, InternetService, PaymentMethod) vs. Churn rates.
- [x] Document key insights & observations for feature selection (`eda.py` & `eda_report.md`).

### Step 2: Data Preprocessing & Feature Engineering
- [x] Clean dataset (convert `TotalCharges` to numeric, handle missing values/blank spaces).
- [x] Handle target variable encoding (`Churn`: Yes/No → 1/0).
- [x] Encode categorical variables (One-Hot Encoding).
- [x] Perform Feature Scaling (`StandardScaler`) on numerical variables.
- [x] Split dataset into Training and Testing sets (80% Train, 20% Test, Stratified).
- [x] Save preprocessor pipeline (`preprocessor.joblib`) and feature metadata (`feature_meta.json`).

### Step 3: Model Training, Hyperparameter Tuning & Evaluation
- [x] Train candidate models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost).
- [x] Handle class imbalance via weighted loss / scale_pos_weight tuning.
- [x] Evaluate models using metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC curve.
- [x] Select best performing model (XGBoost Weighted) and generate visual confusion matrix & ROC plots.

### Step 4: Model & Artifact Serialization
- [x] Save trained model pipeline (`artifacts/model.joblib`).
- [x] Save model metrics summary (`artifacts/model_metrics.json`).
- [x] Save scaler and encoding metadata/transformers for consistent inference.

### Step 5: Streamlit Web Application Development
- [x] Set up Streamlit layout with custom white/off-white high-tech AI theme and glassmorphic cards.
- [x] Build **Single Customer Prediction Form** (interactive input sliders, select boxes, real-time risk gauge, JSON export).
- [x] Build **Batch Prediction Feature** (upload CSV, generate bulk predictions & downloadable CSV).
- [x] Build **Model Performance & EDA Dashboard** tab (visualizing key churn drivers, confusion matrix & ROC curve).

### Step 6: Testing, Refinement & Final Review
- [x] Test application with real sample customer inputs and verify preprocessor + model inference.
- [x] Finalize code quality, inline comments, and project documentation (`tasks.md`, `eda_report.md`, `model_evaluation_report.md`).
