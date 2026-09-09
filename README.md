# ⚡ Telco AI — Customer Churn Intelligence & Prediction Platform

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-22C55E?style=flat)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, high-performance Machine Learning platform and interactive Web Application designed to predict customer churn in telecommunication accounts. Built with **XGBoost**, **Scikit-Learn**, and **Streamlit**, featuring a futuristic white/violet high-tech user interface for both single customer telemetry forecasting and bulk batch CSV processing.

---

## 📸 Key Features

* **🔮 Single Customer Telemetry Predictor**: Interactive input form with real-time churn probability calculation, risk indicators (Low, Medium, High), and downloadable JSON telemetry reports.
* **📁 Bulk CSV Batch Predictor**: Upload raw customer datasets to generate bulk predictions with downloadable output CSV files.
* **📊 Interactive Analytics Dashboard**: Embedded model performance visualizations (Confusion Matrix, ROC Curves, Top 10 Feature Importances) and Exploratory Data Analysis (EDA) distributions.
* **⚖️ Imbalanced Class Handling**: Uses cost-sensitive reweighting (`scale_pos_weight`) to optimize **Recall (78.61%)** and catch at-risk customers before they leave.

---

## 📁 Repository Structure

```text
├── Telco-Customer-Churn.csv   # Raw IBM Telco Churn Dataset
├── eda.py                     # Step 1: Data Inspection & Visual Plot Generation
├── preprocess.py              # Step 2: Data Cleaning, One-Hot Encoding, Scaling & Splitting
├── train.py                   # Step 3 & 4: Multi-Model Training, Evaluation & Serialization
├── app.py                     # Step 5: High-Tech Streamlit Web Application
├── requirements.txt           # Project Dependencies
├── tasks.md                   # Step-by-step Project Implementation Checklist
├── eda_report.md              # Detailed Exploratory Data Analysis Report
├── model_evaluation_report.md # Beginner-Friendly Machine Learning Model Benchmark Report
├── artifacts/                 # Serialized Machine Learning Models & Metadata
│   ├── model.joblib           # Trained XGBoost Binary Model Artifact
│   ├── preprocessor.joblib    # Fitted Scikit-Learn ColumnTransformer
│   ├── feature_meta.json      # Feature Names and UI Input Options
│   └── model_metrics.json     # Saved Benchmark Scores
├── data_processed/            # Preprocessed Train/Test Split Datasets
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
└── eda_plots/                 # Generated EDA & Evaluation Visualizations
    ├── churn_distribution.png
    ├── tenure_distribution.png
    ├── categorical_churn_rates.png
    ├── best_model_confusion_matrix.png
    ├── roc_curves_comparison.png
    └── top_feature_importance.png
```

---

## 📊 Model Evaluation Benchmark

We trained and evaluated **4 candidate algorithms** on 1,409 unseen test records:

| Model Name | Accuracy | Precision | **Recall (Catching Churners)** | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | 74.80% | 51.57% | **79.68%** | 0.6258 | 0.8415 | 🥈 Strong Recall |
| **Random Forest (Balanced)** | 78.42% | 61.16% | 51.07% | 0.5567 | 0.8252 | 🥉 Baseline |
| **Gradient Boosting** | **80.34%** | **66.19%** | 50.80% | 0.5749 | 0.8450 | 🏅 High Precision |
| **XGBoost (Weighted)** 🏆 | **76.01%** | **53.07%** | **78.61%** | **0.6335** | **0.8455** | 👑 **WINNER** |

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup

Clone the repository and ensure you have Python 3.9+ installed:

```bash
# Navigate to project directory
cd ml

# (Optional) Create a virtual environment
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

### 2. How to Run the End-to-End Pipeline

If you want to re-run the pipeline from scratch:

```bash
# Step 1: Run Exploratory Data Analysis & Generate Visualizations
python eda.py

# Step 2: Clean data, perform One-Hot Encoding, scale features & split data
python preprocess.py

# Step 3 & 4: Train models, evaluate metrics & serialize artifacts
python train.py
```

---

### 3. Launching the Streamlit Web Application

To launch the web interface:

```bash
streamlit run app.py
```

Open your browser and navigate to:
👉 `http://localhost:8501`

---

## 🛠️ Tech Stack & Libraries

* **Frontend & Web UI:** [Streamlit](https://streamlit.io/)
* **Machine Learning:** [XGBoost](https://xgboost.readthedocs.io/), [Scikit-Learn](https://scikit-learn.org/)
* **Data Processing & Analytics:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
* **Data Visualization:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
* **Model Serialization:** [Joblib](https://joblib.readthedocs.io/)

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
