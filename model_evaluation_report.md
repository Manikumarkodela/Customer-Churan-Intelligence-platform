# 📊 Beginner-Friendly Machine Learning Model Evaluation Report

This report explains the training results of our **Customer Churn Prediction Models** in plain, simple terms so anyone—regardless of technical background—can understand how well our AI predicts customer churn.

---

## 🎯 What is Customer Churn Prediction?

In business, **"Churn"** means a customer cancels their subscription or stops using a service.
- **`0` (Retained):** The customer stays with the company.
- **`1` (Churned):** The customer leaves the company.

Our goal is to **catch potential churners before they leave** so the customer retention team can offer discounts or support to keep them!

---

## 💡 What Do The Metrics Mean? (In Simple Words)

When evaluating Machine Learning models, we don't look at just one number. Here is what each metric measures:

| Metric | Simple Explanation | Business Relevance |
| :--- | :--- | :--- |
| **Accuracy** | Percentage of overall correct predictions (both stays and leaves). | Good general indicator, but can be misleading when churners are a minority (~26%). |
| **Precision** | Out of all customers the model flagged as *"Going to Churn"*, how many actually churned? | High precision avoids wasting promotional discounts on loyal customers. |
| **Recall (Sensitivity)** | Out of all actual customers who churned, how many did the model **catch**? | **Most critical metric!** High recall means we don't miss at-risk customers. |
| **F1-Score** | The balance between Precision and Recall. | Measures overall model quality without bias toward high precision or high recall. |
| **ROC-AUC** | The model's overall capability to rank a churner higher than a non-churner. | Score of `1.0` is perfection, `0.5` is random guessing. |

---

## 🏆 Model Performance Comparison Table

We trained and evaluated **4 candidate algorithms** on 1,409 unseen test customers:

| Model Name | Accuracy | Precision | Recall (Catching Churners) | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | 74.80% | 51.57% | **79.68%** | 0.6258 | 0.8415 | 🥈 Strong Recall |
| **Random Forest (Balanced)** | 78.42% | 61.16% | 51.07% | 0.5567 | 0.8252 | 🥉 Moderate |
| **Gradient Boosting** | **80.34%** | **66.19%** | 50.80% | 0.5749 | 0.8450 | 🏅 High Precision |
| **XGBoost (Weighted)** 🏆 | **76.01%** | **53.07%** | **78.61%** | **0.6335** | **0.8455** | 👑 **WINNER** |

---

## 👑 Why XGBoost (Weighted) Was Selected as the Winner

1. **High Recall (78.61%)**: Out of 100 customers who were actually going to leave, **XGBoost successfully catches nearly 79 of them!**
2. **Highest Overall ROC-AUC (0.8455)**: Demonstrates the best overall ability to separate churners from loyal customers.
3. **Best F1-Score (0.6335)**: Delivers the optimal balance between catching churners (Recall) and minimizing false alarms (Precision).

---

## 🔑 Top 5 Factors Driving Customer Churn

According to our winning model, the top features influencing whether a customer stays or leaves are:
1. **Contract Type (Month-to-Month vs 1-Year/2-Year)**
2. **Tenure (Length of time customer has been with company)**
3. **Internet Service Type (Fiber Optic vs DSL/None)**
4. **Monthly Charges ($ amount billed)**
5. **Tech Support & Online Security Add-ons**

---

## 🖼️ Saved Visualizations in `eda_plots/`
- **`best_model_confusion_matrix.png`**: Breakdown of True Positives, False Positives, True Negatives, and False Negatives.
- **`roc_curves_comparison.png`**: Comparison curve showing performance of all 4 models.
- **`top_feature_importance.png`**: Bar chart showing top 10 factors predicting churn.
