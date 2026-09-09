import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

def train_and_evaluate(data_dir="data_processed", artifacts_dir="artifacts", plots_dir="eda_plots"):
    """
    Trains multiple ML models for Customer Churn Prediction, evaluates performance metrics,
    selects the best model, saves serialized artifacts and visualization plots.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print("=" * 60)
    print("STEP 3: MODEL TRAINING, EVALUATION & SELECTION")
    print("=" * 60)
    
    # 1. Load Processed Datasets
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    print(f"\n1. Loaded Datasets:")
    print(f"   - Training sample size: {X_train.shape[0]}")
    print(f"   - Testing sample size:  {X_test.shape[0]}")
    
    # Calculate scale_pos_weight for XGBoost to handle class imbalance
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    
    # 2. Define Models Dictionary
    models = {
        "Logistic Regression (Balanced)": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest (Balanced)": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        "XGBoost (Weighted)": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, scale_pos_weight=ratio, random_state=42, eval_metric='logloss')
    }
    
    results = {}
    fitted_models = {}
    
    print("\n2. Training & Evaluating Candidate Models:")
    print("-" * 60)
    
    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
        # Predict class and probability
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4)
        }
        
        print(f"[{name}]")
        print(f"   - Accuracy:  {acc * 100:.2f}%")
        print(f"   - Precision: {prec * 100:.2f}%")
        print(f"   - Recall:    {rec * 100:.2f}%")
        print(f"   - F1-Score:  {f1:.4f}")
        print(f"   - ROC-AUC:   {auc:.4f}\n")
        
    # 3. Model Selection based on ROC-AUC and Recall trade-off
    best_model_name = max(results, key=lambda k: results[k]["ROC-AUC"])
    best_model = fitted_models[best_model_name]
    
    print("=" * 60)
    print(f"🏆 BEST PERFORMING MODEL: {best_model_name}")
    print(f"   ROC-AUC: {results[best_model_name]['ROC-AUC']} | F1-Score: {results[best_model_name]['F1-Score']}")
    print("=" * 60)
    
    # 4. Save Serialized Best Model & Metrics
    joblib.dump(best_model, os.path.join(artifacts_dir, "model.joblib"))
    
    metrics_meta = {
        "best_model_name": best_model_name,
        "all_results": results
    }
    
    with open(os.path.join(artifacts_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics_meta, f, indent=4)
        
    # 5. Generate Visualizations (Confusion Matrix & ROC Curve for Best Model)
    best_y_pred = best_model.predict(X_test)
    best_y_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, best_y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Retained (0)', 'Churned (1)'],
                yticklabels=['Retained (0)', 'Churned (1)'])
    ax.set_title(f"Confusion Matrix: {best_model_name}", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "best_model_confusion_matrix.png"), dpi=300)
    plt.close()
    
    # ROC Curve Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in fitted_models.items():
        y_proba_m = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba_m)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['ROC-AUC']:.3f})", linewidth=2)
        
    ax.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.500)")
    ax.set_title("ROC Curves Comparison across Models", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "roc_curves_comparison.png"), dpi=300)
    plt.close()
    
    # Feature Importance for Tree Models or Logistic Coefficients
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_names = X_train.columns
        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feat_df = feat_df.sort_values(by="Importance", ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis", ax=ax)
        ax.set_title(f"Top 10 Most Important Features ({best_model_name})", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "top_feature_importance.png"), dpi=300)
        plt.close()
        
    print(f"\nModel artifacts saved to '{artifacts_dir}/model.joblib'")
    print(f"Evaluation plots saved to '{plots_dir}/'")

if __name__ == "__main__":
    train_and_evaluate()
