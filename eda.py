import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

def run_eda(csv_path="Telco-Customer-Churn.csv", output_dir="eda_plots"):
    """
    Performs Exploratory Data Analysis (EDA) on Telco Customer Churn dataset,
    saves visualization plots, and prints key statistical findings.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("STEP 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    # 1. Load Data
    df = pd.read_csv(csv_path)
    print(f"\n1. Data Overview:")
    print(f"   - Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   - Customer ID Unique: {df['customerID'].nunique() == len(df)}")
    
    # Clean TotalCharges column for analysis
    df['TotalCharges_clean'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    missing_tc = df['TotalCharges_clean'].isnull().sum()
    print(f"   - Missing/Blank TotalCharges values: {missing_tc}")
    
    # 2. Target Variable Analysis
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    print(f"\n2. Target Distribution (Churn):")
    print(f"   - No:  {churn_counts['No']} ({churn_pct['No']:.2f}%)")
    print(f"   - Yes: {churn_counts['Yes']} ({churn_pct['Yes']:.2f}%)")
    
    # Plot Target Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=churn_counts.index, y=churn_counts.values, palette=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Customer Churn Count", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Customers")
    ax.set_xlabel("Churn Status")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "churn_distribution.png"), dpi=300)
    plt.close()
    
    # 3. Numerical Features Analysis
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges_clean']
    print("\n3. Numerical Features Summary (by Churn):")
    for col in num_cols:
        summary = df.groupby('Churn')[col].agg(['mean', 'median', 'std'])
        print(f"\n--- {col} ---")
        print(summary)
        
        # Distribution plot
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(data=df, x=col, hue="Churn", common_norm=False, palette=["#2ecc71", "#e74c3c"], shade=True, ax=ax)
        ax.set_title(f"Distribution of {col} by Churn", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_distribution.png"), dpi=300)
        plt.close()
        
    # 4. Categorical Features vs Churn Rate
    cat_cols = ['Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'PaperlessBilling', 'SeniorCitizen']
    print("\n4. Key Categorical Feature Churn Rates:")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    key_cats = ['Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'PaperlessBilling']
    
    for idx, col in enumerate(key_cats):
        churn_rates = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        churn_rates.columns = [col, 'ChurnRate']
        churn_rates = churn_rates.sort_values(by='ChurnRate', ascending=False)
        
        print(f"\n--- {col} Churn Rate (%) ---")
        for _, row in churn_rates.iterrows():
            print(f"   {row[col]}: {row['ChurnRate']:.2f}%")
            
        sns.barplot(data=churn_rates, x='ChurnRate', y=col, palette="Blues_r", ax=axes[idx])
        axes[idx].set_title(f"Churn Rate by {col} (%)", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Churn Rate (%)")
        axes[idx].set_ylabel("")
        for p in axes[idx].patches:
            axes[idx].annotate(f"{p.get_width():.1f}%", (p.get_width() + 1, p.get_y() + p.get_height() / 2.),
                               ha='left', va='center', fontsize=9, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "categorical_churn_rates.png"), dpi=300)
    plt.close()
    
    print(f"\nEDA completed successfully! Visualization plots saved to '{output_dir}/'.")

if __name__ == "__main__":
    run_eda()
