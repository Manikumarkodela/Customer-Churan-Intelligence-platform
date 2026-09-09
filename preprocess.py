import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(csv_path="Telco-Customer-Churn.csv", output_dir="data_processed", artifacts_dir="artifacts"):
    """
    Cleans raw Telco Churn data, encodes categorical features, scales numerical features,
    splits data into Train/Test sets, and saves preprocessor artifacts for inference.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print("=" * 60)
    print("STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 60)
    
    # 1. Load raw dataset
    df = pd.read_csv(csv_path)
    print(f"\n1. Loaded Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Data Cleaning
    # Drop customerID (non-predictive identifier)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # Convert TotalCharges to float; replace empty spaces with 0.0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    print(f"2. Cleaned 'TotalCharges' column. Missing values filled with 0.0")
    
    # 3. Target Variable Encoding
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    print(f"3. Target encoded ('Yes': 1, 'No': 0). Class count: {dict(y.value_counts())}")
    
    # 4. Feature Groups Identification
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_features = [col for col in X.columns if col not in num_features]
    
    print(f"\n4. Identified {len(num_features)} Numerical Features and {len(cat_features)} Categorical Features:")
    print(f"   - Numerical: {num_features}")
    print(f"   - Categorical: {cat_features}")
    
    # 5. Build ColumnTransformer for preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
        ],
        remainder='passthrough'
    )
    
    # Fit preprocessor on X
    X_processed = preprocessor.fit_transform(X)
    
    # Extract feature names after OneHotEncoding
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_features)
    all_feature_names = num_features + list(encoded_cat_names)
    
    print(f"5. Applied StandardScaler & OneHotEncoder. Total features after encoding: {len(all_feature_names)}")
    
    # Convert processed array back to DataFrame for saving
    X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)
    
    # 6. Train / Test Split (80% Train, 20% Test, Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed_df, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"\n6. Train / Test Split (80/20 Stratified):")
    print(f"   - X_train: {X_train.shape}")
    print(f"   - X_test:  {X_test.shape}")
    print(f"   - y_train churn ratio: {y_train.mean():.4f}")
    print(f"   - y_test churn ratio:  {y_test.mean():.4f}")
    
    # 7. Save Split Datasets
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    # 8. Save Preprocessor Artifacts
    joblib.dump(preprocessor, os.path.join(artifacts_dir, "preprocessor.joblib"))
    
    # Convert options to native python types for JSON serialization
    cat_options = {}
    for col in cat_features:
        unique_vals = X[col].unique()
        cat_options[col] = [int(v) if isinstance(v, (np.integer, int)) else str(v) for v in unique_vals]
        
    feature_meta = {
        "num_features": num_features,
        "cat_features": cat_features,
        "encoded_feature_names": all_feature_names,
        "cat_options": cat_options
    }
    
    with open(os.path.join(artifacts_dir, "feature_meta.json"), "w") as f:
        json.dump(feature_meta, f, indent=4)
        
    print(f"\nPreprocessed dataset saved to '{output_dir}/'")
    print(f"Preprocessor object and metadata saved to '{artifacts_dir}/'")
    print("=" * 60)

if __name__ == "__main__":
    preprocess_data()
