import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Telco AI — Customer Churn Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Pure White / Off-White Aesthetic & Hiding Default Streamlit Header/Menu
st.markdown("""
<style>
    /* Hide Default Streamlit Header, Deploy Button, Hamburger Menu & Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stAppDeployButton {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    
    /* Typography & Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #0f172a;
    }
    
    /* Main Background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Header Badge */
    .header-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
        color: #7c3aed;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 6px 14px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
        border: 1px solid rgba(124, 58, 237, 0.2);
    }
    
    /* Hero Title */
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #0f172a 0%, #334155 50%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    /* Telemetry Cards */
    .telemetry-card {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    
    .telemetry-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #7c3aed;
    }
    
    .telemetry-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
    }

    /* Risk Badges */
    .risk-badge-low {
        background-color: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .risk-badge-high {
        background-color: #fef2f2;
        color: #dc2626;
        border: 1px solid #fca5a5;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
    }

    .risk-badge-medium {
        background-color: #fffbeb;
        color: #d97706;
        border: 1px solid #fde68a;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
    }

    /* Primary Action Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 14px 0 rgba(124, 58, 237, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(124, 58, 237, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. ARTIFACT LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ml_artifacts():
    """Loads model, preprocessor transformer, and metadata files."""
    model = joblib.load("artifacts/model.joblib")
    preprocessor = joblib.load("artifacts/preprocessor.joblib")
    with open("artifacts/feature_meta.json", "r") as f:
        feature_meta = json.load(f)
    with open("artifacts/model_metrics.json", "r") as f:
        metrics = json.load(f)
    return model, preprocessor, feature_meta, metrics

try:
    model, preprocessor, feature_meta, metrics = load_ml_artifacts()
    artifacts_ready = True
except Exception as err:
    artifacts_ready = False
    st.error(f"Failed to load ML artifacts: {err}")

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & HISTORICAL KPIS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/lightning-bolt.png", width=55)
    st.title("Telco AI Engine")
    st.caption("Next-Gen Customer Retention Intelligence")
    st.divider()
    
    navigation = st.radio(
        "Navigation",
        ["🔮 Single Predictor", "📁 Batch CSV Predictor", "📊 Analytics Dashboard"],
        index=0
    )
    
    st.divider()
    st.markdown("### 📈 Historical KPIs")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("Total Accounts", "7,043")
        st.metric("Avg Monthly Bill", "$64.76")
    with col_sb2:
        st.metric("Avg Churn Rate", "26.54%")
        st.metric("Model AUC", f"{metrics['all_results'][metrics['best_model_name']]['ROC-AUC'] if artifacts_ready else '0.8455'}")

# -----------------------------------------------------------------------------
# 4. MAIN PAGE HERO HEADER
# -----------------------------------------------------------------------------
st.markdown("<div class='header-badge'>⚡ Powered by XGBoost Machine Learning</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-title'>Customer Churn Intelligence Platform</div>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 1rem; margin-bottom: 25px;'>Real-time AI telemetry, churn risk forecasting, and retention action analytics.</p>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. TAB 1: SINGLE CUSTOMER PREDICTOR
# -----------------------------------------------------------------------------
if navigation == "🔮 Single Predictor":
    st.markdown("### 📝 Customer Profile & Telemetry Inputs")
    
    with st.form("single_predict_form"):
        col1, col2, col3 = st.columns(3)
        
        # Column 1: Demographics & Account Info
        with col1:
            st.subheader("👤 Account Details")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (>= 65)" if x == 1 else "No")
            partner = st.selectbox("Partner Status", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (Months active)", min_value=0, max_value=72, value=12)
            
        # Column 2: Subscription & Services
        with col2:
            st.subheader("🌐 Services & Add-ons")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
            device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

        # Column 3: Contract & Billing
        with col3:
            st.subheader("💳 Contract & Billing")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=65.0, step=1.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly_charges), step=10.0)

        submit_btn = st.form_submit_button("⚡ Calculate Churn Probability", use_container_width=True)

    if submit_btn and artifacts_ready:
        # Build dataframe matching preprocessor format
        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])
        
        # Preprocess & Infer
        X_encoded = preprocessor.transform(input_data)
        churn_prob = float(model.predict_proba(X_encoded)[0, 1])
        churn_pred = int(churn_prob >= 0.5)
        retention_prob = 1.0 - churn_prob
        
        st.divider()
        st.markdown("### 🎯 Prediction Results")
        
        out_col1, out_col2 = st.columns([1.2, 1.0])
        
        with out_col1:
            if churn_prob >= 0.60:
                st.markdown(f"""
                <div class='risk-badge-high'>
                    🚨 HIGH CHURN RISK: {churn_prob*100:.1f}% Probability
                    <p style='font-size: 0.85rem; font-weight: normal; margin-top: 4px;'>Customer shows high risk of cancellation. Offer immediate 1-year contract discount.</p>
                </div>
                """, unsafe_allow_html=True)
            elif churn_prob >= 0.35:
                st.markdown(f"""
                <div class='risk-badge-medium'>
                    ⚠️ MEDIUM CHURN RISK: {churn_prob*100:.1f}% Probability
                    <p style='font-size: 0.85rem; font-weight: normal; margin-top: 4px;'>Moderate churn risk. Recommend Tech Support or Security add-on packages.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='risk-badge-low'>
                    ✅ LOW CHURN RISK: {churn_prob*100:.1f}% Probability
                    <p style='font-size: 0.85rem; font-weight: normal; margin-top: 4px;'>Customer is highly loyal and satisfied.</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(churn_prob)
            st.caption(f"Churn Risk: {churn_prob*100:.2f}% | Retention Probability: {retention_prob*100:.2f}%")
            
        with out_col2:
            st.markdown("#### 💡 Telemetry Summary")
            t1, t2 = st.columns(2)
            with t1:
                st.markdown(f"""
                <div class='telemetry-card'>
                    <div class='telemetry-value'>${total_charges:.2f}</div>
                    <div class='telemetry-label'>Est. Lifetime Value</div>
                </div>
                """, unsafe_allow_html=True)
            with t2:
                st.markdown(f"""
                <div class='telemetry-card'>
                    <div class='telemetry-value'>{tenure} mos</div>
                    <div class='telemetry-label'>Account Duration</div>
                </div>
                """, unsafe_allow_html=True)
                
        # Export JSON Report
        report_data = input_data.to_dict(orient="records")[0]
        report_data["churn_probability"] = round(churn_prob, 4)
        report_data["churn_prediction"] = churn_pred
        
        st.download_button(
            label="📥 Download Telemetry Report (JSON)",
            data=json.dumps(report_data, indent=4),
            file_name="churn_telemetry_report.json",
            mime="application/json"
        )

# -----------------------------------------------------------------------------
# 6. TAB 2: BATCH CSV PREDICTOR
# -----------------------------------------------------------------------------
elif navigation == "📁 Batch CSV Predictor":
    st.markdown("### 📁 Bulk Customer Batch Telemetry")
    st.markdown("Upload a CSV file containing customer details to calculate churn probabilities across accounts.")
    
    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None and artifacts_ready:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(batch_df)}** records from `{uploaded_file.name}`.")
        
        if st.button("🚀 Process Batch Telemetry Predictions"):
            with st.spinner("Processing batch predictions with XGBoost AI model..."):
                pred_input = batch_df.copy()
                if "customerID" in pred_input.columns:
                    pred_input = pred_input.drop(columns=["customerID"])
                if "Churn" in pred_input.columns:
                    pred_input = pred_input.drop(columns=["Churn"])
                    
                pred_input['TotalCharges'] = pd.to_numeric(pred_input['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0.0)
                
                X_batch = preprocessor.transform(pred_input)
                probs = model.predict_proba(X_batch)[:, 1]
                preds = (probs >= 0.5).astype(int)
                
                batch_df["Churn_Probability"] = np.round(probs, 4)
                batch_df["Churn_Prediction"] = np.where(preds == 1, "High Risk (Churn)", "Low Risk (Stay)")
                
                st.success(f"Batch prediction completed for {len(batch_df)} accounts!")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Processed", len(batch_df))
                c2.metric("Flagged Churners", int(sum(preds)))
                c3.metric("Predicted Churn Rate", f"{(sum(preds)/len(batch_df))*100:.1f}%")
                
                st.dataframe(batch_df.head(15), use_container_width=True)
                
                csv_bytes = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predicted Results (CSV)",
                    data=csv_bytes,
                    file_name="churn_batch_predictions.csv",
                    mime="text/csv"
                )

# -----------------------------------------------------------------------------
# 7. TAB 3: ANALYTICS DASHBOARD
# -----------------------------------------------------------------------------
elif navigation == "📊 Analytics Dashboard":
    st.markdown("### 📊 Model Performance & EDA Visualizations")
    
    if os.path.exists("eda_plots/best_model_confusion_matrix.png"):
        st.markdown("#### 🎯 Model Evaluation Plots")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.image("eda_plots/best_model_confusion_matrix.png", caption="Confusion Matrix (Test Set)", use_container_width=True)
        with p_col2:
            st.image("eda_plots/roc_curves_comparison.png", caption="ROC Curves Benchmark", use_container_width=True)
            
    if os.path.exists("eda_plots/top_feature_importance.png"):
        st.image("eda_plots/top_feature_importance.png", caption="Top 10 Factors Predicting Churn", use_container_width=True)
        
    st.divider()
    st.markdown("#### 📈 Key Exploratory Data Analysis Charts")
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        if os.path.exists("eda_plots/tenure_distribution.png"):
            st.image("eda_plots/tenure_distribution.png", caption="Tenure vs Churn Distribution", use_container_width=True)
    with e_col2:
        if os.path.exists("eda_plots/categorical_churn_rates.png"):
            st.image("eda_plots/categorical_churn_rates.png", caption="Churn Rates across Service Categories", use_container_width=True)
