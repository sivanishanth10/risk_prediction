import streamlit as st
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from utils.data_preprocessing import preprocess_input
from utils.visualization import shap_summary_plot
from utils.pdf_generator import generate_pdf_report

st.title("🤖 Risk Prediction")

# Load models
@st.cache_resource
def load_models():
    log_reg = pickle.load(open("models/logistic_regression.pkl", "rb"))
    catboost = CatBoostClassifier()
    catboost.load_model("models/catboost_model.cbm")
    return log_reg, catboost

try:
    log_reg, catboost = load_models()
    
    # Create input form
    st.write("### Enter Borrower Information")
    
    # Sample input fields based on the dataset columns
    col1, col2 = st.columns(2)
    
    with col1:
        revolving_utilization = st.number_input("Revolving Utilization", min_value=0.0, max_value=100.0, value=50.0)
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        number_of_time_30_59_days_past_due = st.number_input("30-59 Days Past Due", min_value=0, max_value=100, value=0)
        debt_ratio = st.number_input("Debt Ratio", min_value=0.0, max_value=100.0, value=0.5)
        monthly_income = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=5000)
        
    with col2:
        number_of_open_credit_lines = st.number_input("Open Credit Lines", min_value=0, max_value=100, value=10)
        number_of_times_90_days_late = st.number_input("90+ Days Late", min_value=0, max_value=100, value=0)
        number_real_estate_loans = st.number_input("Real Estate Loans", min_value=0, max_value=100, value=1)
        number_of_time_60_89_days_past_due = st.number_input("60-89 Days Past Due", min_value=0, max_value=100, value=0)
        number_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=2)
    
    if st.button("Predict Risk"):
        # Create input data
        input_data = {
            'RevolvingUtilizationOfUnsecuredLines': revolving_utilization,
            'age': age,
            'NumberOfTime30-59DaysPastDueNotWorse': number_of_time_30_59_days_past_due,
            'DebtRatio': debt_ratio,
            'MonthlyIncome': monthly_income,
            'NumberOfOpenCreditLinesAndLoans': number_of_open_credit_lines,
            'NumberOfTimes90DaysLate': number_of_times_90_days_late,
            'NumberRealEstateLoansOrLines': number_real_estate_loans,
            'NumberOfTime60-89DaysPastDueNotWorse': number_of_time_60_89_days_past_due,
            'NumberOfDependents': number_of_dependents
        }
        
        # Preprocess input
        X_scaled = preprocess_input(input_data)
        input_df = pd.DataFrame([input_data])
        
        # Make predictions
        log_pred = log_reg.predict_proba(X_scaled)[0][1]
        cat_pred = catboost.predict_proba(input_df)[0][1]
        
        # Display results
        st.write("### Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Logistic Regression Risk", f"{log_pred:.3f}")
        with col2:
            st.metric("CatBoost Risk", f"{cat_pred:.3f}")
        
        # Risk interpretation
        avg_risk = (log_pred + cat_pred) / 2
        if avg_risk < 0.3:
            risk_level = "Low Risk"
            color = "green"
        elif avg_risk < 0.7:
            risk_level = "Medium Risk"
            color = "orange"
        else:
            risk_level = "High Risk"
            color = "red"
        
        st.write(f"**Overall Risk Assessment:** {risk_level}")
        
        # Generate SHAP plot and PDF report
        try:
            shap_summary_plot(log_reg, X_scaled, "log_reg")
            pdf_path = generate_pdf_report("Borrower", f"{avg_risk:.3f}")
            st.success(f"📄 Report generated: {pdf_path}")
        except Exception as e:
            st.warning(f"Could not generate report: {str(e)}")

except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.info("Please ensure models are trained first by running: python utils/model_training.py") 