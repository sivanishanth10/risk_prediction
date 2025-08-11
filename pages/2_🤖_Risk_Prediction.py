import streamlit as st
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from utils.data_preprocessing import preprocess_input
from utils.visualization import shap_summary_plot
from utils.pdf_generator import generate_pdf_report
from utils.dashboard_components import display_prediction_summary, create_risk_factors_chart
import numpy as np
import os
from datetime import datetime

st.title("🤖 Risk Prediction")

# Demo section for PDF report
with st.expander("📋 Demo: Generate Sample PDF Report", expanded=False):
    st.write("**Try the enhanced PDF report generation with sample data:**")
    
    if st.button("🎯 Generate Sample Report", type="secondary"):
        try:
            from utils.pdf_generator import create_sample_pdf_report
            
            with st.spinner("Generating sample PDF report..."):
                sample_pdf_path = create_sample_pdf_report()
            
            if sample_pdf_path:
                st.success("✅ Sample PDF report generated successfully!")
                
                # Read and provide download button for sample report
                with open(sample_pdf_path, "rb") as pdf_file:
                    sample_pdf_bytes = pdf_file.read()
                
                st.download_button(
                    label="📥 Download Sample Report (PDF)",
                    data=sample_pdf_bytes,
                    file_name="sample_credit_risk_report.pdf",
                    mime="application/pdf",
                    help="Download the sample report to see all enhanced features",
                    use_container_width=True
                )
                
                st.info("**Sample Report Features:**")
                st.write("• Professional formatting with tables and charts")
                st.write("• Risk visualization charts")
                st.write("• Comprehensive risk factors analysis")
                st.write("• Detailed recommendations and mitigation strategies")
                st.write("• Executive summary and borrower information")
                
            else:
                st.error("Failed to generate sample report")
                
        except Exception as e:
            st.error(f"Error generating sample report: {str(e)}")

st.write("---")

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
        
        # Create predictions dictionary
        predictions = {
            'Logistic Regression': log_pred,
            'CatBoost': cat_pred
        }
        
        # Risk interpretation
        avg_risk = (log_pred + cat_pred) / 2
        if avg_risk < 0.3:
            risk_level = "Low Risk"
        elif avg_risk < 0.7:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
        
        # Display comprehensive prediction summary using dashboard components
        st.write("### 📊 Prediction Results")
        display_prediction_summary(predictions, risk_level, confidence=0.85)
        
        # Risk factors analysis
        st.write("### 🔍 Risk Factors Analysis")
        
        # Create risk factors dictionary
        risk_factors = {
            'Revolving Utilization': revolving_utilization / 100,
            'Age Factor': max(0, (35 - age) / 35),  # Younger = higher risk
            'Payment History': (number_of_time_30_59_days_past_due + number_of_time_60_89_days_past_due + number_of_times_90_days_late) / 300,
            'Debt Burden': debt_ratio,
            'Income Stability': max(0, (monthly_income - 5000) / 50000)
        }
        
        # Display risk factors chart
        st.plotly_chart(create_risk_factors_chart(risk_factors), use_container_width=True)
        
        # Additional insights
        st.write("### 💡 Risk Insights")
        
        # Identify top risk factors
        sorted_factors = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Risk Factors:**")
            for i, (factor, score) in enumerate(sorted_factors[:3]):
                st.write(f"{i+1}. {factor}: {score:.3f}")
        
        with col2:
            st.write("**Recommendations:**")
            if risk_factors['Payment History'] > 0.5:
                st.write("⚠️ Improve payment history")
            if risk_factors['Debt Burden'] > 0.7:
                st.write("⚠️ Reduce debt burden")
            if risk_factors['Revolving Utilization'] > 0.8:
                st.write("⚠️ Lower credit utilization")
            if avg_risk < 0.3:
                st.write("✅ Good credit profile")
        
        # Generate SHAP plot and PDF report
        try:
            shap_summary_plot(log_reg, X_scaled, "log_reg")
            
            # Generate comprehensive PDF report
            pdf_path = generate_pdf_report(
                name="Borrower", 
                prediction=f"{avg_risk:.3f}",
                input_data=input_data,
                risk_factors=risk_factors
            )
            
            if pdf_path:
                st.success("📄 Comprehensive PDF report generated successfully!")
                
                # Add prominent download button
                st.write("### 📥 Download Report")
                
                # Read the PDF file and create download button
                with open(pdf_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                
                st.download_button(
                    label="📥 Download Credit Risk Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"credit_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    help="Click to download the comprehensive credit risk assessment report",
                    use_container_width=True
                )
                
                # Show report details
                st.info(f"**Report Details:** {os.path.basename(pdf_path)}")
                st.write("The report includes:")
                st.write("• Executive summary with risk assessment")
                st.write("• Risk visualization charts")
                st.write("• Borrower information summary")
                st.write("• Risk factors analysis with charts")
                st.write("• SHAP feature importance analysis")
                st.write("• Detailed recommendations")
                st.write("• Risk mitigation strategies")
                
            else:
                st.error("Failed to generate PDF report")
                
        except Exception as e:
            st.warning(f"Could not generate report: {str(e)}")
            st.info("Please ensure all required dependencies are installed")

except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.info("Please ensure models are trained first by running: python utils/model_training.py") 