import streamlit as st
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_preprocessing import load_and_preprocess_data

st.title("📈 Model Analytics")

# Load models
@st.cache_resource
def load_models():
    log_reg = pickle.load(open("models/logistic_regression.pkl", "rb"))
    catboost = CatBoostClassifier()
    catboost.load_model("models/catboost_model.cbm")
    return log_reg, catboost

try:
    log_reg, catboost = load_models()
    
    # Load and prepare data
    df = load_and_preprocess_data()
    df = df.dropna()
    
    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]
    
    # Load scaler for logistic regression
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    X_scaled = scaler.transform(X)
    
    # Calculate predictions
    log_pred_proba = log_reg.predict_proba(X_scaled)[:, 1]
    cat_pred_proba = catboost.predict_proba(X)[:, 1]
    
    log_pred = log_reg.predict(X_scaled)
    cat_pred = catboost.predict(X)
    
    # Calculate metrics
    log_auc = roc_auc_score(y, log_pred_proba)
    cat_auc = roc_auc_score(y, cat_pred_proba)
    
    # Display metrics
    st.write("### Model Performance Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Logistic Regression AUC", f"{log_auc:.3f}")
    with col2:
        st.metric("CatBoost AUC", f"{cat_auc:.3f}")
    
    # Model comparison
    st.write("### Model Comparison")
    comparison_data = {
        'Model': ['Logistic Regression', 'CatBoost'],
        'AUC Score': [log_auc, cat_auc]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df)
    
    # Create comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(comparison_df['Model'], comparison_df['AUC Score'])
    ax.set_ylabel('AUC Score')
    ax.set_title('Model Performance Comparison')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # Confusion matrices
    st.write("### Confusion Matrices")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Logistic Regression**")
        cm_lr = confusion_matrix(y, log_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Logistic Regression Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    
    with col2:
        st.write("**CatBoost**")
        cm_cat = confusion_matrix(y, cat_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_cat, annot=True, fmt='d', cmap='Greens', ax=ax)
        ax.set_title('CatBoost Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    
    # Classification reports
    st.write("### Classification Reports")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Logistic Regression**")
        st.text(classification_report(y, log_pred))
    
    with col2:
        st.write("**CatBoost**")
        st.text(classification_report(y, cat_pred))

except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.info("Please ensure models are trained first by running: python utils/model_training.py") 