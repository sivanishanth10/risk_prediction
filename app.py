import streamlit as st
from utils.model_training import ensure_models_exist

st.set_page_config(page_title="Credit Risk App", page_icon="💳", layout="wide")

st.title("💳 Credit Risk Prediction App")
st.write("Use the sidebar to navigate between pages.")

# Ensure models exist before running
ensure_models_exist()

st.success("Models are ready. Choose a page from the sidebar.")
