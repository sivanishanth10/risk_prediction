import streamlit as st
import pandas as pd
from utils.data_preprocessing import load_and_preprocess_data

st.title("📊 Dataset Overview")

# Load the dataset
df = load_and_preprocess_data()

st.write("### Data Preview")
st.dataframe(df.head())

st.write("### Dataset Information")
st.write(f"**Shape:** {df.shape}")
st.write(f"**Columns:** {list(df.columns)}")

st.write("### Summary Statistics")
st.write(df.describe())

st.write("### Missing Values")
missing_data = df.isnull().sum()
st.write(missing_data)

# Display target distribution
st.write("### Target Variable Distribution")
target_counts = df['SeriousDlqin2yrs'].value_counts()
st.write(target_counts)

# Create a simple bar chart
st.bar_chart(target_counts)
