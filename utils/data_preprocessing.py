import pandas as pd
import pickle

def load_scaler():
    """Load the saved scaler"""
    return pickle.load(open("models/scaler.pkl", "rb"))

def preprocess_input(input_data):
    """Preprocess input data for prediction"""
    scaler = load_scaler()
    input_df = pd.DataFrame([input_data])
    
    # Ensure all required columns are present
    required_columns = [
        'RevolvingUtilizationOfUnsecuredLines',
        'age',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio',
        'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents'
    ]
    
    # Fill missing columns with 0
    for col in required_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[required_columns]
    
    # Handle missing values
    input_df = input_df.fillna(0)
    
    # Scale the data
    scaled_data = scaler.transform(input_df)
    return scaled_data

def load_and_preprocess_data(file_path="data/cs-training.csv"):
    """Load and preprocess the dataset for analysis"""
    df = pd.read_csv(file_path)
    
    # Drop the index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    return df
