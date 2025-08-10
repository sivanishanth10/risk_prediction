# Credit Risk Prediction App

A Streamlit-based web application for credit risk assessment using machine learning models.

## Features

- 📊 **Dataset Overview**: Explore and analyze the credit risk dataset
- 🤖 **Risk Prediction**: Predict credit risk for new borrowers
- 📈 **Model Analytics**: Compare model performance and view detailed metrics
- 📄 **PDF Reports**: Generate detailed risk assessment reports with SHAP visualizations

## Project Structure

```
credit_risk_app/
│── data/
│   ├── cs-training.csv          # Training dataset
│   ├── cs-test.csv             # Test dataset
│   └── sample_entry.json       # Sample borrower data
│
│── models/
│   ├── logistic_regression.pkl  # Logistic Regression model
│   ├── catboost_model.cbm      # CatBoost model
│   └── scaler.pkl              # Feature scaler
│
│── pages/
│   ├── 1_📊_Dataset_Overview.py
│   ├── 2_🤖_Risk_Prediction.py
│   └── 3_📈_Model_Analytics.py
│
│── reports/
│   └── borrower_reports/       # Generated PDF reports
│
│── utils/
│   ├── data_preprocessing.py   # Data preprocessing utilities
│   ├── model_training.py       # Model training functions
│   ├── visualization.py        # SHAP visualization utilities
│   └── pdf_generator.py        # PDF report generation
│
│── app.py                      # Main Streamlit application
│── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. **Clone or download the project**

   ```bash
   cd credit_risk_app
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**

   **Windows:**

   ```bash
   venv\Scripts\activate
   ```

   **Mac/Linux:**

   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### First Time Setup

1. **Train the models** (if not already trained):

   ```bash
   python utils/model_training.py
   ```

   This will:

   - Load the training dataset
   - Train Logistic Regression and CatBoost models
   - Save models to the `models/` directory

2. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

3. **Access the application**:
   - Open your browser and go to `http://localhost:8501`
   - Use the sidebar to navigate between pages

### Regular Usage

After the initial setup, simply run:

```bash
streamlit run app.py
```

## Pages

### 1. Dataset Overview 📊

- View dataset statistics and information
- Explore data distribution
- Check for missing values

### 2. Risk Prediction 🤖

- Enter borrower information
- Get risk predictions from both models
- Generate PDF reports with SHAP visualizations

### 3. Model Analytics 📈

- Compare model performance (AUC scores)
- View confusion matrices
- Analyze classification reports

## Models

The application uses two machine learning models:

1. **Logistic Regression**: Linear model for baseline comparison
2. **CatBoost**: Gradient boosting model for improved performance

Both models are trained on the same dataset and provide risk scores between 0 and 1.

## Output

- **Risk Scores**: Probability of default (0-1 scale)
- **Risk Levels**: Low (< 0.3), Medium (0.3-0.7), High (> 0.7)
- **PDF Reports**: Detailed assessment reports with SHAP visualizations
- **SHAP Plots**: Feature importance analysis

## Troubleshooting

### Common Issues

1. **Models not found**: Run `python utils/model_training.py` to train models
2. **Missing dependencies**: Install with `pip install -r requirements.txt`
3. **Port already in use**: Use `streamlit run app.py --server.port 8502`

### Error Messages

- **EOFError**: Models are corrupted or missing - retrain them
- **ModuleNotFoundError**: Install missing packages
- **FileNotFoundError**: Ensure all required files are present

## Requirements

- Python 3.8+
- 4GB+ RAM (for model training)
- Internet connection (for package installation)

## License

This project is for educational purposes.
