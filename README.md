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

### 3. Enhanced Model Analytics (`pages/3_📈_Model_Analytics.py`)

#### New Visualizations:

- **ROC Curves** - Interactive comparison with AUC scores
- **Prediction Distributions** - Histograms with statistical overlays
- **Feature Importance Comparison** - Side-by-side model analysis
- **Risk Score Analysis** - Pie charts showing risk category distributions
- **Prediction Correlation** - Scatter plots with agreement lines
- **📄 PDF Report Generation** - Comprehensive model analysis reports

#### Performance Metrics:

- Accuracy, Precision, Recall, F1-Score
- AUC scores with confidence intervals
- Model agreement analysis
- Risk distribution analysis

#### PDF Report Features:

- **Model Performance Summary** - AUC scores and comparison
- **Data Quality Metrics** - Sample counts and distribution
- **Feature Importance Analysis** - Comparative model insights
- **Risk Assessment** - Model health indicators
- **Recommendations** - Model improvement suggestions

### 4. Reusable Dashboard Components (`utils/dashboard_components.py`)

#### Component Library:

- **Risk Summary Cards** - HTML-styled risk assessment displays
- **Chart Generators** - Standardized visualization functions
- **Layout Helpers** - Consistent spacing and organization
- **Interactive Elements** - Hover effects and annotations

#### Key Functions:

```python
# Display comprehensive prediction summary
display_prediction_summary(predictions_dict, risk_level, confidence)

# Create standardized charts
create_risk_summary_card(risk_score, risk_level, confidence)
create_prediction_gauge(value, title, color)
create_model_comparison_chart(predictions_dict)
create_risk_breakdown_chart(risk_score)
```

### 5. Enhanced PDF Generation (`utils/pdf_generator.py`)

#### Professional Report Features:

- **A4 Layout** - Professional page sizing and margins
- **Custom Styling** - Professional fonts, colors, and spacing
- **Embedded Charts** - Matplotlib visualizations integrated into PDF
- **Structured Content** - Organized sections with clear headings
- **Data Tables** - Formatted input data and risk factors
- **Executive Summary** - High-level risk assessment overview

#### Chart Generation:

```python
# Risk assessment visualization
create_risk_chart(risk_score, risk_level)

# Risk factors analysis chart
create_risk_factors_chart(risk_factors)

# Comprehensive report generation
generate_pdf_report(
    name="Borrower",
    prediction=risk_score,
    input_data=input_data,
    risk_factors=risk_factors
)
```

#### Report Sections:

1. **Title Page** - Professional header with report metadata
2. **Executive Summary** - Risk assessment and recommendations
3. **Risk Visualization** - Custom charts and graphs
4. **Borrower Information** - Formatted input data tables
5. **Risk Factors Analysis** - Individual factor breakdown
6. **SHAP Analysis** - Feature importance visualization
7. **Recommendations** - Risk-specific action items
8. **Risk Mitigation** - Comprehensive strategies
9. **Professional Footer** - Confidentiality and metadata

## 🎨 Visualization Types

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

## 🚀 New Features Added

### 1. Enhanced Risk Prediction Page (`pages/2_🤖_Risk_Prediction.py`)

#### Visual Components:

- **Risk Summary Card** - Prominent display of overall risk assessment
- **Interactive Risk Gauges** - Visual representation of risk scores from each model
- **Model Comparison Charts** - Side-by-side comparison of predictions
- **Confidence Intervals** - Uncertainty visualization for predictions
- **Risk Breakdown Charts** - Categorical risk level analysis
- **Risk Factors Analysis** - Individual factor contribution visualization
- **Smart Recommendations** - AI-powered insights and suggestions
- **📥 Enhanced PDF Download** - Comprehensive credit risk assessment reports

#### Key Visualizations:

```python
# Risk gauges with color-coded thresholds
create_prediction_gauge(risk_score, "Model Risk")

# Model comparison bar charts
create_model_comparison_chart(predictions_dict)

# Risk breakdown with current position
create_risk_breakdown_chart(average_risk)

# Confidence intervals
create_confidence_interval_chart(predictions_dict)

# PDF Report Generation
pdf_path = generate_pdf_report(
    name="Borrower",
    prediction=risk_score,
    input_data=input_data,
    risk_factors=risk_factors
)
```

#### PDF Report Features:

- **Professional Formatting** - A4 layout with professional styling
- **Executive Summary** - Risk assessment and recommendations
- **Risk Visualization Charts** - Custom matplotlib charts embedded in PDF
- **Borrower Information Tables** - Formatted input data summary
- **Risk Factors Analysis** - Charts and tables for individual factors
- **SHAP Integration** - Feature importance analysis
- **Detailed Recommendations** - Risk-specific action items
- **Risk Mitigation Strategies** - Comprehensive guidance
- **Download Button** - Prominent Streamlit download functionality

### 2. Advanced Prediction Analytics (`pages/4_📊_Prediction_Analytics.py`)

## 🚀 Usage Examples

### Basic Risk Prediction:

```python
# Input borrower information
# Click "Predict Risk" button
# View comprehensive visualizations
# Analyze risk factors and recommendations
# Download detailed PDF report
```

### Advanced Analytics:

```python
# Navigate to Prediction Analytics page
# Select analysis section from sidebar
# Explore feature importance, distributions, correlations
# Compare model performance metrics
```

### Model Analysis:

```python
# Visit Model Analytics page
# View ROC curves and performance metrics
# Analyze prediction distributions
# Compare feature importance between models
# Generate comprehensive PDF analysis report
```

### PDF Report Generation:

```python
# Automatic generation after risk prediction
# Manual generation from Model Analytics page
# Sample report generation for demonstration
# Professional formatting with embedded charts
# Easy download with prominent buttons
```

## 📥 PDF Download Features

### Download Buttons:

- **Primary Download** - After risk prediction with borrower data
- **Sample Report** - Demo PDF with sample data
- **Model Analysis** - Comprehensive model performance report
- **Automatic Naming** - Timestamped filenames for organization

### Report Types:

1. **Credit Risk Assessment** - Individual borrower analysis
2. **Model Analysis Report** - Performance and comparison
3. **Sample Report** - Demonstration of capabilities

### File Features:

- **Professional Formatting** - A4 layout with margins
- **Embedded Visualizations** - Charts and graphs
- **Structured Content** - Clear sections and headings
- **Comprehensive Data** - All input and analysis results
- **Actionable Insights** - Recommendations and strategies

## 🧪 Testing and Validation

### Test Script:

```bash
# Run PDF generation tests
python test_pdf_generation.py
```

### Test Coverage:

- Sample report generation
- Custom report creation
- File integrity verification
- Error handling validation
- Chart generation testing

### Quality Checks:

- File size validation (>1KB minimum)
- Content structure verification
- Chart embedding confirmation
- Professional formatting validation
