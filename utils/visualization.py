import matplotlib.pyplot as plt
import shap
import pickle
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import warnings

def shap_summary_plot(model, X, model_type, save_path="reports/shap_summary.png"):
    """Generate and save SHAP summary plot"""
    try:
        # Create reports directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create explainer based on model type
        if model_type == "log_reg":
            explainer = shap.LinearExplainer(model, X)
        else:
            explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer(X)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        # Suppress NumPy RNG FutureWarning from SHAP internals
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The NumPy global RNG was seeded",
                category=FutureWarning,
            )
            shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    except Exception as e:
        print(f"Error generating SHAP plot: {e}")
        return None

def create_feature_importance_plot(model, feature_names, model_type="tree", save_path="reports/feature_importance.png"):
    """Create feature importance visualization"""
    try:
        if model_type == "log_reg":
            # For logistic regression, use coefficients
            importance = np.abs(model.coef_[0])
        else:
            # For tree-based models, use feature importance
            importance = model.feature_importances_
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        # Create horizontal bar chart
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Feature Importance - {model_type.upper()} Model',
            color='Importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Importance Score",
            yaxis_title="Features"
        )
        
        # Save plot
        fig.write_image(save_path)
        return save_path, fig
        
    except Exception as e:
        print(f"Error generating feature importance plot: {e}")
        return None, None

def create_prediction_distribution(predictions, save_path="reports/prediction_distribution.png"):
    """Create prediction distribution visualization"""
    try:
        fig = px.histogram(
            x=predictions,
            nbins=30,
            title="Prediction Distribution",
            labels={'x': 'Risk Probability', 'y': 'Frequency'},
            color_discrete_sequence=['#1f77b4']
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Risk Probability",
            yaxis_title="Frequency"
        )
        
        # Add vertical line for mean
        mean_pred = np.mean(predictions)
        fig.add_vline(x=mean_pred, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_pred:.3f}")
        
        # Save plot
        fig.write_image(save_path)
        return save_path, fig
        
    except Exception as e:
        print(f"Error generating prediction distribution: {e}")
        return None, None

def create_risk_heatmap(risk_data, save_path="reports/risk_heatmap.png"):
    """Create risk heatmap visualization"""
    try:
        # Create correlation matrix for risk factors
        fig = px.imshow(
            risk_data.corr(),
            title="Risk Factors Correlation Heatmap",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Risk Factors",
            yaxis_title="Risk Factors"
        )
        
        # Save plot
        fig.write_image(save_path)
        return save_path, fig
        
    except Exception as e:
        print(f"Error generating risk heatmap: {e}")
        return None, None

def create_model_comparison_plot(model_results, save_path="reports/model_comparison.png"):
    """Create model comparison visualization"""
    try:
        models = list(model_results.keys())
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # Create grouped bar chart
        fig = go.Figure()
        
        for i, metric in enumerate(metrics):
            values = [model_results[model][metric] for model in models]
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            height=500
        )
        
        # Save plot
        fig.write_image(save_path)
        return save_path, fig
        
    except Exception as e:
        print(f"Error generating model comparison plot: {e}")
        return None, None

def create_risk_timeline_plot(historical_data, save_path="reports/risk_timeline.png"):
    """Create risk timeline visualization"""
    try:
        fig = px.line(
            historical_data,
            x='Date',
            y='Risk_Score',
            title="Risk Score Timeline",
            labels={'Risk_Score': 'Risk Score', 'Date': 'Date'},
            color_discrete_sequence=['#ff7f0e']
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Risk Score"
        )
        
        # Add threshold lines
        fig.add_hline(y=0.3, line_dash="dash", line_color="green", 
                     annotation_text="Low Risk Threshold")
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                     annotation_text="High Risk Threshold")
        
        # Save plot
        fig.write_image(save_path)
        return save_path, fig
        
    except Exception as e:
        print(f"Error generating risk timeline plot: {e}")
        return None, None
