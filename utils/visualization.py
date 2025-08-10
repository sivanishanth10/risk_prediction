import matplotlib.pyplot as plt
import shap
import pickle
import pandas as pd
import os

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
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    except Exception as e:
        print(f"Error generating SHAP plot: {e}")
        return None
