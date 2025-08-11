import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

def create_risk_summary_card(risk_score, risk_level, confidence=0.85):
    """Create a summary card for risk assessment"""
    
    # Determine color based on risk level
    if risk_level == "Low Risk":
        color = "#28a745"
        bg_color = "#d4edda"
        border_color = "#c3e6cb"
    elif risk_level == "Medium Risk":
        color = "#ffc107"
        bg_color = "#fff3cd"
        border_color = "#ffeaa7"
    else:  # High Risk
        color = "#dc3545"
        bg_color = "#f8d7da"
        border_color = "#f5c6cb"
    
    # Create the card using HTML
    card_html = f"""
    <div style="
        background-color: {bg_color};
        border: 2px solid {border_color};
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
    ">
        <h3 style="color: {color}; margin: 0 0 10px 0;">Risk Assessment Summary</h3>
        <div style="
            font-size: 48px;
            font-weight: bold;
            color: {color};
            margin: 20px 0;
        ">{risk_score:.3f}</div>
        <div style="
            font-size: 24px;
            color: {color};
            margin: 10px 0;
        ">{risk_level}</div>
        <div style="
            font-size: 16px;
            color: #6c757d;
            margin: 10px 0;
        ">Confidence: {confidence:.1%}</div>
    </div>
    """
    
    return card_html

def create_prediction_gauge(value, title, color="blue"):
    """Create a gauge chart for prediction visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 0.5},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.7
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_model_comparison_chart(predictions_dict):
    """Create a comparison chart between multiple models"""
    models = list(predictions_dict.keys())
    values = list(predictions_dict.values())
    
    # Generate colors
    colors = px.colors.qualitative.Set3[:len(models)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models, 
            y=values, 
            marker_color=colors, 
            text=[f'{v:.3f}' for v in values], 
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Model Predictions Comparison",
        xaxis_title="Model",
        yaxis_title="Risk Probability",
        yaxis_range=[0, 1],
        height=400
    )
    
    return fig

def create_risk_breakdown_chart(risk_score):
    """Create a risk breakdown visualization"""
    # Define risk categories
    risk_categories = {
        'Very Low': (0, 0.2),
        'Low': (0.2, 0.4),
        'Medium': (0.4, 0.6),
        'High': (0.6, 0.8),
        'Very High': (0.8, 1.0)
    }
    
    # Determine current risk category
    current_category = None
    for category, (min_val, max_val) in risk_categories.items():
        if min_val <= risk_score < max_val:
            current_category = category
            break
    
    # Create risk breakdown chart
    categories = list(risk_categories.keys())
    values = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal segments
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=['lightgreen', 'green', 'yellow', 'orange', 'red'],
            text=[f'{v*100:.0f}%' for v in values],
            textposition='auto'
        )
    ])
    
    # Add marker for current risk
    fig.add_trace(go.Scatter(
        x=[current_category],
        y=[0.25],
        mode='markers',
        name='Current Risk',
        marker=dict(symbol='diamond', size=20, color='black'),
        text=[f'Your Risk: {risk_score:.3f}'],
        textposition='top center'
    ))
    
    fig.update_layout(
        title="Risk Level Breakdown",
        xaxis_title="Risk Category",
        yaxis_title="Risk Range",
        height=400
    )
    
    return fig

def create_feature_importance_chart(feature_names, importance_scores, title="Feature Importance"):
    """Create a feature importance visualization"""
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=title,
        color='Importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Importance Score",
        yaxis_title="Features"
    )
    
    return fig

def create_prediction_distribution_chart(predictions, title="Prediction Distribution"):
    """Create a prediction distribution visualization"""
    fig = px.histogram(
        x=predictions,
        nbins=30,
        title=title,
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
    
    return fig

def create_confidence_interval_chart(predictions_dict):
    """Create confidence interval visualization"""
    models = list(predictions_dict.keys())
    values = list(predictions_dict.values())
    
    fig = go.Figure()
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=models,
        y=values,
        mode='markers',
        name='Prediction',
        marker=dict(size=15, color=['#1f77b4', '#ff7f0e'])
    ))
    
    # Add confidence intervals (simplified)
    for i, (model, value) in enumerate(predictions_dict.items()):
        ci_lower = max(0, value - 0.1)
        ci_upper = min(1, value + 0.1)
        
        fig.add_trace(go.Scatter(
            x=[model, model],
            y=[ci_lower, ci_upper],
            mode='lines',
            name='Confidence Interval',
            line=dict(color=['#1f77b4', '#ff7f0e'][i], width=3),
            showlegend=False
        ))
    
    fig.update_layout(
        title="Prediction Confidence Intervals",
        yaxis_title="Risk Probability",
        yaxis_range=[0, 1],
        height=400
    )
    
    return fig

def create_risk_factors_chart(risk_factors_dict):
    """Create a risk factors analysis chart"""
    factors = list(risk_factors_dict.keys())
    values = list(risk_factors_dict.values())
    
    # Normalize values to 0-1 scale
    normalized_values = [min(1, max(0, v)) for v in values]
    
    fig = px.bar(
        x=factors,
        y=normalized_values,
        title="Individual Risk Factors",
        labels={'x': 'Risk Factor', 'y': 'Risk Level'},
        color=normalized_values,
        color_continuous_scale='RdYlGn_r'
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Risk Factor",
        yaxis_title="Risk Level",
        xaxis_tickangle=-45
    )
    
    return fig

def display_prediction_summary(predictions_dict, risk_level, confidence=0.85):
    """Display a comprehensive prediction summary"""
    
    # Calculate average risk
    avg_risk = np.mean(list(predictions_dict.values()))
    
    # Display risk summary card
    st.markdown(create_risk_summary_card(avg_risk, risk_level, confidence), unsafe_allow_html=True)
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Model comparison
        st.plotly_chart(create_model_comparison_chart(predictions_dict), use_container_width=True)
        
        # Risk breakdown
        st.plotly_chart(create_risk_breakdown_chart(avg_risk), use_container_width=True)
    
    with col2:
        # Confidence intervals
        st.plotly_chart(create_confidence_interval_chart(predictions_dict), use_container_width=True)
        
        # Individual risk gauges
        for model, pred in predictions_dict.items():
            st.plotly_chart(create_prediction_gauge(pred, f"{model} Risk"), use_container_width=True)
    
    # Display detailed metrics
    st.write("### 📈 Detailed Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Risk", f"{avg_risk:.3f}")
    with col2:
        st.metric("Risk Level", risk_level)
    with col3:
        st.metric("Confidence", f"{confidence:.1%}")
    
    # Model agreement
    if len(predictions_dict) > 1:
        values = list(predictions_dict.values())
        agreement = 1 - (max(values) - min(values))
        st.metric("Model Agreement", f"{agreement:.3f}")
