# Utility functions for the Diabetes Prediction App

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime
import json

from config import VALIDATION_CONFIG, FEATURE_CONFIG

logger = logging.getLogger(__name__)

def validate_medical_inputs(inputs: Dict[str, float]) -> List[str]:
    """
    Validate user inputs for medical reasonableness
    
    Args:
        inputs: Dictionary of input values
        
    Returns:
        List of validation warnings
    """
    warnings = []
    
    # Glucose validation
    glucose = inputs.get('glucose', 0)
    if glucose > 0:
        if glucose < VALIDATION_CONFIG.glucose_normal_range[0]:
            warnings.append("⚠️ Glucose level seems low (normal fasting: 70-100 mg/dL)")
        elif glucose > 140:
            warnings.append("⚠️ Glucose level is elevated (normal fasting: 70-100 mg/dL)")
        elif glucose > 126:
            warnings.append("⚠️ Glucose level indicates possible diabetes (>126 mg/dL)")
    
    # Blood pressure validation
    bp = inputs.get('bp', 0)
    if bp > 0:
        if bp < VALIDATION_CONFIG.bp_normal_range[0]:
            warnings.append("⚠️ Blood pressure seems low (normal diastolic: 60-80 mmHg)")
        elif bp > 90:
            warnings.append("⚠️ Blood pressure is high (normal diastolic: 60-80 mmHg)")
        elif bp > VALIDATION_CONFIG.bp_normal_range[1]:
            warnings.append("⚠️ Blood pressure is elevated (normal diastolic: 60-80 mmHg)")
    
    # BMI validation
    bmi = inputs.get('bmi', 0)
    if bmi > 0:
        if bmi < 18.5:
            warnings.append("⚠️ BMI indicates underweight (normal: 18.5-24.9)")
        elif bmi > 30:
            warnings.append("⚠️ BMI indicates obesity (normal: 18.5-24.9)")
        elif bmi > 25:
            warnings.append("⚠️ BMI indicates overweight (normal: 18.5-24.9)")
    
    # Age validation
    age = inputs.get('age', 0)
    if age > 65:
        warnings.append("ℹ️ Advanced age is a risk factor for diabetes")
    
    # Pregnancy validation
    pregnancies = inputs.get('pregnancies', 0)
    if pregnancies > 5:
        warnings.append("ℹ️ Multiple pregnancies can increase diabetes risk")
    
    # Insulin validation
    insulin = inputs.get('insulin', 0)
    if insulin > 200:
        warnings.append("⚠️ High insulin levels may indicate insulin resistance")
    
    return warnings

def get_risk_interpretation(probability: float) -> Tuple[str, str, str]:
    """
    Provide detailed risk interpretation based on probability
    
    Args:
        probability: Prediction probability
        
    Returns:
        Tuple of (risk_level, description, emoji)
    """
    if probability < 0.2:
        return (
            "Very Low Risk", 
            "Your diabetes risk appears to be very low based on the provided parameters. Continue maintaining healthy lifestyle habits.",
            "🟢"
        )
    elif probability < 0.4:
        return (
            "Low Risk", 
            "Your diabetes risk appears to be low. Continue with regular health monitoring and maintain healthy habits.",
            "🟢"
        )
    elif probability < 0.6:
        return (
            "Moderate Risk", 
            "You have a moderate risk of diabetes. Consider lifestyle modifications, regular monitoring, and consult with healthcare providers.",
            "🟡"
        )
    elif probability < 0.8:
        return (
            "High Risk", 
            "You have a high risk of diabetes. Please consult with a healthcare provider for proper evaluation and consider preventive measures.",
            "🔴"
        )
    else:
        return (
            "Very High Risk", 
            "You have a very high risk of diabetes. Immediate consultation with a healthcare provider is strongly recommended.",
            "🔴"
        )

def get_bmi_category(bmi: float) -> str:
    """Get BMI category based on standard classifications"""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal Weight"
    elif bmi < 30:
        return "Overweight"
    elif bmi < 35:
        return "Obese Class I"
    elif bmi < 40:
        return "Obese Class II"
    else:
        return "Obese Class III"

def get_age_group(age: int) -> str:
    """Get age group classification"""
    if age < 25:
        return "Young Adult"
    elif age < 35:
        return "Adult"
    elif age < 45:
        return "Middle-aged Adult"
    elif age < 65:
        return "Older Adult"
    else:
        return "Senior"

def get_glucose_status(glucose: float) -> str:
    """Get glucose status based on medical standards"""
    if glucose < 70:
        return "Hypoglycemic"
    elif glucose < 100:
        return "Normal"
    elif glucose < 126:
        return "Pre-diabetic Range"
    else:
        return "Diabetic Range"

def get_blood_pressure_status(bp: float) -> str:
    """Get blood pressure status"""
    if bp < 60:
        return "Low"
    elif bp < 80:
        return "Normal"
    elif bp < 90:
        return "High Normal"
    else:
        return "High"

def create_radar_chart(inputs: Dict[str, float]) -> go.Figure:
    """
    Create a radar chart showing input parameters
    
    Args:
        inputs: Dictionary of input values
        
    Returns:
        Plotly figure object
    """
    categories = [
        'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
        'Insulin', 'BMI', 'Pedigree Function', 'Age'
    ]
    
    # Normalize values to 0-1 scale for visualization
    max_values = {
        'pregnancies': 17,
        'glucose': 200,
        'bp': 122,
        'skin': 99,
        'insulin': 846,
        'bmi': 67.1,
        'pedigree': 2.5,
        'age': 90
    }
    
    normalized_values = [
        inputs['pregnancies'] / max_values['pregnancies'],
        inputs['glucose'] / max_values['glucose'],
        inputs['bp'] / max_values['bp'],
        inputs['skin'] / max_values['skin'],
        inputs['insulin'] / max_values['insulin'],
        inputs['bmi'] / max_values['bmi'],
        inputs['pedigree'] / max_values['pedigree'],
        inputs['age'] / max_values['age']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=categories,
        fill='toself',
        name='Your Profile',
        line_color='rgb(31, 119, 180)',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='array',
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['0%', '25%', '50%', '75%', '100%']
            )
        ),
        showlegend=True,
        title={
            'text': "Your Health Profile Radar Chart",
            'x': 0.5,
            'xanchor': 'center'
        },
        font=dict(size=12)
    )
    
    return fig

def create_risk_gauge(probability: float) -> go.Figure:
    """
    Create a gauge chart showing risk level
    
    Args:
        probability: Risk probability
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Diabetes Risk Level (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "lightgreen"},
                {'range': [20, 40], 'color': "yellow"},
                {'range': [40, 60], 'color': "orange"},
                {'range': [60, 80], 'color': "red"},
                {'range': [80, 100], 'color': "darkred"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def generate_recommendations(inputs: Dict[str, float], probability: float) -> List[str]:
    """
    Generate personalized recommendations based on inputs and risk level
    
    Args:
        inputs: Dictionary of input values
        probability: Risk probability
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # General recommendations
    recommendations.append("🏃‍♂️ Maintain regular physical activity (150 minutes/week)")
    recommendations.append("🥗 Follow a balanced, low-glycemic diet")
    recommendations.append("💧 Stay well hydrated")
    recommendations.append("😴 Ensure adequate sleep (7-9 hours/night)")
    
    # Specific recommendations based on inputs
    if inputs['glucose'] > 100:
        recommendations.append("🍯 Monitor blood glucose levels regularly")
        recommendations.append("🚫 Limit refined sugars and processed foods")
    
    if inputs['bmi'] > 25:
        recommendations.append("⚖️ Consider weight management strategies")
        recommendations.append("🍽️ Practice portion control")
    
    if inputs['bp'] > 80:
        recommendations.append("💓 Monitor blood pressure regularly")
        recommendations.append("🧂 Reduce sodium intake")
    
    if inputs['age'] > 45:
        recommendations.append("🩺 Schedule regular health screenings")
        recommendations.append("💊 Discuss preventive measures with your doctor")
    
    if probability > 0.6:
        recommendations.append("🏥 Consult with a healthcare provider soon")
        recommendations.append("📊 Consider diabetes screening tests")
        recommendations.append("👨‍⚕️ Discuss family history with your doctor")
    
    if inputs['pregnancies'] > 3:
        recommendations.append("🤱 Monitor for gestational diabetes in future pregnancies")
    
    return recommendations

def export_prediction_data(history: List[Dict]) -> str:
    """
    Export prediction history to CSV format
    
    Args:
        history: List of prediction history entries
        
    Returns:
        CSV string
    """
    if not history:
        return ""
    
    df_data = []
    for entry in history:
        row = {
            'Timestamp': entry['timestamp'],
            'Prediction': 'High Risk' if entry['prediction'] == 1 else 'Low Risk',
            'Probability': entry['probability'],
            **entry['inputs']
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    return df.to_csv(index=False)

def calculate_risk_factors(inputs: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate and categorize risk factors
    
    Args:
        inputs: Dictionary of input values
        
    Returns:
        Dictionary with risk factor analysis
    """
    risk_factors = {
        'high_risk': [],
        'moderate_risk': [],
        'protective_factors': []
    }
    
    # High risk factors
    if inputs['glucose'] > 126:
        risk_factors['high_risk'].append("🔴 Diabetic-range glucose levels")
    elif inputs['glucose'] > 100:
        risk_factors['moderate_risk'].append("🟡 Pre-diabetic glucose levels")
    
    if inputs['bmi'] > 30:
        risk_factors['high_risk'].append("🔴 Obesity (BMI > 30)")
    elif inputs['bmi'] > 25:
        risk_factors['moderate_risk'].append("🟡 Overweight (BMI 25-30)")
    
    if inputs['age'] > 65:
        risk_factors['high_risk'].append("🔴 Advanced age (>65 years)")
    elif inputs['age'] > 45:
        risk_factors['moderate_risk'].append("🟡 Age factor (45-65 years)")
    
    if inputs['pedigree'] > 1.0:
        risk_factors['high_risk'].append("🔴 Strong genetic predisposition")
    elif inputs['pedigree'] > 0.5:
        risk_factors['moderate_risk'].append("🟡 Moderate genetic predisposition")
    
    if inputs['pregnancies'] > 5:
        risk_factors['moderate_risk'].append("🟡 Multiple pregnancies")
    
    if inputs['bp'] > 90:
        risk_factors['moderate_risk'].append("🟡 Elevated blood pressure")
    
    # Protective factors
    if inputs['bmi'] < 25 and inputs['bmi'] > 18.5:
        risk_factors['protective_factors'].append("🟢 Normal BMI")
    
    if inputs['glucose'] < 100:
        risk_factors['protective_factors'].append("🟢 Normal glucose levels")
    
    if inputs['age'] < 35:
        risk_factors['protective_factors'].append("🟢 Young age")
    
    return risk_factors

def format_medical_value(value: float, unit: str, decimal_places: int = 1) -> str:
    """
    Format medical values with appropriate units and precision
    
    Args:
        value: Numeric value
        unit: Unit string
        decimal_places: Number of decimal places
        
    Returns:
        Formatted string
    """
    if decimal_places == 0:
        return f"{int(value)} {unit}"
    else:
        return f"{value:.{decimal_places}f} {unit}"

def log_prediction(inputs: Dict[str, float], prediction: int, probability: float):
    """
    Log prediction for monitoring and analysis
    
    Args:
        inputs: Input values
        prediction: Model prediction
        probability: Prediction probability
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'inputs': inputs,
        'prediction': prediction,
        'probability': probability
    }
    
    logger.info(f"Prediction made: {log_entry}")