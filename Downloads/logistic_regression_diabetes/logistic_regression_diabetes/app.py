
import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import logging
import hashlib
import time
from typing import Dict, List, Tuple, Any, Optional
import warnings
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Configure warnings and logging for Streamlit Cloud
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import custom modules with error handling for Streamlit Cloud
try:
    from config import MODEL_CONFIG, UI_CONFIG, FEATURE_CONFIG, VALIDATION_CONFIG
    from utils import (
        validate_medical_inputs, get_risk_interpretation, get_bmi_category,
        get_age_group, get_glucose_status, create_radar_chart, create_risk_gauge,
        generate_recommendations, calculate_risk_factors, format_medical_value
    )
    from data_analysis import DataAnalyzer, display_data_analysis_page
    from model_explainer import ModelExplainer, display_model_explanation_page
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.info("Some advanced features may not be available. The basic prediction functionality will still work.")
    
    # Fallback configurations
    class FallbackConfig:
        def __init__(self):
            self.page_title = "Diabetes Risk Predictor"
            self.page_icon = "🩺"
            self.layout = "wide"
    
    UI_CONFIG = FallbackConfig()
    
    # Fallback functions
    def validate_medical_inputs(inputs):
        return []
    
    def get_risk_interpretation(prob):
        if prob < 0.3:
            return "Low Risk", "Low diabetes risk detected.", "🟢"
        elif prob < 0.6:
            return "Moderate Risk", "Moderate diabetes risk detected.", "🟡"
        else:
            return "High Risk", "High diabetes risk detected.", "🔴"

# Page configuration with enhanced settings
st.set_page_config(
    page_title=UI_CONFIG.page_title,
    page_icon=UI_CONFIG.page_icon,
    layout=UI_CONFIG.layout,
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/diabetes-predictor',
        'Report a bug': 'https://github.com/your-repo/diabetes-predictor/issues',
        'About': """
        # Advanced Diabetes Risk Predictor
        
        This application uses machine learning to assess diabetes risk based on key health indicators.
        
        **Version:** 2.0.0
        **Author:** Enhanced by AI Assistant
        **License:** MIT
        """
    }
)

# Enhanced CSS for better styling and responsiveness
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    /* Risk level cards */
    .risk-very-high {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(220,38,38,0.3);
        border: 2px solid rgba(220,38,38,0.5);
        animation: pulse 2s infinite;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(255,107,107,0.3);
        border: 2px solid rgba(255,107,107,0.5);
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(251,191,36,0.3);
        border: 2px solid rgba(251,191,36,0.5);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(81,207,102,0.3);
        border: 2px solid rgba(81,207,102,0.5);
    }
    
    .risk-very-low {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(34,197,94,0.3);
        border: 2px solid rgba(34,197,94,0.5);
    }
    
    /* Pulse animation for very high risk */
    @keyframes pulse {
        0% { box-shadow: 0 8px 32px rgba(220,38,38,0.3); }
        50% { box-shadow: 0 8px 32px rgba(220,38,38,0.6); }
        100% { box-shadow: 0 8px 32px rgba(220,38,38,0.3); }
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: rgba(34, 197, 94, 0.1);
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Button enhancements */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .metric-card, .risk-high, .risk-low, .risk-moderate, .risk-very-high, .risk-very-low {
            padding: 1rem;
            margin: 0.25rem 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# Enhanced session state initialization
def initialize_session_state():
    """Initialize all session state variables"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'theme': 'light',
            'show_advanced_metrics': False,
            'auto_save_predictions': True,
            'show_explanations': True
        }
    
    if 'app_stats' not in st.session_state:
        st.session_state.app_stats = {
            'total_predictions': 0,
            'session_start_time': datetime.now(),
            'high_risk_predictions': 0,
            'low_risk_predictions': 0
        }
    
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None

# Initialize session state
initialize_session_state()

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler with enhanced error handling and caching"""
    try:
        model_path = Path("logistic_model.pkl")
        scaler_path = Path("scaler.pkl")
        
        # Check if model files exist
        if not model_path.exists() or not scaler_path.exists():
            with st.spinner("🔄 Training model for the first time... This may take a moment."):
                # Import and run training
                import subprocess
                result = subprocess.run([sys.executable, "train_model.py"], 
                                      capture_output=True, text=True, cwd=".")
                
                if result.returncode != 0:
                    st.error(f"❌ Model training failed: {result.stderr}")
                    logger.error(f"Model training failed: {result.stderr}")
                    return None, None
                
            st.success("✅ Model trained successfully!")
        
        # Load model with security checks
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        # Validate loaded objects
        if not hasattr(model, 'predict') or not hasattr(scaler, 'transform'):
            raise ValueError("Invalid model or scaler objects")
        
        logger.info("Model and scaler loaded successfully")
        st.session_state.model_loaded = True
        
        return model, scaler
        
    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        st.error(error_msg)
        logger.error(f"Error loading model: {str(e)}")
        st.session_state.model_loaded = False
        return None, None

def validate_model_inputs(inputs: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Enhanced input validation with security checks
    
    Args:
        inputs: Dictionary of input values
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check for required inputs
    required_fields = ['pregnancies', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    for field in required_fields:
        if field not in inputs:
            errors.append(f"Missing required field: {field}")
    
    # Range validation
    ranges = {
        'pregnancies': (0, 17),
        'glucose': (0, 300),  # Extended range for edge cases
        'bp': (0, 200),
        'skin': (0, 150),
        'insulin': (0, 1000),
        'bmi': (10, 80),
        'pedigree': (0, 3),
        'age': (18, 120)
    }
    
    for field, (min_val, max_val) in ranges.items():
        if field in inputs:
            value = inputs[field]
            if not isinstance(value, (int, float)):
                errors.append(f"{field} must be a number")
            elif value < min_val or value > max_val:
                errors.append(f"{field} must be between {min_val} and {max_val}")
            elif np.isnan(value) or np.isinf(value):
                errors.append(f"{field} contains invalid value")
    
    return len(errors) == 0, errors

def create_enhanced_input_form() -> Dict[str, float]:
    """Create an enhanced input form with better UX"""
    st.markdown("### 📝 Patient Information Input")
    
    # Create tabs for different input categories
    tab1, tab2, tab3 = st.tabs(["🔢 Basic Info", "🧪 Lab Results", "📊 Physical Metrics"])
    
    inputs = {}
    
    with tab1:
        st.markdown("#### Basic Demographics")
        col1, col2 = st.columns(2)
        
        with col1:
            inputs['pregnancies'] = st.slider(
                "👶 Number of Pregnancies",
                min_value=0, max_value=17, value=3,
                help=FEATURE_CONFIG.feature_descriptions['pregnancies']
            )
            
            inputs['age'] = st.slider(
                "🎂 Age (years)",
                min_value=21, max_value=90, value=33,
                help=FEATURE_CONFIG.feature_descriptions['age']
            )
        
        with col2:
            # Age group indicator
            age_group = get_age_group(inputs['age'])
            st.info(f"**Age Group:** {age_group}")
            
            # Pregnancy risk indicator
            if inputs['pregnancies'] > 5:
                st.warning("⚠️ Multiple pregnancies may increase diabetes risk")
    
    with tab2:
        st.markdown("#### Laboratory Results")
        col1, col2 = st.columns(2)
        
        with col1:
            inputs['glucose'] = st.slider(
                "🍯 Glucose Level (mg/dL)",
                min_value=0, max_value=200, value=120,
                help=FEATURE_CONFIG.feature_descriptions['glucose']
            )
            
            inputs['insulin'] = st.slider(
                "💉 Insulin Level (μU/mL)",
                min_value=0, max_value=846, value=79,
                help=FEATURE_CONFIG.feature_descriptions['insulin']
            )
        
        with col2:
            # Glucose status indicator
            glucose_status = get_glucose_status(inputs['glucose'])
            if glucose_status == "Normal":
                st.success(f"**Glucose Status:** {glucose_status}")
            elif glucose_status == "Pre-diabetic Range":
                st.warning(f"**Glucose Status:** {glucose_status}")
            else:
                st.error(f"**Glucose Status:** {glucose_status}")
            
            # Insulin level indicator
            if inputs['insulin'] > 200:
                st.warning("⚠️ Elevated insulin levels detected")
    
    with tab3:
        st.markdown("#### Physical Measurements")
        col1, col2 = st.columns(2)
        
        with col1:
            inputs['bp'] = st.slider(
                "💓 Blood Pressure (mmHg)",
                min_value=0, max_value=122, value=70,
                help=FEATURE_CONFIG.feature_descriptions['blood_pressure']
            )
            
            inputs['skin'] = st.slider(
                "📏 Skin Thickness (mm)",
                min_value=0, max_value=99, value=20,
                help=FEATURE_CONFIG.feature_descriptions['skin_thickness']
            )
            
            inputs['bmi'] = st.slider(
                "⚖️ BMI",
                min_value=0.0, max_value=67.1, value=32.0,
                help=FEATURE_CONFIG.feature_descriptions['bmi']
            )
        
        with col2:
            # BMI category indicator
            bmi_category = get_bmi_category(inputs['bmi'])
            if bmi_category == "Normal Weight":
                st.success(f"**BMI Category:** {bmi_category}")
            elif bmi_category in ["Overweight", "Obese Class I"]:
                st.warning(f"**BMI Category:** {bmi_category}")
            else:
                st.error(f"**BMI Category:** {bmi_category}")
            
            # Blood pressure status
            if inputs['bp'] < 60:
                st.warning("⚠️ Low blood pressure detected")
            elif inputs['bp'] > 90:
                st.error("⚠️ High blood pressure detected")
            else:
                st.success("✅ Normal blood pressure range")
    
    # Genetic factors
    st.markdown("#### 🧬 Genetic Factors")
    inputs['pedigree'] = st.slider(
        "Diabetes Pedigree Function",
        min_value=0.0, max_value=2.5, value=0.47, step=0.01,
        help=FEATURE_CONFIG.feature_descriptions['pedigree']
    )
    
    if inputs['pedigree'] > 1.0:
        st.warning("⚠️ Strong genetic predisposition detected")
    elif inputs['pedigree'] > 0.5:
        st.info("ℹ️ Moderate genetic predisposition")
    
    return inputs

def make_enhanced_prediction(inputs: Dict[str, float], model, scaler) -> Dict[str, Any]:
    """
    Make enhanced prediction with comprehensive analysis
    
    Args:
        inputs: Dictionary of input values
        model: Trained model
        scaler: Fitted scaler
        
    Returns:
        Dictionary with prediction results and analysis
    """
    try:
        # Validate inputs
        is_valid, errors = validate_model_inputs(inputs)
        if not is_valid:
            return {'error': f"Input validation failed: {'; '.join(errors)}"}
        
        # Prepare input data
        input_df = pd.DataFrame([[
            inputs['pregnancies'], inputs['glucose'], inputs['bp'], inputs['skin'],
            inputs['insulin'], inputs['bmi'], inputs['pedigree'], inputs['age']
        ]], columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ])
        
        # Scale input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Get risk interpretation
        risk_level, risk_description, risk_emoji = get_risk_interpretation(probabilities[1])
        
        # Calculate risk factors
        risk_factors = calculate_risk_factors(inputs)
        
        # Generate recommendations
        recommendations = generate_recommendations(inputs, probabilities[1])
        
        # Medical validation warnings
        medical_warnings = validate_medical_inputs(inputs)
        
        # Prepare comprehensive result
        result = {
            'prediction': int(prediction),
            'probability_no_diabetes': float(probabilities[0]),
            'probability_diabetes': float(probabilities[1]),
            'risk_level': risk_level,
            'risk_description': risk_description,
            'risk_emoji': risk_emoji,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'medical_warnings': medical_warnings,
            'input_summary': {
                'bmi_category': get_bmi_category(inputs['bmi']),
                'age_group': get_age_group(inputs['age']),
                'glucose_status': get_glucose_status(inputs['glucose']),
                'bp_status': 'Normal' if 60 <= inputs['bp'] <= 80 else 'Abnormal'
            },
            'confidence_score': max(probabilities) * 100,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update session statistics
        st.session_state.app_stats['total_predictions'] += 1
        if prediction == 1:
            st.session_state.app_stats['high_risk_predictions'] += 1
        else:
            st.session_state.app_stats['low_risk_predictions'] += 1
        
        st.session_state.last_prediction = result
        
        return result
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return {'error': f"Prediction failed: {str(e)}"}

def display_prediction_results(result: Dict[str, Any], inputs: Dict[str, float]):
    """Display comprehensive prediction results"""
    if 'error' in result:
        st.error(result['error'])
        return
    
    # Main prediction result
    st.markdown("### 🎯 Prediction Results")
    
    probability = result['probability_diabetes']
    risk_level = result['risk_level']
    
    # Choose appropriate CSS class based on risk level
    if risk_level == "Very High Risk":
        css_class = "risk-very-high"
    elif risk_level == "High Risk":
        css_class = "risk-high"
    elif risk_level == "Moderate Risk":
        css_class = "risk-moderate"
    elif risk_level == "Low Risk":
        css_class = "risk-low"
    else:
        css_class = "risk-very-low"
    
    st.markdown(f"""
    <div class="{css_class}">
        <h2>{result['risk_emoji']} {risk_level}</h2>
        <h3>Probability: {probability:.1%}</h3>
        <h4>Confidence: {result['confidence_score']:.1f}%</h4>
        <p>{result['risk_description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed analysis in columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk factors analysis
        st.markdown("#### 🎯 Risk Factor Analysis")
        
        risk_factors = result['risk_factors']
        
        if risk_factors['high_risk']:
            st.markdown("**🔴 High Risk Factors:**")
            for factor in risk_factors['high_risk']:
                st.write(f"• {factor}")
        
        if risk_factors['moderate_risk']:
            st.markdown("**🟡 Moderate Risk Factors:**")
            for factor in risk_factors['moderate_risk']:
                st.write(f"• {factor}")
        
        if risk_factors['protective_factors']:
            st.markdown("**🟢 Protective Factors:**")
            for factor in risk_factors['protective_factors']:
                st.write(f"• {factor}")
        
        if not any([risk_factors['high_risk'], risk_factors['moderate_risk']]):
            st.success("✅ No major risk factors identified")
    
    with col2:
        # Recommendations
        st.markdown("#### 💡 Personalized Recommendations")
        recommendations = result['recommendations']
        
        for i, rec in enumerate(recommendations[:8], 1):  # Limit to 8 recommendations
            st.write(f"{i}. {rec}")
    
    # Medical warnings
    if result['medical_warnings']:
        st.markdown("#### ⚠️ Medical Validation Warnings")
        for warning in result['medical_warnings']:
            st.warning(warning)
    
    # Input summary
    st.markdown("#### 📋 Input Summary")
    summary = result['input_summary']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("BMI Category", summary['bmi_category'])
    with col2:
        st.metric("Age Group", summary['age_group'])
    with col3:
        st.metric("Glucose Status", summary['glucose_status'])
    with col4:
        st.metric("BP Status", summary['bp_status'])
    
    # Visualizations
    st.markdown("#### 📊 Visual Analysis")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Radar chart
        radar_fig = create_radar_chart(inputs)
        st.plotly_chart(radar_fig, use_container_width=True)
    
    with viz_col2:
        # Risk gauge
        gauge_fig = create_risk_gauge(probability)
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    # Save prediction to history
    if st.session_state.user_preferences['auto_save_predictions']:
        save_prediction_to_history(inputs, result)
        st.success("✅ Prediction saved to history")

def save_prediction_to_history(inputs: Dict[str, float], result: Dict[str, Any]):
    """Save prediction to history with enhanced data"""
    history_entry = {
        'timestamp': result['timestamp'],
        'inputs': inputs,
        'prediction': result['prediction'],
        'probability': result['probability_diabetes'],
        'risk_level': result['risk_level'],
        'confidence_score': result['confidence_score'],
        'session_id': hashlib.md5(str(st.session_state.app_stats['session_start_time']).encode()).hexdigest()[:8]
    }
    
    st.session_state.prediction_history.append(history_entry)
    
    # Limit history size to prevent memory issues
    if len(st.session_state.prediction_history) > 100:
        st.session_state.prediction_history = st.session_state.prediction_history[-100:]

def create_advanced_analytics_dashboard():
    """Create advanced analytics dashboard"""
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    if not st.session_state.prediction_history:
        st.info("No predictions made yet. Make some predictions to see analytics!")
        return
    
    # Convert history to DataFrame
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    # Summary statistics
    st.markdown("#### 📈 Session Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(history_df))
    with col2:
        high_risk_count = len(history_df[history_df['prediction'] == 1])
        st.metric("High Risk Cases", high_risk_count)
    with col3:
        avg_probability = history_df['probability'].mean()
        st.metric("Avg Risk Probability", f"{avg_probability:.1%}")
    with col4:
        avg_confidence = history_df['confidence_score'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    # Time series analysis
    if len(history_df) > 1:
        st.markdown("#### 📈 Prediction Trends")
        
        # Convert timestamp to datetime
        history_df['datetime'] = pd.to_datetime(history_df['timestamp'])
        
        # Create time series plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=history_df['datetime'],
            y=history_df['probability'],
            mode='lines+markers',
            name='Risk Probability',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Risk Probability Over Time',
            xaxis_title='Time',
            yaxis_title='Diabetes Risk Probability',
            yaxis=dict(tickformat='.1%'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk level distribution
    st.markdown("#### 🎯 Risk Level Distribution")
    risk_counts = history_df['risk_level'].value_counts()
    
    fig = go.Figure(data=[
        go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.4,
            marker_colors=['#22c55e', '#51cf66', '#fbbf24', '#ff6b6b', '#dc2626']
        )
    ])
    
    fig.update_layout(
        title='Distribution of Risk Levels',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature analysis
    st.markdown("#### 🔍 Input Feature Analysis")
    
    # Extract input features from history
    feature_data = []
    for entry in st.session_state.prediction_history:
        feature_row = entry['inputs'].copy()
        feature_row['risk_level'] = entry['risk_level']
        feature_row['prediction'] = entry['prediction']
        feature_data.append(feature_row)
    
    feature_df = pd.DataFrame(feature_data)
    
    # Feature correlation with risk
    numeric_features = ['pregnancies', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    
    if len(feature_df) > 5:  # Need sufficient data for correlation
        correlations = []
        for feature in numeric_features:
            if feature in feature_df.columns:
                corr = feature_df[feature].corr(feature_df['prediction'])
                correlations.append({'Feature': feature.title(), 'Correlation': corr})
        
        corr_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(
                x=corr_df['Feature'],
                y=corr_df['Correlation'],
                marker_color=['#FF6B6B' if x < 0 else '#4ECDC4' for x in corr_df['Correlation']]
            )
        ])
        
        fig.update_layout(
            title='Feature Correlation with Diabetes Risk',
            xaxis_title='Features',
            yaxis_title='Correlation Coefficient',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def save_prediction_history(inputs, prediction, probability):
    """Save prediction to history"""
    history_entry = {
        'timestamp': datetime.now().isoformat(),
        'inputs': inputs,
        'prediction': prediction,
        'probability': probability
    }
    st.session_state.prediction_history.append(history_entry)

def display_model_performance():
    """Display model performance metrics"""
    try:
        # Load test data for performance metrics
        df = pd.read_csv("diabetes.csv")
        
        # Apply same preprocessing as in training
        cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in cols:
            df[col] = df[col].replace(0, df[col].median())
        
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        
        model, scaler = load_model_and_scaler()
        if model and scaler:
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            accuracy = accuracy_score(y, y_pred)
            
            st.subheader("📊 Model Performance")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Accuracy</h3>
                    <h2>{accuracy:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                precision = len(y_pred[(y_pred == 1) & (y == 1)]) / len(y_pred[y_pred == 1]) if len(y_pred[y_pred == 1]) > 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Precision</h3>
                    <h2>{precision:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                recall = len(y_pred[(y_pred == 1) & (y == 1)]) / len(y[y == 1]) if len(y[y == 1]) > 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Recall</h3>
                    <h2>{recall:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Confusion Matrix
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error displaying model performance: {str(e)}")

# Enhanced main application
def main():
    """Main application with enhanced features and navigation"""
    
    # Header with app info
    st.markdown('<h1 class="main-header">🩺 Advanced Diabetes Risk Predictor v2.0</h1>', unsafe_allow_html=True)
    
    # Load model with caching
    model, scaler = load_model_and_scaler()
    if not model or not scaler:
        st.error("❌ Unable to load model. Please check the model files.")
        st.stop()
    
    # Enhanced sidebar with user preferences
    with st.sidebar:
        st.title("🔧 Navigation & Settings")
        
        # Navigation
        page = st.selectbox(
            "Choose a page:",
            [
                "🏠 Prediction",
                "📊 Data Analysis", 
                "🔍 Model Explanation",
                "📈 Analytics Dashboard",
                "📋 Prediction History",
                "⚙️ Settings",
                "ℹ️ About"
            ]
        )
        
        st.divider()
        
        # Quick stats
        st.markdown("#### 📊 Session Stats")
        stats = st.session_state.app_stats
        st.metric("Total Predictions", stats['total_predictions'])
        st.metric("High Risk Cases", stats['high_risk_predictions'])
        
        if stats['total_predictions'] > 0:
            risk_rate = stats['high_risk_predictions'] / stats['total_predictions'] * 100
            st.metric("High Risk Rate", f"{risk_rate:.1f}%")
        
        st.divider()
        
        # User preferences
        st.markdown("#### ⚙️ Quick Settings")
        st.session_state.user_preferences['auto_save_predictions'] = st.checkbox(
            "Auto-save predictions", 
            value=st.session_state.user_preferences['auto_save_predictions']
        )
        
        st.session_state.user_preferences['show_advanced_metrics'] = st.checkbox(
            "Show advanced metrics",
            value=st.session_state.user_preferences['show_advanced_metrics']
        )
        
        st.session_state.user_preferences['show_explanations'] = st.checkbox(
            "Show explanations",
            value=st.session_state.user_preferences['show_explanations']
        )
        
        st.divider()
        
        # Model status
        if st.session_state.model_loaded:
            st.success("✅ Model Ready")
        else:
            st.error("❌ Model Not Loaded")
        
        # Session info
        session_duration = datetime.now() - stats['session_start_time']
        st.info(f"Session: {session_duration.seconds // 60}m {session_duration.seconds % 60}s")
    
    # Page routing with enhanced features
    if page == "🏠 Prediction":
        # Enhanced prediction interface
        inputs = create_enhanced_input_form()
        
        # Prediction button and results
        st.markdown("### 🎯 Risk Assessment")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            predict_button = st.button(
                "🔮 Analyze Diabetes Risk", 
                type="primary", 
                use_container_width=True,
                help="Click to get comprehensive diabetes risk analysis"
            )
        
        with col2:
            if st.button("🔄 Reset Form", use_container_width=True):
                st.experimental_rerun()
        
        with col3:
            if st.button("📊 Quick Stats", use_container_width=True):
                st.info(f"Session predictions: {st.session_state.app_stats['total_predictions']}")
        
        if predict_button:
            with st.spinner("🔍 Analyzing your health data..."):
                result = make_enhanced_prediction(inputs, model, scaler)
                display_prediction_results(result, inputs)
                
                # Show model explanation if enabled
                if st.session_state.user_preferences['show_explanations']:
                    with st.expander("🔍 See Detailed Model Explanation", expanded=False):
                        explainer = ModelExplainer()
                        explanation = explainer.explain_prediction(inputs)
                        if 'error' not in explanation:
                            exp_fig = explainer.create_prediction_explanation_plot(explanation)
                            st.plotly_chart(exp_fig, use_container_width=True)
    
    elif page == "📊 Data Analysis":
        display_data_analysis_page()
    
    elif page == "🔍 Model Explanation":
        display_model_explanation_page()
    
    elif page == "📈 Analytics Dashboard":
        create_advanced_analytics_dashboard()
    
    elif page == "📋 Prediction History":
        display_prediction_history_page()
    
    elif page == "⚙️ Settings":
        display_settings_page()
    
    elif page == "ℹ️ About":
        display_about_page()

def display_prediction_history_page():
    """Display enhanced prediction history page"""
    st.markdown("### 📈 Prediction History & Analytics")
    
    if not st.session_state.prediction_history:
        st.info("📝 No predictions made yet. Go to the Prediction page to start!")
        return
    
    # Convert history to DataFrame
    history_data = []
    for entry in st.session_state.prediction_history:
        row = {
            'Timestamp': entry['timestamp'],
            'Risk Level': entry.get('risk_level', 'High Risk' if entry['prediction'] == 1 else 'Low Risk'),
            'Probability': f"{entry['probability']:.1%}",
            'Confidence': f"{entry.get('confidence_score', 0):.1f}%",
            'Age': entry['inputs']['age'],
            'BMI': entry['inputs']['bmi'],
            'Glucose': entry['inputs']['glucose'],
            'BP': entry['inputs']['bp'],
            'Session': entry.get('session_id', 'Unknown')
        }
        history_data.append(row)
    
    history_df = pd.DataFrame(history_data)
    
    # Summary statistics
    st.markdown("#### 📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(history_df))
    with col2:
        high_risk_count = len([h for h in st.session_state.prediction_history if h['prediction'] == 1])
        st.metric("High Risk Cases", high_risk_count)
    with col3:
        avg_prob = np.mean([h['probability'] for h in st.session_state.prediction_history])
        st.metric("Average Risk", f"{avg_prob:.1%}")
    with col4:
        latest_prediction = st.session_state.prediction_history[-1] if st.session_state.prediction_history else None
        if latest_prediction:
            st.metric("Latest Risk", f"{latest_prediction['probability']:.1%}")
    
    # History table with filtering
    st.markdown("#### 📋 Detailed History")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.selectbox("Filter by Risk Level", 
                                 ["All"] + list(history_df['Risk Level'].unique()))
    with col2:
        date_range = st.date_input("Date Range", value=[], help="Select date range to filter")
    with col3:
        show_details = st.checkbox("Show All Columns", value=False)
    
    # Apply filters
    filtered_df = history_df.copy()
    if risk_filter != "All":
        filtered_df = filtered_df[filtered_df['Risk Level'] == risk_filter]
    
    # Display columns based on selection
    if show_details:
        display_columns = filtered_df.columns.tolist()
    else:
        display_columns = ['Timestamp', 'Risk Level', 'Probability', 'Age', 'BMI', 'Glucose']
    
    st.dataframe(filtered_df[display_columns], use_container_width=True)
    
    # Export options
    st.markdown("#### 📥 Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📄 Download as CSV",
            data=csv_data,
            file_name=f"diabetes_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        json_data = json.dumps(st.session_state.prediction_history, indent=2, default=str)
        st.download_button(
            label="📋 Download as JSON",
            data=json_data,
            file_name=f"diabetes_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col3:
        if st.button("🗑️ Clear History", type="secondary"):
            if st.button("⚠️ Confirm Clear", type="primary"):
                st.session_state.prediction_history = []
                st.success("History cleared!")
                st.experimental_rerun()

def display_settings_page():
    """Display application settings page"""
    st.markdown("### ⚙️ Application Settings")
    
    # User preferences
    st.markdown("#### 👤 User Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.user_preferences['auto_save_predictions'] = st.checkbox(
            "🔄 Auto-save predictions to history",
            value=st.session_state.user_preferences['auto_save_predictions'],
            help="Automatically save all predictions to history"
        )
        
        st.session_state.user_preferences['show_advanced_metrics'] = st.checkbox(
            "📊 Show advanced metrics",
            value=st.session_state.user_preferences['show_advanced_metrics'],
            help="Display additional statistical metrics and confidence scores"
        )
        
        st.session_state.user_preferences['show_explanations'] = st.checkbox(
            "🔍 Show model explanations",
            value=st.session_state.user_preferences['show_explanations'],
            help="Display detailed model explanations for predictions"
        )
    
    with col2:
        # Theme selection (placeholder for future implementation)
        theme = st.selectbox(
            "🎨 Theme",
            ["Light", "Dark", "Auto"],
            index=0,
            help="Choose application theme (feature coming soon)"
        )
        
        # Notification settings
        notifications = st.checkbox(
            "🔔 Enable notifications",
            value=False,
            help="Enable browser notifications for important alerts"
        )
        
        # Data retention
        retention_days = st.slider(
            "📅 History retention (days)",
            min_value=1, max_value=365, value=30,
            help="Number of days to keep prediction history"
        )
    
    # Application info
    st.markdown("#### ℹ️ Application Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.info(f"""
        **Version:** 2.0.0
        **Model Status:** {'✅ Loaded' if st.session_state.model_loaded else '❌ Not Loaded'}
        **Session Duration:** {(datetime.now() - st.session_state.app_stats['session_start_time']).seconds // 60} minutes
        """)
    
    with info_col2:
        st.info(f"""
        **Total Predictions:** {st.session_state.app_stats['total_predictions']}
        **History Size:** {len(st.session_state.prediction_history)} entries
        **Memory Usage:** {sys.getsizeof(st.session_state) / 1024:.1f} KB
        """)
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings", expanded=False):
        st.markdown("#### 🔧 Model Configuration")
        
        # Model retraining option
        if st.button("🔄 Retrain Model", type="secondary"):
            with st.spinner("Retraining model..."):
                try:
                    import subprocess
                    result = subprocess.run([sys.executable, "train_model.py"], 
                                          capture_output=True, text=True, cwd=".")
                    if result.returncode == 0:
                        st.success("✅ Model retrained successfully!")
                        st.experimental_rerun()
                    else:
                        st.error(f"❌ Retraining failed: {result.stderr}")
                except Exception as e:
                    st.error(f"❌ Error retraining model: {str(e)}")
        
        # Cache management
        if st.button("🗑️ Clear Cache", type="secondary"):
            st.cache_resource.clear()
            st.success("✅ Cache cleared!")
        
        # Reset application state
        if st.button("🔄 Reset Application", type="secondary"):
            if st.button("⚠️ Confirm Reset", type="primary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("✅ Application reset!")
                st.experimental_rerun()

def display_about_page():
    """Display enhanced about page"""
    st.markdown("### ℹ️ About Advanced Diabetes Risk Predictor")
    
    # Application overview
    st.markdown("""
    <div class="info-box">
    <h4>🎯 Mission</h4>
    <p>This advanced diabetes prediction application leverages machine learning to provide comprehensive diabetes risk assessment based on key health indicators, empowering users with actionable insights for better health management.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features overview
    st.markdown("#### 🚀 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔍 Core Functionality:**
        - Real-time diabetes risk assessment
        - Comprehensive input validation
        - Medical reasonableness checks
        - Personalized recommendations
        - Risk factor analysis
        
        **📊 Analytics & Insights:**
        - Interactive visualizations
        - Prediction history tracking
        - Advanced analytics dashboard
        - Model performance metrics
        - Feature importance analysis
        """)
    
    with col2:
        st.markdown("""
        **🎨 User Experience:**
        - Intuitive tabbed interface
        - Responsive design
        - Real-time feedback
        - Export capabilities
        - Session management
        
        **🔬 Technical Features:**
        - Model explainability
        - Performance monitoring
        - Data validation
        - Error handling
        - Caching optimization
        """)
    
    # Technical details
    st.markdown("#### 🔬 Technical Specifications")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        **🤖 Machine Learning:**
        - Algorithm: Logistic Regression
        - Preprocessing: StandardScaler
        - Features: 8 health indicators
        - Validation: Cross-validation
        - Metrics: Accuracy, AUC, Precision, Recall
        """)
    
    with tech_col2:
        st.markdown("""
        **💻 Technology Stack:**
        - Frontend: Streamlit
        - ML Library: scikit-learn
        - Visualization: Plotly, Matplotlib
        - Data Processing: Pandas, NumPy
        - Styling: Custom CSS
        """)
    
    # Dataset information
    st.markdown("#### 📊 Dataset Information")
    st.markdown("""
    **Source:** Pima Indians Diabetes Database (UCI Machine Learning Repository)
    
    **Features:**
    - **Pregnancies:** Number of times pregnant
    - **Glucose:** Plasma glucose concentration (mg/dL)
    - **Blood Pressure:** Diastolic blood pressure (mm Hg)
    - **Skin Thickness:** Triceps skin fold thickness (mm)
    - **Insulin:** 2-Hour serum insulin (μU/mL)
    - **BMI:** Body mass index (kg/m²)
    - **Diabetes Pedigree Function:** Genetic predisposition score
    - **Age:** Age in years
    """)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Important Medical Disclaimer</h4>
    <p>This application is designed for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers regarding any medical condition or treatment decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Version and credits
    st.markdown("#### 👨‍💻 Development Information")
    
    st.markdown("""
    **Version:** 2.0.0 (Enhanced Edition)
    **Last Updated:** December 2024
    **License:** MIT License
    
    **Enhanced Features by:** AI Assistant
    **Original Concept:** Diabetes Risk Assessment
    
    **Acknowledgments:**
    - Pima Indians Diabetes Database contributors
    - Streamlit development team
    - scikit-learn community
    - Open source community
    """)
    
    # Contact and support
    st.markdown("#### 📞 Support & Resources")
    
    support_col1, support_col2 = st.columns(2)
    
    with support_col1:
        st.markdown("""
        **📚 Resources:**
        - [User Guide](https://example.com/guide)
        - [API Documentation](https://example.com/docs)
        - [FAQ](https://example.com/faq)
        - [Tutorials](https://example.com/tutorials)
        """)
    
    with support_col2:
        st.markdown("""
        **🤝 Community:**
        - [GitHub Repository](https://github.com/example/diabetes-predictor)
        - [Issue Tracker](https://github.com/example/diabetes-predictor/issues)
        - [Discussions](https://github.com/example/diabetes-predictor/discussions)
        - [Contributing Guide](https://github.com/example/diabetes-predictor/blob/main/CONTRIBUTING.md)
        """)

def display_model_performance():
    """Display enhanced model performance metrics"""
    st.markdown("### 📊 Model Performance Analysis")
    
    try:
        # Load test data for performance metrics
        df = pd.read_csv("diabetes.csv")
        
        # Apply same preprocessing as in training
        cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in cols:
            df[col] = df[col].replace(0, df[col].median())
        
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        
        model, scaler = load_model_and_scaler()
        if model and scaler:
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
            
            # Calculate comprehensive metrics
            accuracy = accuracy_score(y, y_pred)
            auc_score = roc_auc_score(y, y_pred_proba)
            
            # Precision and recall
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            
            st.markdown("#### 📈 Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Accuracy</h3>
                    <h2>{accuracy:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>AUC Score</h3>
                    <h2>{auc_score:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Precision</h3>
                    <h2>{precision:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Recall</h3>
                    <h2>{recall:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Confusion Matrix
            st.markdown("#### 📊 Confusion Matrix")
            cm = confusion_matrix(y, y_pred)
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted No Diabetes', 'Predicted Diabetes'],
                y=['Actual No Diabetes', 'Actual Diabetes'],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='Actual',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Classification report
            st.markdown("#### 📋 Detailed Classification Report")
            report = classification_report(y, y_pred, output_dict=True)
            
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)
            
    except Exception as e:
        st.error(f"Error displaying model performance: {str(e)}")
        logger.error(f"Error in model performance display: {str(e)}")

if __name__ == "__main__":
    main()
