#!/usr/bin/env python3
"""
Streamlit Cloud optimized entry point for the Advanced Diabetes Risk Predictor
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set page config first (must be the first Streamlit command)
st.set_page_config(
    page_title="Advanced Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Manupati-Suresh/ragas',
        'Report a bug': 'https://github.com/Manupati-Suresh/ragas/issues',
        'About': """
        # Advanced Diabetes Risk Predictor v2.0
        
        A comprehensive machine learning application for diabetes risk assessment.
        
        **Features:**
        - Real-time risk assessment
        - Model explainability
        - Advanced analytics
        - Interactive visualizations
        
        **Version:** 2.0.0
        **License:** MIT
        """
    }
)

# Import and run the main app
try:
    from app import main
    
    # Add a header for Streamlit Cloud
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">🩺 Advanced Diabetes Risk Predictor v2.0</h1>
        <p style="color: white; margin: 0; opacity: 0.9;">Powered by Machine Learning | Deployed on Streamlit Cloud</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Run the main application
    if __name__ == "__main__":
        main()
        
except Exception as e:
    st.error(f"Error loading application: {str(e)}")
    st.info("Please check the logs for more details.")
    
    # Fallback simple app
    st.title("🩺 Diabetes Risk Predictor (Fallback Mode)")
    st.warning("Running in fallback mode with basic functionality.")
    
    # Simple prediction interface
    st.header("Basic Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.slider("Pregnancies", 0, 17, 3)
        glucose = st.slider("Glucose Level", 0, 200, 120)
        bp = st.slider("Blood Pressure", 0, 122, 70)
        skin = st.slider("Skin Thickness", 0, 99, 20)
    
    with col2:
        insulin = st.slider("Insulin", 0, 846, 79)
        bmi = st.slider("BMI", 0.0, 67.1, 32.0)
        pedigree = st.slider("Pedigree Function", 0.0, 2.5, 0.47)
        age = st.slider("Age", 21, 90, 33)
    
    if st.button("Predict Diabetes Risk"):
        try:
            # Try to load model
            import pickle
            model = pickle.load(open("logistic_model.pkl", "rb"))
            scaler = pickle.load(open("scaler.pkl", "rb"))
            
            # Make prediction
            input_data = [[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]]
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            # Display result
            if prediction == 1:
                st.error(f"🔴 High Risk - Probability: {probability:.1%}")
            else:
                st.success(f"🟢 Low Risk - Probability: {probability:.1%}")
                
        except Exception as model_error:
            st.error(f"Model prediction failed: {str(model_error)}")
            st.info("Please ensure model files are available.")