import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os
from typing import Dict, List, Tuple

# Import custom modules
from config import *
from utils import (
    preprocess_text, analyze_text_features, create_confidence_chart,
    create_feature_radar, create_history_chart, get_confidence_level,
    format_analysis_summary, validate_input, get_text_statistics
)

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .positive-sentiment {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .negative-sentiment {
        background: linear-gradient(90deg, #cb2d3e 0%, #ef473a 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .neutral-sentiment {
        background: linear-gradient(90deg, #f7971e 0%, #ffd200 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    
    .stButton button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model with error handling"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at {MODEL_PATH}. Please train the model first.")
            st.stop()
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def get_prediction_confidence(model, text: str) -> Tuple[int, float, np.ndarray]:
    """Get prediction with confidence scores"""
    try:
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        confidence = max(probabilities)
        return prediction, confidence, probabilities
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Load model
model = load_model()

# Header
st.markdown("""
<div class="main-header">
    <h1>🎬 Advanced IMDb Sentiment Analyzer</h1>
    <p>Powered by Machine Learning • Real-time Analysis • Detailed Insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Analysis Settings")
    
    show_confidence = st.checkbox("Show Confidence Scores", value=True)
    show_features = st.checkbox("Show Text Features", value=True)
    show_history = st.checkbox("Show Analysis History", value=False)
    
    st.header("📝 Sample Reviews")
    selected_sample = st.selectbox("Choose a sample:", [""] + list(SAMPLE_REVIEWS.keys()))
    
    if st.button("Use Sample") and selected_sample:
        st.session_state.sample_text = SAMPLE_REVIEWS[selected_sample]

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎭 Enter Your Movie Review")
    
    # Text input with sample handling
    default_text = st.session_state.get('sample_text', '')
    user_input = st.text_area(
        "Write your movie review here:",
        value=default_text,
        height=150,
        placeholder="Enter a detailed movie review to analyze its sentiment..."
    )
    
    # Clear sample text after use
    if 'sample_text' in st.session_state:
        del st.session_state.sample_text
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        analyze_btn = st.button("🔍 Analyze Sentiment", type="primary")
    
    with col_btn2:
        clear_btn = st.button("🗑️ Clear")
    
    if clear_btn:
        st.rerun()

with col2:
    st.header("📈 Quick Stats")
    if user_input.strip():
        features = analyze_text_features(user_input)
        
        st.metric("Word Count", features['word_count'])
        st.metric("Sentences", features['sentence_count'])
        st.metric("Avg Word Length", f"{features['avg_word_length']:.1f}")

# Analysis section
if analyze_btn:
    # Validate input
    is_valid, error_message = validate_input(user_input)
    
    if not is_valid:
        st.warning(f"⚠️ {error_message}")
    else:
        with st.spinner("🤖 Analyzing sentiment..."):
            # Add a small delay for better UX
            time.sleep(0.5)
            
            # Preprocess text
            processed_text = preprocess_text(user_input)
            
            # Get prediction
            prediction, confidence, probabilities = get_prediction_confidence(model, processed_text)
            
            if prediction is not None:
                # Get comprehensive features
                features = analyze_text_features(processed_text)
                
                # Store in history with more details
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now(),
                    'text': user_input[:100] + "..." if len(user_input) > 100 else user_input,
                    'prediction': prediction,
                    'confidence': confidence,
                    'sentiment': 'Positive' if prediction == 1 else 'Negative',
                    'word_count': features['word_count']
                })
            
                # Results section
                st.header("🎯 Analysis Results")
                
                # Main result with enhanced styling
                sentiment_label = "Positive" if prediction == 1 else "Negative"
                confidence_level, confidence_message = get_confidence_level(confidence)
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="positive-sentiment">
                        <h2>✅ Positive Sentiment</h2>
                        <p>This review expresses a positive opinion about the movie!</p>
                        <p><strong>Confidence:</strong> {confidence:.1%} ({confidence_level})</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="negative-sentiment">
                        <h2>❌ Negative Sentiment</h2>
                        <p>This review expresses a negative opinion about the movie!</p>
                        <p><strong>Confidence:</strong> {confidence:.1%} ({confidence_level})</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Analysis summary
                st.markdown(format_analysis_summary(prediction, confidence, features))
                
                # Detailed analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    if show_confidence:
                        st.subheader("📊 Confidence Analysis")
                        fig_conf = create_confidence_chart(probabilities)
                        st.plotly_chart(fig_conf, use_container_width=True)
                        
                        # Confidence interpretation
                        if confidence_level == "High":
                            st.success(f"🎯 {confidence_message}")
                        elif confidence_level == "Moderate":
                            st.info(f"⚖️ {confidence_message}")
                        else:
                            st.warning(f"⚠️ {confidence_message}")
                
                with col2:
                    if show_features:
                        st.subheader("📝 Text Analysis")
                        
                        # Enhanced feature metrics
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Words", features['word_count'])
                            st.metric("Sentences", features['sentence_count'])
                            st.metric("Exclamations", features['exclamation_count'])
                            st.metric("Unique Words", features['unique_words'])
                        
                        with col_b:
                            st.metric("Questions", features['question_count'])
                            st.metric("Avg Word Length", f"{features['avg_word_length']:.1f}")
                            st.metric("Uppercase %", f"{features['uppercase_ratio']:.1%}")
                            st.metric("Readability", f"{features['readability_score']:.1f}/10")
                
                # Enhanced feature visualization
                if show_features:
                    st.subheader("🎯 Comprehensive Feature Analysis")
                    fig_radar = create_feature_radar(features)
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # Additional insights
                    with st.expander("📋 Detailed Text Statistics"):
                        stats = get_text_statistics(processed_text)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write("**Length Metrics:**")
                            st.write(f"- Characters: {stats['character_count']}")
                            st.write(f"- Characters (no spaces): {stats['character_count_no_spaces']}")
                            st.write(f"- Longest word: {stats['longest_word']} chars")
                            st.write(f"- Shortest word: {stats['shortest_word']} chars")
                        
                        with col2:
                            st.write("**Structure Metrics:**")
                            st.write(f"- Avg sentence length: {stats['avg_sentence_length']:.1f} words")
                            st.write(f"- Punctuation density: {stats['punctuation_density']:.1%}")
                            st.write(f"- Vocabulary richness: {stats['unique_words']}/{stats['word_count']} unique")
                        
                        with col3:
                            st.write("**Style Indicators:**")
                            st.write(f"- Readability score: {stats['readability_score']:.1f}/10")
                            st.write(f"- Emotional indicators: {stats['exclamation_count']} !, {stats['question_count']} ?")
                            st.write(f"- Emphasis level: {stats['uppercase_ratio']:.1%} uppercase")

# History section
if show_history and st.session_state.analysis_history:
    st.header("📚 Analysis History")
    
    history_df = pd.DataFrame(st.session_state.analysis_history)
    history_df['sentiment'] = history_df['prediction'].map({1: 'Positive', 0: 'Negative'})
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Analyses", len(history_df))
    
    with col2:
        positive_count = (history_df['prediction'] == 1).sum()
        st.metric("Positive Reviews", positive_count)
    
    with col3:
        avg_confidence = history_df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Enhanced history chart
    if len(history_df) > 1:
        fig_history = create_history_chart(history_df)
        if fig_history:
            st.plotly_chart(fig_history, use_container_width=True)
    
    # Recent analyses table
    st.subheader("Recent Analyses")
    display_df = history_df[['timestamp', 'text', 'sentiment', 'confidence']].tail(5)
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
    st.dataframe(display_df, use_container_width=True)
    
    if st.button("🗑️ Clear History"):
        st.session_state.analysis_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🤖 Powered by Scikit-learn & TF-IDF | Built with ❤️ using Streamlit</p>
    <p>📊 Advanced sentiment analysis with confidence scoring and feature extraction</p>
</div>
""", unsafe_allow_html=True)