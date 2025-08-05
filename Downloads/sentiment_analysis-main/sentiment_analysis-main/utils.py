# Utility functions for IMDb Sentiment Analyzer

import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import streamlit as st
from config import COLORS, FEATURE_NORMALIZATION

def preprocess_text(text: str) -> str:
    """
    Advanced text preprocessing with multiple cleaning steps
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Cleaned and preprocessed text
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove very short words (likely typos)
    words = text.split()
    words = [word for word in words if len(word) > 1 or word.lower() in ['i', 'a']]
    text = ' '.join(words)
    
    return text.strip()

def analyze_text_features(text: str) -> Dict:
    """
    Comprehensive text feature analysis
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Dict: Dictionary containing various text features
    """
    if not text:
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'uppercase_ratio': 0,
            'punctuation_density': 0,
            'unique_words': 0,
            'readability_score': 0
        }
    
    words = text.split()
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Basic features
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    
    # Punctuation analysis
    exclamation_count = text.count('!')
    question_count = text.count('?')
    punctuation_chars = '!@#$%^&*()_+-=[]{}|;:,.<>?'
    punctuation_count = sum(1 for char in text if char in punctuation_chars)
    punctuation_density = punctuation_count / len(text) if text else 0
    
    # Case analysis
    uppercase_count = sum(1 for c in text if c.isupper())
    uppercase_ratio = uppercase_count / len(text) if text else 0
    
    # Vocabulary richness
    unique_words = len(set(word.lower() for word in words))
    
    # Simple readability score (based on avg sentence length and word length)
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    readability_score = max(0, 10 - (avg_sentence_length / 10 + avg_word_length))
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': round(avg_word_length, 2),
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'uppercase_ratio': round(uppercase_ratio, 3),
        'punctuation_density': round(punctuation_density, 3),
        'unique_words': unique_words,
        'readability_score': round(readability_score, 2)
    }

def create_confidence_chart(probabilities: np.ndarray) -> go.Figure:
    """
    Create an enhanced confidence visualization
    
    Args:
        probabilities (np.ndarray): Prediction probabilities [negative, positive]
        
    Returns:
        go.Figure: Plotly figure object
    """
    labels = ['Negative', 'Positive']
    colors = [COLORS['negative'], COLORS['positive']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=probabilities,
            marker_color=colors,
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Prediction Confidence",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        yaxis_title="Probability",
        yaxis=dict(tickformat='.0%'),
        showlegend=False,
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_feature_radar(features: Dict) -> go.Figure:
    """
    Create an enhanced radar chart for text features
    
    Args:
        features (Dict): Text features dictionary
        
    Returns:
        go.Figure: Plotly figure object
    """
    categories = [
        'Word Count', 'Sentences', 'Avg Word Len', 
        'Exclamations', 'Questions', 'Uppercase %',
        'Punctuation', 'Unique Words', 'Readability'
    ]
    
    # Normalize values for radar chart (0-10 scale)
    values = [
        min(features['word_count'] / FEATURE_NORMALIZATION['word_count_scale'], 10),
        min(features['sentence_count'], FEATURE_NORMALIZATION['max_sentences']),
        min(features['avg_word_length'], FEATURE_NORMALIZATION['max_word_length']),
        min(features['exclamation_count'], FEATURE_NORMALIZATION['max_punctuation']),
        min(features['question_count'], FEATURE_NORMALIZATION['max_punctuation']),
        features['uppercase_ratio'] * 100,
        features['punctuation_density'] * 100,
        min(features['unique_words'] / 10, 10),
        features['readability_score']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Text Features',
        line_color=COLORS['primary'],
        fillcolor=f"rgba(102, 126, 234, 0.3)",
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickmode='linear',
                tick0=0,
                dtick=2
            )),
        showlegend=False,
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        title={
            'text': "Text Feature Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        }
    )
    
    return fig

def create_history_chart(history_df: pd.DataFrame) -> go.Figure:
    """
    Create a comprehensive history visualization
    
    Args:
        history_df (pd.DataFrame): Analysis history dataframe
        
    Returns:
        go.Figure: Plotly figure object
    """
    if len(history_df) < 2:
        return None
    
    fig = px.scatter(
        history_df, 
        x='timestamp', 
        y='confidence',
        color='sentiment',
        size='word_count',
        title="Analysis History Over Time",
        color_discrete_map={'Positive': COLORS['positive'], 'Negative': COLORS['negative']},
        hover_data=['word_count', 'confidence']
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def get_confidence_level(confidence: float) -> Tuple[str, str]:
    """
    Determine confidence level and appropriate message
    
    Args:
        confidence (float): Confidence score (0-1)
        
    Returns:
        Tuple[str, str]: (level, message)
    """
    if confidence >= 0.8:
        return "High", "Very reliable prediction!"
    elif confidence >= 0.6:
        return "Moderate", "Good prediction with reasonable confidence."
    else:
        return "Low", "Less reliable prediction. Consider reviewing the text."

def format_analysis_summary(prediction: int, confidence: float, features: Dict) -> str:
    """
    Create a formatted summary of the analysis
    
    Args:
        prediction (int): Sentiment prediction (0 or 1)
        confidence (float): Confidence score
        features (Dict): Text features
        
    Returns:
        str: Formatted summary string
    """
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence_level, _ = get_confidence_level(confidence)
    
    summary = f"""
    **Analysis Summary:**
    - **Sentiment:** {sentiment} ({confidence:.1%} confidence)
    - **Confidence Level:** {confidence_level}
    - **Text Length:** {features['word_count']} words, {features['sentence_count']} sentences
    - **Writing Style:** {features['exclamation_count']} exclamations, {features['question_count']} questions
    - **Vocabulary:** {features['unique_words']} unique words
    - **Readability Score:** {features['readability_score']}/10
    """
    
    return summary

def validate_input(text: str, min_length: int = 10, max_length: int = 10000) -> Tuple[bool, str]:
    """
    Validate user input text
    
    Args:
        text (str): Input text to validate
        min_length (int): Minimum required length
        max_length (int): Maximum allowed length
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Please enter some text to analyze."
    
    if len(text.strip()) < min_length:
        return False, f"Text is too short. Please enter at least {min_length} characters."
    
    if len(text) > max_length:
        return False, f"Text is too long. Please limit to {max_length} characters."
    
    # Check for meaningful content (not just punctuation/numbers)
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    if len(words) < 3:
        return False, "Please enter more meaningful text with at least 3 words."
    
    return True, ""

def export_history_to_csv(history_data: List[Dict]) -> str:
    """
    Export analysis history to CSV format
    
    Args:
        history_data (List[Dict]): List of analysis records
        
    Returns:
        str: CSV formatted string
    """
    if not history_data:
        return ""
    
    df = pd.DataFrame(history_data)
    return df.to_csv(index=False)

def get_text_statistics(text: str) -> Dict:
    """
    Get comprehensive text statistics
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Comprehensive statistics
    """
    features = analyze_text_features(text)
    
    # Additional statistics
    words = text.split()
    word_lengths = [len(word) for word in words]
    
    stats = {
        **features,
        'character_count': len(text),
        'character_count_no_spaces': len(text.replace(' ', '')),
        'longest_word': max(word_lengths) if word_lengths else 0,
        'shortest_word': min(word_lengths) if word_lengths else 0,
        'avg_sentence_length': features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0
    }
    
    return stats