# Configuration file for the Diabetes Prediction App

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class ModelConfig:
    """Configuration for model parameters"""
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    max_iter: int = 1000

@dataclass
class UIConfig:
    """Configuration for UI elements"""
    page_title: str = "Diabetes Risk Predictor"
    page_icon: str = "🩺"
    layout: str = "wide"
    
    # Color schemes
    colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'primary': '#1f77b4',
                'success': '#51cf66',
                'warning': '#ffd43b',
                'danger': '#ff6b6b',
                'info': '#339af0'
            }

@dataclass
class FeatureConfig:
    """Configuration for feature parameters"""
    feature_ranges: Dict[str, Tuple[float, float]] = None
    feature_defaults: Dict[str, float] = None
    feature_descriptions: Dict[str, str] = None
    
    def __post_init__(self):
        if self.feature_ranges is None:
            self.feature_ranges = {
                'pregnancies': (0, 17),
                'glucose': (0, 200),
                'blood_pressure': (0, 122),
                'skin_thickness': (0, 99),
                'insulin': (0, 846),
                'bmi': (0.0, 67.1),
                'pedigree': (0.0, 2.5),
                'age': (21, 90)
            }
        
        if self.feature_defaults is None:
            self.feature_defaults = {
                'pregnancies': 3,
                'glucose': 120,
                'blood_pressure': 70,
                'skin_thickness': 20,
                'insulin': 79,
                'bmi': 32.0,
                'pedigree': 0.47,
                'age': 33
            }
        
        if self.feature_descriptions is None:
            self.feature_descriptions = {
                'pregnancies': 'Number of times pregnant',
                'glucose': 'Plasma glucose concentration after 2 hours in oral glucose tolerance test',
                'blood_pressure': 'Diastolic blood pressure (mm Hg)',
                'skin_thickness': 'Triceps skin fold thickness (mm)',
                'insulin': '2-Hour serum insulin (mu U/ml)',
                'bmi': 'Body mass index (weight in kg/(height in m)²)',
                'pedigree': 'Diabetes pedigree function (genetic predisposition)',
                'age': 'Age in years'
            }

@dataclass
class ValidationConfig:
    """Configuration for input validation"""
    glucose_normal_range: Tuple[float, float] = (70, 100)
    bp_normal_range: Tuple[float, float] = (60, 80)
    bmi_categories: Dict[str, Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.bmi_categories is None:
            self.bmi_categories = {
                'underweight': (0, 18.5),
                'normal': (18.5, 25),
                'overweight': (25, 30),
                'obese': (30, float('inf'))
            }

# Global configuration instances
MODEL_CONFIG = ModelConfig()
UI_CONFIG = UIConfig()
FEATURE_CONFIG = FeatureConfig()
VALIDATION_CONFIG = ValidationConfig()

# File paths
DATA_PATH = "diabetes.csv"
MODEL_PATH = "logistic_model.pkl"
SCALER_PATH = "scaler.pkl"
METADATA_PATH = "model_metadata.json"
REPORT_PATH = "model_evaluation_report.png"

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'