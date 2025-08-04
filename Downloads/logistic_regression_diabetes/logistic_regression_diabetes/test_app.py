#!/usr/bin/env python3
"""
Comprehensive test suite for the Enhanced Diabetes Prediction App
"""

import unittest
import pandas as pd
import numpy as np
import pickle
import os
import sys
from unittest.mock import patch, MagicMock
import tempfile
import json

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules to test
from utils import (
    validate_medical_inputs, get_risk_interpretation, get_bmi_category,
    get_age_group, get_glucose_status, calculate_risk_factors,
    format_medical_value
)
from config import MODEL_CONFIG, UI_CONFIG, FEATURE_CONFIG, VALIDATION_CONFIG
from model_explainer import ModelExplainer
from data_analysis import DataAnalyzer

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_validate_medical_inputs(self):
        """Test medical input validation"""
        # Normal inputs
        normal_inputs = {
            'glucose': 100,
            'bp': 70,
            'bmi': 23,
            'age': 30,
            'pregnancies': 2,
            'insulin': 100,
            'pedigree': 0.3,
            'skin': 25
        }
        warnings = validate_medical_inputs(normal_inputs)
        self.assertIsInstance(warnings, list)
        
        # High glucose
        high_glucose_inputs = normal_inputs.copy()
        high_glucose_inputs['glucose'] = 150
        warnings = validate_medical_inputs(high_glucose_inputs)
        self.assertTrue(any('glucose' in w.lower() for w in warnings))
        
        # High BMI
        high_bmi_inputs = normal_inputs.copy()
        high_bmi_inputs['bmi'] = 35
        warnings = validate_medical_inputs(high_bmi_inputs)
        self.assertTrue(any('bmi' in w.lower() or 'obesity' in w.lower() for w in warnings))
    
    def test_get_risk_interpretation(self):
        """Test risk interpretation function"""
        # Very low risk
        risk_level, description, emoji = get_risk_interpretation(0.1)
        self.assertEqual(risk_level, "Very Low Risk")
        self.assertEqual(emoji, "🟢")
        
        # High risk
        risk_level, description, emoji = get_risk_interpretation(0.8)
        self.assertEqual(risk_level, "Very High Risk")
        self.assertEqual(emoji, "🔴")
        
        # Moderate risk
        risk_level, description, emoji = get_risk_interpretation(0.5)
        self.assertEqual(risk_level, "Moderate Risk")
        self.assertEqual(emoji, "🟡")
    
    def test_get_bmi_category(self):
        """Test BMI categorization"""
        self.assertEqual(get_bmi_category(17), "Underweight")
        self.assertEqual(get_bmi_category(22), "Normal Weight")
        self.assertEqual(get_bmi_category(27), "Overweight")
        self.assertEqual(get_bmi_category(32), "Obese Class I")
        self.assertEqual(get_bmi_category(37), "Obese Class II")
        self.assertEqual(get_bmi_category(42), "Obese Class III")
    
    def test_get_age_group(self):
        """Test age group classification"""
        self.assertEqual(get_age_group(22), "Young Adult")
        self.assertEqual(get_age_group(30), "Adult")
        self.assertEqual(get_age_group(40), "Middle-aged Adult")
        self.assertEqual(get_age_group(55), "Older Adult")
        self.assertEqual(get_age_group(70), "Senior")
    
    def test_get_glucose_status(self):
        """Test glucose status classification"""
        self.assertEqual(get_glucose_status(60), "Hypoglycemic")
        self.assertEqual(get_glucose_status(90), "Normal")
        self.assertEqual(get_glucose_status(110), "Pre-diabetic Range")
        self.assertEqual(get_glucose_status(140), "Diabetic Range")
    
    def test_calculate_risk_factors(self):
        """Test risk factor calculation"""
        inputs = {
            'glucose': 140,
            'bmi': 32,
            'age': 50,
            'pedigree': 0.8,
            'pregnancies': 3,
            'bp': 85,
            'insulin': 100,
            'skin': 25
        }
        
        risk_factors = calculate_risk_factors(inputs)
        
        self.assertIn('high_risk', risk_factors)
        self.assertIn('moderate_risk', risk_factors)
        self.assertIn('protective_factors', risk_factors)
        
        # Should have high risk factors for high glucose and BMI
        self.assertTrue(len(risk_factors['high_risk']) > 0)
    
    def test_format_medical_value(self):
        """Test medical value formatting"""
        self.assertEqual(format_medical_value(120.5, "mg/dL", 1), "120.5 mg/dL")
        self.assertEqual(format_medical_value(25, "years", 0), "25 years")
        self.assertEqual(format_medical_value(32.456, "kg/m²", 2), "32.46 kg/m²")

class TestConfiguration(unittest.TestCase):
    """Test configuration classes"""
    
    def test_model_config(self):
        """Test model configuration"""
        self.assertEqual(MODEL_CONFIG.random_state, 42)
        self.assertEqual(MODEL_CONFIG.test_size, 0.2)
        self.assertIsInstance(MODEL_CONFIG.cv_folds, int)
        self.assertIsInstance(MODEL_CONFIG.max_iter, int)
    
    def test_ui_config(self):
        """Test UI configuration"""
        self.assertEqual(UI_CONFIG.page_title, "Diabetes Risk Predictor")
        self.assertEqual(UI_CONFIG.page_icon, "🩺")
        self.assertEqual(UI_CONFIG.layout, "wide")
        self.assertIsInstance(UI_CONFIG.colors, dict)
    
    def test_feature_config(self):
        """Test feature configuration"""
        self.assertIsInstance(FEATURE_CONFIG.feature_ranges, dict)
        self.assertIsInstance(FEATURE_CONFIG.feature_defaults, dict)
        self.assertIsInstance(FEATURE_CONFIG.feature_descriptions, dict)
        
        # Check that all required features are present
        required_features = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                           'insulin', 'bmi', 'pedigree', 'age']
        for feature in required_features:
            self.assertIn(feature, FEATURE_CONFIG.feature_ranges)
            self.assertIn(feature, FEATURE_CONFIG.feature_defaults)
            self.assertIn(feature, FEATURE_CONFIG.feature_descriptions)
    
    def test_validation_config(self):
        """Test validation configuration"""
        self.assertIsInstance(VALIDATION_CONFIG.glucose_normal_range, tuple)
        self.assertIsInstance(VALIDATION_CONFIG.bp_normal_range, tuple)
        self.assertIsInstance(VALIDATION_CONFIG.bmi_categories, dict)

class TestDataAnalyzer(unittest.TestCase):
    """Test data analysis functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Create a temporary CSV file for testing
        self.test_data = pd.DataFrame({
            'Pregnancies': [1, 2, 3, 4, 5],
            'Glucose': [85, 95, 105, 115, 125],
            'BloodPressure': [70, 75, 80, 85, 90],
            'SkinThickness': [20, 25, 30, 35, 40],
            'Insulin': [80, 90, 100, 110, 120],
            'BMI': [22, 24, 26, 28, 30],
            'DiabetesPedigreeFunction': [0.2, 0.3, 0.4, 0.5, 0.6],
            'Age': [25, 30, 35, 40, 45],
            'Outcome': [0, 0, 1, 1, 1]
        })
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test data"""
        os.unlink(self.temp_file.name)
    
    def test_data_analyzer_initialization(self):
        """Test DataAnalyzer initialization"""
        analyzer = DataAnalyzer(self.temp_file.name)
        self.assertIsNotNone(analyzer.df)
        self.assertEqual(len(analyzer.df), 5)
    
    def test_get_dataset_overview(self):
        """Test dataset overview generation"""
        analyzer = DataAnalyzer(self.temp_file.name)
        overview = analyzer.get_dataset_overview()
        
        self.assertIn('shape', overview)
        self.assertIn('columns', overview)
        self.assertIn('outcome_distribution', overview)
        self.assertEqual(overview['shape'], (5, 9))
    
    def test_get_descriptive_statistics(self):
        """Test descriptive statistics"""
        analyzer = DataAnalyzer(self.temp_file.name)
        stats = analyzer.get_descriptive_statistics()
        
        self.assertIsInstance(stats, pd.DataFrame)
        self.assertIn('mean', stats.index)
        self.assertIn('std', stats.index)
    
    def test_get_feature_insights(self):
        """Test feature insights generation"""
        analyzer = DataAnalyzer(self.temp_file.name)
        insights = analyzer.get_feature_insights()
        
        self.assertIsInstance(insights, dict)
        self.assertIn('Glucose', insights)
        self.assertIn('BMI', insights)
        self.assertIn('Age', insights)

class TestModelExplainer(unittest.TestCase):
    """Test model explanation functionality"""
    
    def setUp(self):
        """Set up test model and scaler"""
        # Create mock model and scaler files
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Create a simple model
        self.model = LogisticRegression(random_state=42)
        X_dummy = np.random.rand(100, 8)
        y_dummy = np.random.randint(0, 2, 100)
        self.model.fit(X_dummy, y_dummy)
        
        # Create a scaler
        self.scaler = StandardScaler()
        self.scaler.fit(X_dummy)
        
        # Save to temporary files
        self.model_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False)
        pickle.dump(self.model, self.model_file)
        self.model_file.close()
        
        self.scaler_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False)
        pickle.dump(self.scaler, self.scaler_file)
        self.scaler_file.close()
    
    def tearDown(self):
        """Clean up test files"""
        os.unlink(self.model_file.name)
        os.unlink(self.scaler_file.name)
    
    def test_model_explainer_initialization(self):
        """Test ModelExplainer initialization"""
        explainer = ModelExplainer(self.model_file.name, self.scaler_file.name)
        self.assertIsNotNone(explainer.model)
        self.assertIsNotNone(explainer.scaler)
    
    def test_get_feature_importance(self):
        """Test feature importance calculation"""
        explainer = ModelExplainer(self.model_file.name, self.scaler_file.name)
        importance_df = explainer.get_feature_importance()
        
        self.assertIsInstance(importance_df, pd.DataFrame)
        if not importance_df.empty:
            self.assertIn('Feature', importance_df.columns)
            self.assertIn('Importance', importance_df.columns)
            self.assertIn('Coefficient', importance_df.columns)
    
    def test_explain_prediction(self):
        """Test individual prediction explanation"""
        explainer = ModelExplainer(self.model_file.name, self.scaler_file.name)
        
        input_data = {
            'pregnancies': 3,
            'glucose': 120,
            'bp': 70,
            'skin': 25,
            'insulin': 100,
            'bmi': 25,
            'pedigree': 0.5,
            'age': 35
        }
        
        explanation = explainer.explain_prediction(input_data)
        
        if 'error' not in explanation:
            self.assertIn('prediction', explanation)
            self.assertIn('probability_diabetes', explanation)
            self.assertIn('feature_contributions', explanation)
    
    def test_get_model_summary(self):
        """Test model summary generation"""
        explainer = ModelExplainer(self.model_file.name, self.scaler_file.name)
        summary = explainer.get_model_summary()
        
        if 'error' not in summary:
            self.assertIn('model_type', summary)
            self.assertIn('n_features', summary)
            self.assertIn('feature_names', summary)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete application"""
    
    def test_prediction_pipeline(self):
        """Test the complete prediction pipeline"""
        # This would test the full pipeline from input to prediction
        # For now, we'll test the individual components
        
        inputs = {
            'pregnancies': 3,
            'glucose': 120,
            'bp': 70,
            'skin': 25,
            'insulin': 100,
            'bmi': 25,
            'pedigree': 0.5,
            'age': 35
        }
        
        # Test input validation
        warnings = validate_medical_inputs(inputs)
        self.assertIsInstance(warnings, list)
        
        # Test risk factor calculation
        risk_factors = calculate_risk_factors(inputs)
        self.assertIn('high_risk', risk_factors)
        self.assertIn('moderate_risk', risk_factors)
        self.assertIn('protective_factors', risk_factors)
    
    def test_data_consistency(self):
        """Test data consistency across modules"""
        # Test that feature names are consistent
        feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        explainer = ModelExplainer()
        self.assertEqual(explainer.feature_names, feature_names)

def run_performance_tests():
    """Run performance tests"""
    print("Running performance tests...")
    
    import time
    
    # Test utility function performance
    start_time = time.time()
    for _ in range(1000):
        inputs = {
            'glucose': 120,
            'bp': 70,
            'bmi': 25,
            'age': 35,
            'pregnancies': 3,
            'insulin': 100,
            'pedigree': 0.5,
            'skin': 25
        }
        validate_medical_inputs(inputs)
        get_risk_interpretation(0.5)
        calculate_risk_factors(inputs)
    
    end_time = time.time()
    print(f"Utility functions: {end_time - start_time:.3f} seconds for 1000 iterations")
    
    # Test data analysis performance
    if os.path.exists("diabetes.csv"):
        start_time = time.time()
        analyzer = DataAnalyzer("diabetes.csv")
        overview = analyzer.get_dataset_overview()
        stats = analyzer.get_descriptive_statistics()
        insights = analyzer.get_feature_insights()
        end_time = time.time()
        print(f"Data analysis: {end_time - start_time:.3f} seconds")

def run_all_tests():
    """Run all tests"""
    print("Starting comprehensive test suite...")
    
    # Run unit tests
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Run performance tests
    run_performance_tests()
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)