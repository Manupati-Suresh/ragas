#!/usr/bin/env python3
"""
Test script for IMDb Sentiment Analyzer
Run this to verify all components work correctly
"""

import sys
import os
import traceback

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import joblib
        print("✅ Joblib imported successfully")
    except ImportError as e:
        print(f"❌ Joblib import failed: {e}")
        return False
    
    try:
        import plotly
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ Pandas and NumPy imported successfully")
    except ImportError as e:
        print(f"❌ Pandas/NumPy import failed: {e}")
        return False
    
    return True

def test_custom_modules():
    """Test if custom modules can be imported"""
    print("\nTesting custom modules...")
    
    try:
        from config import APP_TITLE, COLORS, SAMPLE_REVIEWS
        print("✅ Config module imported successfully")
        print(f"   - App title: {APP_TITLE}")
        print(f"   - Colors defined: {len(COLORS)}")
        print(f"   - Sample reviews: {len(SAMPLE_REVIEWS)}")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from utils import preprocess_text, analyze_text_features
        print("✅ Utils module imported successfully")
        
        # Test basic functionality
        test_text = "This is a great movie! I loved it."
        processed = preprocess_text(test_text)
        features = analyze_text_features(processed)
        
        print(f"   - Text processing works: {len(processed)} chars")
        print(f"   - Feature analysis works: {features['word_count']} words")
        
    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Utils functionality failed: {e}")
        return False
    
    return True

def test_model_path():
    """Test if model file exists"""
    print("\nTesting model availability...")
    
    model_path = "notebooks/model/imdb_pipeline.pkl"
    
    if os.path.exists(model_path):
        print(f"✅ Model file found at {model_path}")
        
        try:
            import joblib
            model = joblib.load(model_path)
            print("✅ Model loaded successfully")
            
            # Test prediction
            test_review = "This movie was amazing!"
            prediction = model.predict([test_review])[0]
            probabilities = model.predict_proba([test_review])[0]
            
            print(f"✅ Model prediction works: {prediction} with confidence {max(probabilities):.2f}")
            
        except Exception as e:
            print(f"❌ Model loading/prediction failed: {e}")
            return False
    else:
        print(f"⚠️ Model file not found at {model_path}")
        print("   Run the training script first: python notebooks/imdb_sentiment_analysis.py")
        return False
    
    return True

def test_app_structure():
    """Test if app.py has correct structure"""
    print("\nTesting app structure...")
    
    if not os.path.exists("app.py"):
        print("❌ app.py not found")
        return False
    
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    required_elements = [
        "import streamlit as st",
        "from config import",
        "from utils import",
        "st.set_page_config",
        "load_model",
        "st.title"
    ]
    
    for element in required_elements:
        if element in content:
            print(f"✅ Found: {element}")
        else:
            print(f"❌ Missing: {element}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 IMDb Sentiment Analyzer - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Custom Module Tests", test_custom_modules),
        ("Model Availability", test_model_path),
        ("App Structure", test_app_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your app should work correctly.")
        print("Run: streamlit run app.py")
    else:
        print(f"\n⚠️ {len(results) - passed} test(s) failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())