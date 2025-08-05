#!/usr/bin/env python3
"""
Test script for Temperature Forecast Dashboard
Validates core functionality without running the full Streamlit app
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import traceback

def test_data_loading():
    """Test data loading functionality"""
    print("🧪 Testing data loading...")
    try:
        # Test with sample data
        df = pd.read_csv('daily-min-temperatures.csv', parse_dates=['Date'], index_col='Date')
        
        # Validate data structure
        assert 'Temp' in df.columns, "Temperature column not found"
        assert len(df) > 0, "Dataset is empty"
        assert df['Temp'].dtype in ['float64', 'int64'], "Temperature data should be numeric"
        
        print(f"✅ Data loaded successfully: {len(df)} records")
        print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Temperature range: {df['Temp'].min():.1f}°C to {df['Temp'].max():.1f}°C")
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        return False

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        import plotly.express as px
        from statsmodels.tsa.stattools import adfuller
        from pmdarima import auto_arima
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        print("✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {str(e)}")
        print("💡 Try running: pip install -r requirements.txt")
        return False

def test_arima_model():
    """Test ARIMA model functionality with sample data"""
    print("🧪 Testing ARIMA model...")
    try:
        # Create sample time series data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        temps = 20 + 5 * np.sin(np.arange(100) * 2 * np.pi / 365) + np.random.normal(0, 2, 100)
        sample_data = pd.Series(temps, index=dates)
        
        # Test stationarity check
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(sample_data)
        
        print(f"   ADF Statistic: {result[0]:.4f}")
        print(f"   P-value: {result[1]:.6f}")
        
        # Test ARIMA fitting (simplified)
        from pmdarima import auto_arima
        model = auto_arima(
            sample_data, 
            seasonal=False, 
            stepwise=True, 
            suppress_warnings=True,
            max_p=2, max_q=2, max_d=1,
            trace=False
        )
        
        # Test forecasting
        forecast, conf_int = model.predict(n_periods=7, return_conf_int=True)
        
        print(f"✅ ARIMA model test successful")
        print(f"   Model order: {model.order}")
        print(f"   Forecast generated for 7 periods")
        return True
        
    except Exception as e:
        print(f"❌ ARIMA model test failed: {str(e)}")
        return False

def test_metrics_calculation():
    """Test performance metrics calculation"""
    print("🧪 Testing metrics calculation...")
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Sample data
        y_true = np.array([20, 21, 19, 22, 18])
        y_pred = np.array([19.5, 21.2, 19.1, 21.8, 18.3])
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        print(f"✅ Metrics calculation successful")
        print(f"   RMSE: {rmse:.3f}")
        print(f"   MAE: {mae:.3f}")
        print(f"   R²: {r2:.3f}")
        print(f"   MAPE: {mape:.1f}%")
        return True
        
    except Exception as e:
        print(f"❌ Metrics calculation failed: {str(e)}")
        return False

def test_plotting():
    """Test plotting functionality"""
    print("🧪 Testing plotting...")
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        # Create sample plot
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        temps = 20 + np.random.normal(0, 2, 30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=temps, mode='lines', name='Temperature'))
        
        print("✅ Plotting test successful")
        return True
        
    except Exception as e:
        print(f"❌ Plotting test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🌡️ Temperature Forecast Dashboard - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Metrics Calculation", test_metrics_calculation),
        ("Plotting", test_plotting),
        ("ARIMA Model", test_arima_model),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("📊 Test Results Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application should work correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)