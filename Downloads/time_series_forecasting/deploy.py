#!/usr/bin/env python3
"""
Deployment script for Temperature Forecast Dashboard
Handles setup, validation, and launch of the Streamlit application
"""

import subprocess
import sys
import os
import importlib.util
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_data_file():
    """Check if data file exists"""
    data_file = Path("daily-min-temperatures.csv")
    if data_file.exists():
        print("✅ Data file found")
        return True
    else:
        print("⚠️  Data file not found - you can upload your own data in the app")
        return True

def validate_installation():
    """Validate that all required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'statsmodels', 'pmdarima', 'sklearn'
    ]
    
    missing_packages = []
    for package in required_packages:
        if package == 'sklearn':
            package = 'scikit-learn'
        
        spec = importlib.util.find_spec(package.replace('-', '_'))
        if spec is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def launch_app():
    """Launch the Streamlit application"""
    print("🚀 Launching Temperature Forecast Dashboard...")
    print("📱 The app will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "temperature_forecast.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"❌ Failed to launch application: {e}")

def main():
    """Main deployment function"""
    print("🌡️ Temperature Forecast Dashboard - Deployment Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Validate installation
    if not validate_installation():
        print("🔄 Retrying installation...")
        if not install_requirements() or not validate_installation():
            sys.exit(1)
    
    # Check data file
    check_data_file()
    
    print("\n✅ Setup complete!")
    print("=" * 60)
    
    # Launch application
    launch_app()

if __name__ == "__main__":
    main()