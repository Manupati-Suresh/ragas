#!/usr/bin/env python3
"""
Setup script for IMDb Sentiment Analyzer
Handles installation, model training, and verification
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   Please use Python 3.7 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    # Install packages
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    )

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "notebooks/model",
        "data",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def train_model():
    """Train the sentiment analysis model"""
    print("🤖 Training model...")
    
    model_path = "notebooks/model/imdb_pipeline.pkl"
    
    # Check if model already exists
    if os.path.exists(model_path):
        response = input("Model already exists. Retrain? (y/N): ").lower()
        if response != 'y':
            print("✅ Using existing model")
            return True
    
    # Check if training script exists
    training_script = "notebooks/imdb_sentiment_analysis.py"
    if not os.path.exists(training_script):
        print(f"❌ Training script not found: {training_script}")
        return False
    
    # Run training
    print("⏳ This may take several minutes...")
    success = run_command(
        f"{sys.executable} {training_script}",
        "Training sentiment analysis model"
    )
    
    if success and os.path.exists(model_path):
        print("✅ Model trained and saved successfully")
        return True
    else:
        print("❌ Model training failed")
        return False

def verify_installation():
    """Verify that everything is working"""
    print("🧪 Verifying installation...")
    
    # Run test script
    if os.path.exists("test_app.py"):
        return run_command(
            f"{sys.executable} test_app.py",
            "Running verification tests"
        )
    else:
        print("⚠️ Test script not found, skipping verification")
        return True

def create_launch_script():
    """Create a launch script for easy startup"""
    print("🚀 Creating launch script...")
    
    if os.name == 'nt':  # Windows
        script_content = f"""@echo off
echo Starting IMDb Sentiment Analyzer...
{sys.executable} -m streamlit run app.py
pause
"""
        script_name = "launch.bat"
    else:  # Unix/Linux/Mac
        script_content = f"""#!/bin/bash
echo "Starting IMDb Sentiment Analyzer..."
{sys.executable} -m streamlit run app.py
"""
        script_name = "launch.sh"
    
    with open(script_name, "w") as f:
        f.write(script_content)
    
    if os.name != 'nt':
        os.chmod(script_name, 0o755)
    
    print(f"✅ Created launch script: {script_name}")
    return True

def main():
    """Main setup process"""
    print("🎬 IMDb Sentiment Analyzer - Setup")
    print("=" * 50)
    
    steps = [
        ("Python Version Check", check_python_version),
        ("Directory Creation", create_directories),
        ("Dependency Installation", install_dependencies),
        ("Model Training", train_model),
        ("Installation Verification", verify_installation),
        ("Launch Script Creation", create_launch_script)
    ]
    
    for step_name, step_func in steps:
        print(f"\n🔍 {step_name}...")
        
        try:
            if not step_func():
                print(f"\n❌ Setup failed at: {step_name}")
                print("Please fix the issues above and run setup again.")
                return 1
        except KeyboardInterrupt:
            print(f"\n⚠️ Setup interrupted by user")
            return 1
        except Exception as e:
            print(f"\n❌ Unexpected error in {step_name}: {e}")
            return 1
    
    # Success message
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nTo start the application:")
    print("  Option 1: streamlit run app.py")
    
    if os.name == 'nt':
        print("  Option 2: Double-click launch.bat")
    else:
        print("  Option 2: ./launch.sh")
    
    print("\n📖 For more information, see README.md")
    
    # Ask if user wants to start the app now
    response = input("\nStart the application now? (Y/n): ").lower()
    if response != 'n':
        print("\n🚀 Starting application...")
        time.sleep(2)
        os.system(f"{sys.executable} -m streamlit run app.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())