#!/usr/bin/env python3
"""
Enhanced launcher script for the Diabetes Prediction App
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn', 
        'plotly', 'seaborn', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Installing missing packages...")
        
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], check=True)
            logger.info("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    else:
        logger.info("✅ All dependencies are installed")
    
    return True

def check_model_files():
    """Check if model files exist, train if necessary"""
    logger.info("Checking model files...")
    
    model_files = ['logistic_model.pkl', 'scaler.pkl']
    missing_files = [f for f in model_files if not Path(f).exists()]
    
    if missing_files:
        logger.info(f"Missing model files: {missing_files}")
        logger.info("Training model...")
        
        try:
            subprocess.run([sys.executable, 'train_model.py'], check=True)
            logger.info("✅ Model trained successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to train model: {e}")
            return False
    else:
        logger.info("✅ Model files found")
    
    return True

def run_streamlit_app(port=8501, host='localhost', debug=False):
    """Run the Streamlit application"""
    logger.info(f"Starting Streamlit app on {host}:{port}")
    
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.port', str(port),
        '--server.address', host,
        '--server.headless', 'false',
        '--browser.gatherUsageStats', 'false'
    ]
    
    if debug:
        cmd.extend(['--logger.level', 'debug'])
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to run application: {e}")
        return False
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run the Enhanced Diabetes Prediction App')
    parser.add_argument('--port', '-p', type=int, default=8501,
                       help='Port to run the application on (default: 8501)')
    parser.add_argument('--host', default='localhost',
                       help='Host to run the application on (default: localhost)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip dependency and model checks')
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if not args.skip_checks:
        # Check dependencies
        if not check_dependencies():
            logger.error("❌ Dependency check failed")
            sys.exit(1)
        
        # Check model files
        if not check_model_files():
            logger.error("❌ Model check failed")
            sys.exit(1)
    
    # Run the application
    logger.info("🚀 Launching Enhanced Diabetes Prediction App v2.0")
    logger.info(f"📱 Access the app at: http://{args.host}:{args.port}")
    logger.info("🛑 Press Ctrl+C to stop the application")
    
    success = run_streamlit_app(args.port, args.host, args.debug)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()