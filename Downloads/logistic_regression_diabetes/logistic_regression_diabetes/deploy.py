#!/usr/bin/env python3
"""
Deployment script for the Enhanced Diabetes Prediction App
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AppDeployer:
    """Handles deployment of the diabetes prediction app"""
    
    def __init__(self, environment='local'):
        self.environment = environment
        self.project_root = Path(__file__).parent
        self.required_files = [
            'app.py',
            'config.py',
            'utils.py',
            'model_explainer.py',
            'data_analysis.py',
            'train_model.py',
            'requirements.txt',
            'diabetes.csv'
        ]
    
    def check_requirements(self):
        """Check if all required files exist"""
        logger.info("Checking requirements...")
        
        missing_files = []
        for file in self.required_files:
            if not (self.project_root / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False
        
        logger.info("✅ All required files present")
        return True
    
    def install_dependencies(self):
        """Install Python dependencies"""
        logger.info("Installing dependencies...")
        
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], check=True, cwd=self.project_root)
            logger.info("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def train_model(self):
        """Train the machine learning model"""
        logger.info("Training model...")
        
        try:
            subprocess.run([
                sys.executable, 'train_model.py'
            ], check=True, cwd=self.project_root)
            logger.info("✅ Model trained successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to train model: {e}")
            return False
    
    def run_tests(self):
        """Run the test suite"""
        logger.info("Running tests...")
        
        try:
            subprocess.run([
                sys.executable, 'test_app.py'
            ], check=True, cwd=self.project_root)
            logger.info("✅ All tests passed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Tests failed: {e}")
            return False
    
    def create_deployment_package(self):
        """Create deployment package"""
        logger.info("Creating deployment package...")
        
        package_dir = self.project_root / 'deployment_package'
        if package_dir.exists():
            shutil.rmtree(package_dir)
        
        package_dir.mkdir()
        
        # Copy essential files
        essential_files = [
            'app.py', 'config.py', 'utils.py', 'model_explainer.py',
            'data_analysis.py', 'requirements.txt', 'diabetes.csv',
            'logistic_model.pkl', 'scaler.pkl'
        ]
        
        for file in essential_files:
            src = self.project_root / file
            if src.exists():
                shutil.copy2(src, package_dir / file)
        
        # Create deployment info
        deployment_info = {
            'version': '2.0.0',
            'environment': self.environment,
            'created_at': str(pd.Timestamp.now()),
            'files': essential_files
        }
        
        with open(package_dir / 'deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info(f"✅ Deployment package created at {package_dir}")
        return True
    
    def deploy_local(self):
        """Deploy locally using Streamlit"""
        logger.info("Starting local deployment...")
        
        try:
            subprocess.run([
                sys.executable, '-m', 'streamlit', 'run', 'app.py',
                '--server.port', '8501',
                '--server.address', '0.0.0.0'
            ], cwd=self.project_root)
        except KeyboardInterrupt:
            logger.info("Local deployment stopped by user")
        except Exception as e:
            logger.error(f"❌ Local deployment failed: {e}")
            return False
        
        return True
    
    def deploy_docker(self):
        """Deploy using Docker"""
        logger.info("Creating Docker deployment...")
        
        dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
        
        with open(self.project_root / 'Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        # Create .dockerignore
        dockerignore_content = """
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis
.DS_Store
"""
        
        with open(self.project_root / '.dockerignore', 'w') as f:
            f.write(dockerignore_content)
        
        logger.info("✅ Docker files created")
        logger.info("To build and run with Docker:")
        logger.info("  docker build -t diabetes-predictor .")
        logger.info("  docker run -p 8501:8501 diabetes-predictor")
        
        return True
    
    def deploy_heroku(self):
        """Create Heroku deployment files"""
        logger.info("Creating Heroku deployment files...")
        
        # Create Procfile
        with open(self.project_root / 'Procfile', 'w') as f:
            f.write('web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0\n')
        
        # Create setup.sh
        setup_content = """mkdir -p ~/.streamlit/

echo "\\
[general]\\n\\
email = \\"your-email@domain.com\\"\\n\\
" > ~/.streamlit/credentials.toml

echo "\\
[server]\\n\\
headless = true\\n\\
enableCORS=false\\n\\
port = $PORT\\n\\
" > ~/.streamlit/config.toml
"""
        
        with open(self.project_root / 'setup.sh', 'w') as f:
            f.write(setup_content)
        
        os.chmod(self.project_root / 'setup.sh', 0o755)
        
        logger.info("✅ Heroku deployment files created")
        logger.info("To deploy to Heroku:")
        logger.info("  1. heroku create your-app-name")
        logger.info("  2. git add .")
        logger.info("  3. git commit -m 'Deploy to Heroku'")
        logger.info("  4. git push heroku main")
        
        return True
    
    def full_deployment(self):
        """Run full deployment process"""
        logger.info("Starting full deployment process...")
        
        steps = [
            ("Checking requirements", self.check_requirements),
            ("Installing dependencies", self.install_dependencies),
            ("Training model", self.train_model),
            ("Running tests", self.run_tests),
            ("Creating deployment package", self.create_deployment_package)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}")
            if not step_func():
                logger.error(f"❌ Deployment failed at step: {step_name}")
                return False
        
        # Environment-specific deployment
        if self.environment == 'local':
            self.deploy_local()
        elif self.environment == 'docker':
            self.deploy_docker()
        elif self.environment == 'heroku':
            self.deploy_heroku()
        
        logger.info("✅ Deployment process completed successfully!")
        return True

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Deploy the Enhanced Diabetes Prediction App')
    parser.add_argument('--environment', '-e', 
                       choices=['local', 'docker', 'heroku'], 
                       default='local',
                       help='Deployment environment')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip running tests')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training')
    
    args = parser.parse_args()
    
    deployer = AppDeployer(args.environment)
    
    # Override methods if skipping
    if args.skip_tests:
        deployer.run_tests = lambda: True
    if args.skip_training:
        deployer.train_model = lambda: True
    
    success = deployer.full_deployment()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()