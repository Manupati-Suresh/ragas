#!/usr/bin/env python3
"""
Deployment script for RAG Quote System.
This script helps prepare the project for GitHub deployment.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_git_status():
    """Check if git is initialized and has changes."""
    if not os.path.exists('.git'):
        print("❌ Git repository not initialized")
        return False
    
    result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    if result.stdout.strip():
        print("📝 Changes detected in repository")
        return True
    else:
        print("✅ No changes detected")
        return False

def create_initial_commit():
    """Create initial commit if needed."""
    if not os.path.exists('.git'):
        print("🔄 Initializing git repository...")
        run_command('git init', "Git initialization")
    
    # Add all files
    run_command('git add .', "Adding files to git")
    
    # Check if there are changes to commit
    result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    if result.stdout.strip():
        run_command('git commit -m "Initial commit: RAG Quote System with open source models"', "Creating initial commit")
        return True
    else:
        print("✅ No changes to commit")
        return False

def setup_remote_repository():
    """Setup remote repository."""
    print("\n🌐 GitHub Repository Setup")
    print("="*50)
    print("To complete the deployment, you need to:")
    print("1. Create a new repository on GitHub")
    print("2. Copy the repository URL")
    print("3. Run the following commands:")
    print()
    print("git remote add origin <YOUR_REPOSITORY_URL>")
    print("git branch -M main")
    print("git push -u origin main")
    print()
    
    repo_url = input("Enter your GitHub repository URL (or press Enter to skip): ").strip()
    if repo_url:
        run_command(f'git remote add origin {repo_url}', "Adding remote origin")
        run_command('git branch -M main', "Setting main branch")
        run_command('git push -u origin main', "Pushing to GitHub")
        return True
    return False

def create_release_notes():
    """Create release notes for the initial version."""
    release_notes = """# Release v1.0.0 - Initial Release

## 🎉 What's New

This is the initial release of the RAG Quote System, a comprehensive Retrieval-Augmented Generation system built entirely with open source models.

### ✨ Features

- **Multiple Retrieval Methods**: Semantic search, BM25, contextual search, and hybrid approaches
- **Open Source LLM**: Microsoft DialoGPT for answer generation
- **Advanced Evaluation**: Custom evaluation framework and RAGAS integration
- **No API Dependencies**: Works completely offline with local models
- **Comprehensive Testing**: Built-in test suite and demo scripts

### 🔧 Technical Details

- **Python**: 3.8+
- **Memory**: 2-4 GB RAM recommended
- **Models**: All open source (no API keys required)
- **Performance**: 0.1-2 seconds per query

### 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python test_rag_system.py

# Try the demo
python demo_rag_system.py
```

### 📊 System Statistics

- **Total Quotes**: 2,508
- **Retrieval Methods**: 5 (semantic, BM25, contextual, keyword advanced, hybrid)
- **Evaluation Metrics**: 5 (context relevance, answer relevance, precision, recall, faithfulness)

### 🐛 Known Issues

- RAGAS evaluation may fail if models are not properly configured (expected behavior)
- LLM generation may fall back to simple formatting if model loading fails

### 🔮 Future Plans

- Add more evaluation metrics
- Support for additional languages
- Web interface improvements
- Model fine-tuning capabilities

---

**Note**: This system is designed to work completely offline with open source models. No API keys or external services are required.
"""
    
    with open('RELEASE_NOTES.md', 'w', encoding='utf-8') as f:
        f.write(release_notes)
    print("✅ Release notes created: RELEASE_NOTES.md")

def main():
    """Main deployment function."""
    print("🚀 RAG Quote System - Deployment Script")
    print("="*50)
    
    # Check current directory
    if not os.path.exists('rag_pipeline_enhanced.py'):
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Create release notes
    create_release_notes()
    
    # Check git status
    has_changes = check_git_status()
    
    # Create initial commit if needed
    if has_changes:
        create_initial_commit()
    
    # Setup remote repository
    setup_remote_repository()
    
    print("\n🎉 Deployment preparation completed!")
    print("="*50)
    print("Next steps:")
    print("1. Review the files in your repository")
    print("2. Update the repository URL in setup.py and pyproject.toml")
    print("3. Update your email in the configuration files")
    print("4. Push to GitHub when ready")
    print("5. Create a release on GitHub with the release notes")
    
    print("\n📋 Files created/modified:")
    files = [
        '.gitignore',
        'setup.py', 
        'pyproject.toml',
        'LICENSE',
        'CONTRIBUTING.md',
        'CODE_OF_CONDUCT.md',
        '.github/workflows/ci.yml',
        'RELEASE_NOTES.md'
    ]
    
    for file in files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (missing)")

if __name__ == "__main__":
    main() 