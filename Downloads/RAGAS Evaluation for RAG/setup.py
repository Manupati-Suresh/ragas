from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rag-quote-system",
    version="1.0.0",
    author="RAG Development Team",
    author_email="your.email@example.com",
    description="A comprehensive RAG system for quote retrieval and question answering using open source models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rag-quote-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-demo=demo_rag_system:main",
            "rag-test=test_rag_system:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.json", "*.txt"],
    },
    keywords="rag, retrieval, generation, quotes, nlp, machine learning, transformers",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/rag-quote-system/issues",
        "Source": "https://github.com/yourusername/rag-quote-system",
        "Documentation": "https://github.com/yourusername/rag-quote-system#readme",
    },
) 