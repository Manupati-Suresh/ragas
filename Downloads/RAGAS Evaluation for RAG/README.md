# RAG System with Open Source Models

A comprehensive Retrieval-Augmented Generation (RAG) system for quote retrieval and question answering, built entirely with open source models and no external API dependencies.

## 🚀 Features

- **Multiple Retrieval Methods**: Semantic search, BM25, contextual search, and hybrid approaches
- **Open Source LLM**: Uses Microsoft DialoGPT for answer generation
- **Advanced Evaluation**: Custom evaluation framework and RAGAS integration
- **No API Dependencies**: Works completely offline with local models
- **Comprehensive Testing**: Built-in test suite and demo scripts

## 📋 Requirements

- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- Internet connection for initial model downloads

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd RAG-Evaluation-for-RAG
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (if not already downloaded):
   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('punkt')
   ```

## 📁 Project Structure

```
RAG-Evaluation-for-RAG/
├── rag_pipeline_enhanced.py      # Main RAG pipeline
├── custom_rag_evaluator.py       # Custom evaluation framework
├── ragas_evaluator.py           # RAGAS evaluation integration
├── comprehensive_rag_evaluation.py # Comprehensive evaluation
├── test_rag_system.py           # Test suite
├── demo_rag_system.py           # Demo script
├── requirements.txt             # Dependencies
├── rag_processed_quotes.csv     # Quote dataset
└── fine_tuned_quote_model/      # Fine-tuned sentence transformer
```

## 🎯 Quick Start

### 1. Test the System

Run the comprehensive test suite:

```bash
python test_rag_system.py
```

### 2. Try the Demo

Run the interactive demo:

```bash
python demo_rag_system.py
```

### 3. Use the RAG Pipeline

```python
from rag_pipeline_enhanced import RAGPipelineEnhanced

# Initialize the pipeline
rag_pipeline = RAGPipelineEnhanced()

# Ask a question
query = "What are some inspiring quotes about hope?"
answer = rag_pipeline.answer_query(query, method="hybrid", use_llm=True)
print(answer)
```

## 🔍 Retrieval Methods

The system supports multiple retrieval strategies:

1. **Semantic Search**: Uses sentence transformers for semantic similarity
2. **BM25**: Traditional keyword-based retrieval
3. **Contextual Search**: Context-aware retrieval
4. **Keyword Advanced**: Combines BM25 and TF-IDF
5. **Hybrid**: Combines all methods for best results

## 🤖 Answer Generation

The system can generate answers in two modes:

1. **Simple Mode** (`use_llm=False`): Formats retrieved quotes without LLM
2. **LLM Mode** (`use_llm=True`): Uses Microsoft DialoGPT for enhanced responses

## 📊 Evaluation

### Custom Evaluation

```python
from custom_rag_evaluator import CustomRAGEvaluator

# Initialize evaluator
evaluator = CustomRAGEvaluator(rag_pipeline)

# Create evaluation dataset
dataset = evaluator.create_evaluation_dataset(num_samples=30)

# Run evaluation
results = evaluator.run_evaluation(dataset)
evaluator.analyze_results()
```

### RAGAS Evaluation

```python
from ragas_evaluator import RAGASEvaluator

# Initialize RAGAS evaluator
evaluator = RAGASEvaluator(rag_pipeline)

# Create dataset and run evaluation
dataset = evaluator.create_evaluation_dataset(num_samples=20)
results = evaluator.run_evaluation(dataset)
```

## 🔧 Configuration

### Model Configuration

The system uses these open source models:

- **Sentence Transformer**: Fine-tuned model for embeddings
- **Cross-Encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2` for re-ranking
- **LLM**: `microsoft/DialoGPT-small` for answer generation

### Performance Tuning

You can adjust various parameters:

```python
# Adjust hybrid retrieval weights
results = rag_pipeline.retrieve_quotes_hybrid(
    query, 
    k=5, 
    alpha=0.4,  # Semantic weight
    beta=0.3,   # Contextual weight
    gamma=0.3   # Keyword weight
)

# Adjust LLM generation parameters
# (These are set in the generate_answer_with_llm method)
```

## 🧪 Testing

The test suite covers:

- **Data Integrity**: Verifies dataset and model files
- **RAG Pipeline**: Tests all retrieval methods and answer generation
- **Custom Evaluator**: Tests the evaluation framework
- **RAGAS Evaluator**: Tests RAGAS integration

Run tests:

```bash
python test_rag_system.py
```

## 📈 Performance

Typical performance metrics:

- **Initialization**: 10-30 seconds (depending on hardware)
- **Query Processing**: 0.1-2 seconds per query
- **LLM Generation**: 1-5 seconds per answer
- **Memory Usage**: 2-4 GB RAM

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure sufficient disk space for model downloads
   - Check internet connection for initial downloads
   - Try running with `use_llm=False` if LLM fails

2. **Memory Issues**:
   - Reduce batch sizes in the pipeline
   - Use smaller models (already configured)
   - Close other applications to free memory

3. **RAGAS Evaluation Fails**:
   - This is expected if RAGAS models aren't available
   - Use the custom evaluator as an alternative
   - Check RAGAS documentation for setup

### Error Messages

- `"LLM generation failed"`: Falls back to simple formatting
- `"RAGAS evaluation failed"`: Use custom evaluator instead
- `"Model loading error"`: Check dependencies and internet connection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Sentence Transformers**: For semantic search capabilities
- **Microsoft**: For DialoGPT models
- **RAGAS**: For evaluation framework
- **Hugging Face**: For model hosting and transformers library

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Run the test suite to identify problems
3. Review the demo script for usage examples
4. Open an issue with detailed error information

---

**Note**: This system is designed to work completely offline with open source models. No API keys or external services are required. 