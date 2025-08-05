# 🎬 Advanced IMDb Sentiment Analyzer

A sophisticated web application that analyzes movie review sentiments using Machine Learning. Built with Streamlit, this app provides real-time sentiment analysis with confidence scoring, detailed text feature analysis, and beautiful visualizations.

## ✨ Features

### 🎯 Core Functionality
- **Real-time Sentiment Analysis**: Instant classification of movie reviews as positive or negative
- **Confidence Scoring**: Get probability scores for predictions with visual confidence indicators
- **Text Preprocessing**: Automatic cleaning and preprocessing of input text
- **Sample Reviews**: Pre-loaded examples for quick testing

### 📊 Advanced Analytics
- **Interactive Visualizations**: Beautiful charts using Plotly for confidence scores and feature analysis
- **Text Feature Analysis**: Detailed breakdown of text characteristics including:
  - Word count and sentence analysis
  - Average word length
  - Punctuation usage (exclamations, questions)
  - Uppercase text ratio
- **Feature Radar Chart**: Visual representation of text features
- **Analysis History**: Track and visualize past analyses with timestamps

### 🎨 User Experience
- **Modern UI**: Clean, responsive design with gradient backgrounds and custom styling
- **Sidebar Controls**: Easy-to-use settings panel for customizing the analysis view
- **Real-time Metrics**: Live statistics about your input text
- **Loading Animations**: Smooth user experience with progress indicators

### 📈 Data Insights
- **Historical Tracking**: Keep track of all your analyses
- **Trend Visualization**: See sentiment patterns over time
- **Summary Statistics**: Quick overview of analysis history
- **Export-ready Data**: Structured data format for further analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd imdb-sentiment-analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model** (if not already done)
```bash
python notebooks/imdb_sentiment_analysis.py
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser** and navigate to `http://localhost:8501`

## 🛠️ Technical Details

### Model Architecture
- **Algorithm**: Logistic Regression with TF-IDF vectorization
- **Features**: 20,000 most important TF-IDF features
- **Training Data**: IMDb movie reviews dataset (50,000 reviews)
- **Preprocessing**: Stop word removal, text cleaning, HTML tag removal

### Performance Metrics
- **Accuracy**: ~88-90% on test data
- **Speed**: Real-time inference (<100ms per review)
- **Memory**: Efficient model size (~50MB)

### Technology Stack
- **Frontend**: Streamlit with custom CSS
- **ML Framework**: Scikit-learn
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: Joblib

## 📱 Usage Guide

### Basic Analysis
1. Enter or paste a movie review in the text area
2. Click "🔍 Analyze Sentiment" to get results
3. View the sentiment classification and confidence score

### Advanced Features
- **Confidence Scores**: Toggle to see prediction probabilities
- **Text Features**: Enable detailed text analysis
- **Sample Reviews**: Use pre-loaded examples for testing
- **Analysis History**: Track your analysis sessions

### Interpreting Results
- **Green (Positive)**: Review expresses positive sentiment
- **Red (Negative)**: Review expresses negative sentiment
- **Confidence Level**: 
  - High (>80%): Very reliable prediction
  - Moderate (60-80%): Good prediction
  - Low (<60%): Less reliable, review text manually

## 🎨 Customization

### Styling
The app uses custom CSS for enhanced visual appeal:
- Gradient backgrounds
- Rounded corners and shadows
- Color-coded sentiment indicators
- Responsive design

### Configuration
Modify these settings in the sidebar:
- Show/hide confidence scores
- Enable/disable text feature analysis
- Toggle analysis history
- Clear data and reset

## 📊 Data Analysis

### Text Features Analyzed
- **Word Count**: Total number of words
- **Sentence Count**: Number of sentences
- **Average Word Length**: Mean character count per word
- **Punctuation Usage**: Exclamation marks and questions
- **Text Style**: Uppercase character ratio

### Visualization Types
- **Bar Charts**: Confidence score comparison
- **Radar Charts**: Multi-dimensional feature analysis
- **Scatter Plots**: Historical trend analysis
- **Metrics Cards**: Key statistics display

## 🔧 Development

### Project Structure
```
├── app.py                          # Main Streamlit application
├── notebooks/
│   ├── imdb_sentiment_analysis.py  # Model training script
│   └── model/
│       └── imdb_pipeline.pkl       # Trained model file
├── requirements.txt                # Python dependencies
├── README.md                      # Documentation
└── .gitignore                     # Git ignore rules
```

### Adding New Features
1. **New Text Features**: Add analysis functions in `analyze_text_features()`
2. **Visualization**: Create new chart functions using Plotly
3. **Model Improvements**: Modify the training script for better performance
4. **UI Enhancements**: Update CSS styles and layout

### Error Handling
The app includes comprehensive error handling for:
- Missing model files
- Invalid input text
- Prediction failures
- File I/O operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Credits

- **Dataset**: IMDb movie reviews via Hugging Face Datasets
- **ML Framework**: Scikit-learn community
- **UI Framework**: Streamlit team
- **Visualizations**: Plotly developers

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check the documentation
- Review the code comments for implementation details

---

**Built with ❤️ for movie enthusiasts and ML practitioners**