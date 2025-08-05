# Configuration file for IMDb Sentiment Analyzer

# Model Configuration
MODEL_PATH = "notebooks/model/imdb_pipeline.pkl"
MODEL_MAX_FEATURES = 20000
MODEL_MAX_ITER = 200

# UI Configuration
APP_TITLE = "Advanced IMDb Sentiment Analyzer"
APP_ICON = "🎬"
PAGE_LAYOUT = "wide"

# Analysis Configuration
CONFIDENCE_THRESHOLDS = {
    'high': 0.8,
    'moderate': 0.6,
    'low': 0.0
}

# Text Processing Configuration
MAX_TEXT_LENGTH = 10000
MIN_TEXT_LENGTH = 10

# Visualization Configuration
CHART_HEIGHT = 300
RADAR_CHART_HEIGHT = 400
HISTORY_CHART_HEIGHT = 400

# Color Scheme
COLORS = {
    'positive': '#56ab2f',
    'negative': '#ef473a',
    'neutral': '#f7971e',
    'primary': '#667eea',
    'secondary': '#764ba2'
}

# Sample Reviews
SAMPLE_REVIEWS = {
    "Positive Example": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout. Highly recommended!",
    "Negative Example": "Terrible movie with poor acting and a confusing plot. I wasted my time watching this garbage.",
    "Mixed Example": "The movie had some good moments but overall it was just okay. Not the best but not the worst either.",
    "Action Movie": "Amazing action sequences and stunning visual effects! The fight scenes were choreographed perfectly and kept me on the edge of my seat.",
    "Drama Review": "A deeply moving story with incredible performances. The emotional depth and character development were outstanding.",
    "Comedy Review": "Hilarious from start to finish! Great comedic timing and witty dialogue that had me laughing throughout."
}

# Feature Analysis Configuration
FEATURE_NORMALIZATION = {
    'word_count_scale': 10,
    'max_sentences': 10,
    'max_word_length': 10,
    'max_punctuation': 10
}