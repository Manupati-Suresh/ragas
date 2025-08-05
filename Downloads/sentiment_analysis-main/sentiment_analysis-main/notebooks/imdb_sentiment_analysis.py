from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib
import os

# Load IMDb dataset
print("Loading dataset...")
dataset = load_dataset("imdb")
train_texts = dataset['train']['text']
train_labels = dataset['train']['label']

# Create pipeline
pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english', max_features=20000),
    LogisticRegression(max_iter=200)
)

# Train model
print("Training model...")
pipeline.fit(train_texts, train_labels)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/imdb_pipeline.pkl")
print("Model saved!")