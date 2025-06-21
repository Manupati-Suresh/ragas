
from datasets import load_dataset
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

def data_preparation_rag():
    # Load the dataset
    dataset = load_dataset("Abirate/english_quotes")
    df = dataset["train"].to_pandas()

    # Preprocessing function
    def preprocess_quote(text):
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special characters and numbers
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    # Apply preprocessing to 'quote' and 'author' columns
    df['cleaned_quote'] = df['quote'].apply(preprocess_quote)
    df['cleaned_author'] = df['author'].apply(preprocess_quote)

    # Handle missing values in 'tags' column (if any) by filling with empty list string
    df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])

    print("Original DataFrame head:")
    print(df[['quote', 'author', 'tags']].head())
    print("\nCleaned DataFrame head:")
    print(df[['cleaned_quote', 'cleaned_author', 'tags']].head())
    print("\nDataFrame Info:")
    print(df.info())

    df.to_csv("rag_processed_quotes.csv", index=False)
    print("Processed RAG data saved to rag_processed_quotes.csv")
    return df

if __name__ == "__main__":
    data_preparation_rag()


