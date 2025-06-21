
import pandas as pd
from sentence_transformers import SentenceTransformer

def model_fine_tuning_rag():
    df = pd.read_csv("rag_processed_quotes.csv")

    # Load a pre-trained model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Save the model to a local directory
    model_save_path = "./fine_tuned_quote_model"
    model.save(model_save_path)
    print(f"Fine-tuned model saved to {model_save_path}")

if __name__ == "__main__":
    model_fine_tuning_rag()


