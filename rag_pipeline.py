
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGPipeline:
    def __init__(self, model_path="fine_tuned_quote_model", data_path="rag_processed_quotes.csv"):
        self.model = SentenceTransformer(model_path)
        self.df = pd.read_csv(data_path)
        self.index = None
        self._build_index()

    def _build_index(self):
        # Ensure tags are handled correctly (they are already lists from data_preparation)
        # Convert tags from string representation of list to actual list
        self.df["tags"] = self.df["tags"].apply(lambda x: eval(x) if isinstance(x, str) else x)

        # Create a combined text for embedding
        self.df["combined_text"] = self.df.apply(
            lambda row: f'{row["cleaned_quote"]} {row["cleaned_author"]} {", ".join(row["tags"])}',
            axis=1
        )

        corpus_embeddings = self.model.encode(self.df["combined_text"].tolist(), show_progress_bar=True)
        embedding_dim = corpus_embeddings.shape[1]

        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity
        self.index.add(np.array(corpus_embeddings).astype("float32"))
        print(f"FAISS index built with {self.index.ntotal} embeddings.")

    def retrieve_quotes(self, query, k=5):
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding).astype("float32"), k)  # D: distances, I: indices

        retrieved_quotes = []
        for i, score in zip(I[0], D[0]):
            retrieved_quotes.append({
                "quote": self.df.iloc[i]["quote"],
                "author": self.df.iloc[i]["author"],
                "tags": self.df.iloc[i]["tags"],
                "similarity_score": 1 - score / (2 * np.max(D)) # Normalize to 0-1 range, higher is better
            })
        return retrieved_quotes

    def answer_query(self, query):
        # For this assignment, we will simply return the retrieved quotes
        # A full RAG implementation would use an LLM to synthesize an answer
        # based on the retrieved context.
        retrieved_quotes = self.retrieve_quotes(query)
        if not retrieved_quotes:
            return "No relevant quotes found."

        response = "Here are some relevant quotes:\n\n"
        for quote_info in retrieved_quotes:
            response += f'Quote: {quote_info["quote"]}\n'
            response += f'Author: {quote_info["author"]}\n'
            response += f'Tags: {", ".join(quote_info["tags"])}\n'
            response += f'Similarity: {quote_info["similarity_score"]:.4f}\n\n'
        return response

if __name__ == "__main__":
    # Example usage
    rag_pipeline = RAGPipeline()
    query = "quotes about hope by Oscar Wilde"
    answer = rag_pipeline.answer_query(query)
    print(answer)

    query_2 = "motivational quotes about success"
    answer_2 = rag_pipeline.answer_query(query_2)
    print(answer_2)


