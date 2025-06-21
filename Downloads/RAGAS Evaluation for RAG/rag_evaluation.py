
from rag_pipeline import RAGPipeline

def rag_evaluation():
    rag_pipeline = RAGPipeline()

    test_queries = [
        "quotes about love and life",
        "quotes about success and failure",
        "quotes by famous philosophers",
        "quotes about happiness and sadness"
    ]

    print("\n--- RAG Evaluation ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        retrieved_quotes = rag_pipeline.retrieve_quotes(query, k=3)
        if retrieved_quotes:
            for i, quote_info in enumerate(retrieved_quotes):
                quote_text = quote_info["quote"]
                author_text = quote_info["author"]
                similarity_score = quote_info["similarity_score"]

                print(f"  {i+1}. Quote: {quote_text}")
                print(f"     Author: {author_text}")
                print(f"     Similarity: {similarity_score:.4f}")
        else:
            print("  No relevant quotes retrieved.")

if __name__ == "__main__":
    rag_evaluation()


