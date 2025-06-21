

## Architectural Changes for LLM Augmentation and BM25 Integration

### LLM for Answer Generation

To integrate an LLM for generating answers based on retrieved quotes, the `RAGPipeline` class in `rag_pipeline.py` will be modified. Currently, the `answer_query` method simply formats and returns the retrieved quotes. This will be updated to:

1.  **Load an LLM:** A suitable open-source LLM (e.g., from Hugging Face Transformers) will be loaded. Considerations will be given to model size and computational requirements for efficient execution within the sandbox.
2.  **Construct a Prompt:** The retrieved quotes, along with the user's query, will be used to construct a prompt for the LLM. This prompt will instruct the LLM to synthesize a concise and coherent answer based on the provided context.
3.  **Generate Answer:** The LLM will generate an answer based on the constructed prompt.
4.  **Return Generated Answer:** The `answer_query` method will return the LLM-generated answer.

### BM25 for Hybrid Search

To implement BM25 for hybrid search, the `RAGPipeline` class will be further enhanced:

1.  **BM25 Indexing:** A BM25 index will be created for the `cleaned_quote` and `cleaned_author` fields of the `rag_processed_quotes.csv` dataset. This will involve using a library like `rank_bm25`.
2.  **BM25 Retrieval:** A new method, `retrieve_quotes_bm25`, will be added to the `RAGPipeline` class. This method will perform BM25 retrieval based on the user's query and return a ranked list of quotes.
3.  **Hybrid Search Logic:** The `retrieve_quotes` method will be modified to incorporate hybrid search. This will involve:
    *   Performing both FAISS (semantic) retrieval and BM25 (keyword) retrieval.
    *   Combining the results from both retrieval methods. A common approach is to re-rank the combined results based on a weighted sum of their similarity scores (e.g., RRF - Reciprocal Rank Fusion).
    *   Returning the top `k` quotes from the hybrid search.

### Modified `RAGPipeline` Structure (Conceptual)

```python
class RAGPipeline:
    def __init__(self, ...):
        # ... existing initialization ...
        self.bm25_index = None
        self.llm = None # For answer generation
        self._build_bm25_index()
        self._load_llm()

    def _build_bm25_index(self):
        # Logic to build BM25 index
        pass

    def _load_llm(self):
        # Logic to load LLM for answer generation
        pass

    def retrieve_quotes_bm25(self, query, k=5):
        # Logic for BM25 retrieval
        pass

    def retrieve_quotes(self, query, k=5): # Modified for hybrid search
        # Perform FAISS retrieval
        # Perform BM25 retrieval
        # Combine and re-rank results (e.g., RRF)
        pass

    def answer_query(self, query): # Modified to use LLM for answer generation
        retrieved_quotes = self.retrieve_quotes(query)
        # Construct prompt for LLM using retrieved_quotes and query
        # Generate answer using LLM
        # Return LLM-generated answer
        pass
```

This architectural plan ensures that the existing semantic search capabilities are retained while enhancing the RAG pipeline with keyword-based retrieval and LLM-powered answer generation.

