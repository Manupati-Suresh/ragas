import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from rank_bm25 import BM25Okapi
import re

class RAGPipelineEnhanced:
    def __init__(self, model_path="fine_tuned_quote_model", data_path="rag_processed_quotes.csv"):
        self.model = SentenceTransformer(model_path)
        self.df = pd.read_csv(data_path)
        self.index = None
        self.bm25_index = None
        self.llm_tokenizer = None
        self.llm_model = None
        self._build_index()
        self._build_bm25_index()
        self._load_llm()

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

    def _build_bm25_index(self):
        """Build BM25 index for keyword-based retrieval"""
        # Tokenize the combined text for BM25
        tokenized_corpus = []
        for text in self.df["combined_text"]:
            # Simple tokenization - split by whitespace and remove punctuation
            tokens = re.findall(r'\b\w+\b', text.lower())
            tokenized_corpus.append(tokens)
        
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print(f"BM25 index built with {len(tokenized_corpus)} documents.")

    def _load_llm(self):
        """Load a lightweight LLM for answer generation"""
        try:
            # Using a small, efficient model for answer generation
            model_name = "microsoft/DialoGPT-small"
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if it doesn't exist
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            print(f"LLM loaded: {model_name}")
        except Exception as e:
            print(f"Error loading LLM: {e}")
            print("Falling back to simple answer formatting")
            self.llm_model = None
            self.llm_tokenizer = None

    def retrieve_quotes_bm25(self, query, k=5):
        """Retrieve quotes using BM25 keyword search"""
        # Tokenize query
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        retrieved_quotes = []
        for i in top_indices:
            if scores[i] > 0:  # Only include quotes with positive scores
                retrieved_quotes.append({
                    "quote": self.df.iloc[i]["quote"],
                    "author": self.df.iloc[i]["author"],
                    "tags": self.df.iloc[i]["tags"],
                    "bm25_score": scores[i],
                    "index": i
                })
        
        return retrieved_quotes

    def retrieve_quotes_semantic(self, query, k=5):
        """Retrieve quotes using semantic search (FAISS)"""
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding).astype("float32"), k)

        retrieved_quotes = []
        for i, score in zip(I[0], D[0]):
            retrieved_quotes.append({
                "quote": self.df.iloc[i]["quote"],
                "author": self.df.iloc[i]["author"],
                "tags": self.df.iloc[i]["tags"],
                "semantic_score": 1 - score / (2 * np.max(D)) if np.max(D) > 0 else 1.0,
                "index": i
            })
        return retrieved_quotes

    def retrieve_quotes_hybrid(self, query, k=5, alpha=0.5):
        """
        Hybrid retrieval combining BM25 and semantic search using Reciprocal Rank Fusion (RRF)
        
        Args:
            query: User query
            k: Number of quotes to retrieve
            alpha: Weight for combining scores (0.5 = equal weight)
        """
        # Get results from both methods
        bm25_results = self.retrieve_quotes_bm25(query, k=k*2)  # Get more to ensure diversity
        semantic_results = self.retrieve_quotes_semantic(query, k=k*2)
        
        # Create a dictionary to store combined scores
        combined_scores = {}
        
        # Add BM25 scores with RRF
        for rank, result in enumerate(bm25_results):
            idx = result["index"]
            rrf_score = 1 / (rank + 1)  # Reciprocal rank
            combined_scores[idx] = combined_scores.get(idx, 0) + alpha * rrf_score
        
        # Add semantic scores with RRF
        for rank, result in enumerate(semantic_results):
            idx = result["index"]
            rrf_score = 1 / (rank + 1)  # Reciprocal rank
            combined_scores[idx] = combined_scores.get(idx, 0) + (1 - alpha) * rrf_score
        
        # Sort by combined score and get top k
        sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:k]
        
        # Build final results
        hybrid_results = []
        for idx in sorted_indices:
            hybrid_results.append({
                "quote": self.df.iloc[idx]["quote"],
                "author": self.df.iloc[idx]["author"],
                "tags": self.df.iloc[idx]["tags"],
                "hybrid_score": combined_scores[idx],
                "index": idx
            })
        
        return hybrid_results

    def retrieve_quotes(self, query, k=5, method="hybrid"):
        """
        Main retrieval method that can use different strategies
        
        Args:
            query: User query
            k: Number of quotes to retrieve
            method: "semantic", "bm25", or "hybrid"
        """
        if method == "semantic":
            return self.retrieve_quotes_semantic(query, k)
        elif method == "bm25":
            return self.retrieve_quotes_bm25(query, k)
        else:  # hybrid
            return self.retrieve_quotes_hybrid(query, k)

    def generate_answer_with_llm(self, query, retrieved_quotes):
        """Generate answer using LLM based on retrieved quotes"""
        if self.llm_model is None or self.llm_tokenizer is None:
            return self._format_simple_answer(query, retrieved_quotes)
        
        try:
            # Construct context from retrieved quotes
            context = "Here are some relevant quotes:\n"
            for i, quote_info in enumerate(retrieved_quotes[:3]):  # Use top 3 quotes
                context += f"{i+1}. \"{quote_info['quote']}\" - {quote_info['author']}\n"
            
            # Create prompt for the LLM
            prompt = f"Based on the following quotes, provide a thoughtful response to the question: {query}\n\n{context}\n\nResponse:"
            
            # Tokenize and generate
            inputs = self.llm_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,  # Generate additional 100 tokens
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id
                )
            
            # Decode the response
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            generated_text = response[len(prompt):].strip()
            
            if generated_text:
                return f"Based on the retrieved quotes, here's a thoughtful response:\n\n{generated_text}\n\n" + self._format_simple_answer(query, retrieved_quotes)
            else:
                return self._format_simple_answer(query, retrieved_quotes)
                
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return self._format_simple_answer(query, retrieved_quotes)

    def _format_simple_answer(self, query, retrieved_quotes):
        """Fallback method to format answer without LLM"""
        if not retrieved_quotes:
            return "No relevant quotes found."

        response = f"Here are some relevant quotes for your query '{query}':\n\n"
        for i, quote_info in enumerate(retrieved_quotes):
            response += f"{i+1}. \"{quote_info['quote']}\"\n"
            response += f"   - {quote_info['author']}\n"
            response += f"   - Tags: {', '.join(quote_info['tags'])}\n"
            
            # Add score information based on available keys
            if 'hybrid_score' in quote_info:
                response += f"   - Relevance Score: {quote_info['hybrid_score']:.4f}\n"
            elif 'semantic_score' in quote_info:
                response += f"   - Semantic Score: {quote_info['semantic_score']:.4f}\n"
            elif 'bm25_score' in quote_info:
                response += f"   - BM25 Score: {quote_info['bm25_score']:.4f}\n"
            
            response += "\n"
        
        return response

    def answer_query(self, query, method="hybrid", use_llm=True):
        """
        Main method to answer user queries
        
        Args:
            query: User query
            method: Retrieval method ("semantic", "bm25", or "hybrid")
            use_llm: Whether to use LLM for answer generation
        """
        # Retrieve relevant quotes
        retrieved_quotes = self.retrieve_quotes(query, method=method)
        
        if use_llm:
            return self.generate_answer_with_llm(query, retrieved_quotes)
        else:
            return self._format_simple_answer(query, retrieved_quotes)

if __name__ == "__main__":
    # Example usage
    print("Initializing Enhanced RAG Pipeline...")
    rag_pipeline = RAGPipelineEnhanced()
    
    # Test queries
    test_queries = [
        "quotes about hope and inspiration",
        "what did Oscar Wilde say about life?",
        "motivational quotes about success"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        # Test different methods
        for method in ["semantic", "bm25", "hybrid"]:
            print(f"\n--- {method.upper()} RETRIEVAL ---")
            answer = rag_pipeline.answer_query(query, method=method, use_llm=True)
            print(answer)

