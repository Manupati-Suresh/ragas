import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from rank_bm25 import BM25Okapi
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import logging
from typing import List, Dict, Optional

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class RAGPipelineEnhanced:
    def __init__(self, model_path="fine_tuned_quote_model", data_path="rag_processed_quotes.csv"):
        # Setup logging
        self.logger = self._setup_logging()
        self.logger.info("Initializing Enhanced RAG Pipeline...")
        
        self.model = SentenceTransformer(model_path)
        self.df = pd.read_csv(data_path)
        self.index = None
        self.bm25_index = None
        self.llm_tokenizer = None
        self.llm_model = None
        
        # Advanced components
        self.cross_encoder = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.contextual_index = None
        self.query_expansion_cache = {}
        
        # Initialize all components
        self._build_advanced_indices()
        self._load_advanced_models()
        self._setup_query_expansion()
        
        self.logger.info("Enhanced RAG Pipeline initialized successfully")

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def _build_advanced_indices(self):
        """Build multiple advanced indices for different retrieval strategies"""
        self.logger.info("Building advanced indices...")
        
        # Ensure tags are handled correctly
        self.df["tags"] = self.df["tags"].apply(lambda x: eval(x) if isinstance(x, str) else x)

        # Create multiple text representations for different retrieval strategies
        self._create_enhanced_text_representations()
        
        # Build semantic index (FAISS)
        self._build_semantic_index()
        
        # Build keyword index (BM25)
        self._build_bm25_index()
        
        # Build TF-IDF index
        self._build_tfidf_index()
        
        # Build contextual index
        self._build_contextual_index()

    def _create_enhanced_text_representations(self):
        """Create multiple text representations for different retrieval strategies"""
        self.logger.info("Creating enhanced text representations...")
        
        # Basic combined text (original)
        self.df["combined_text"] = self.df.apply(
            lambda row: f'{row["cleaned_quote"]} {row["cleaned_author"]} {", ".join(row["tags"])}',
            axis=1
        )
        
        # Contextual text with structured format
        self.df["contextual_text"] = self.df.apply(
            lambda row: f'Quote: {row["quote"]} Author: {row["author"]} Topics: {", ".join(row["tags"])}',
            axis=1
        )
        
        # Semantic text for better semantic search
        self.df["semantic_text"] = self.df.apply(
            lambda row: f'{row["quote"]} by {row["author"]} about {", ".join(row["tags"])}',
            axis=1
        )
        
        # Keyword-focused text
        self.df["keyword_text"] = self.df.apply(
            lambda row: f'{row["cleaned_quote"]} {row["cleaned_author"]} {" ".join(row["tags"])}',
            axis=1
        )

    def _build_semantic_index(self):
        """Build enhanced FAISS index for semantic search"""
        self.logger.info("Building semantic index...")
        
        # Encode semantic text
        semantic_embeddings = self.model.encode(
            self.df["semantic_text"].tolist(), 
            show_progress_bar=True,
            batch_size=32
        )
        
        embedding_dim = semantic_embeddings.shape[1]
        
        # Use IVF index for better performance with large datasets
        nlist = min(100, len(self.df) // 30)  # Number of clusters
        
        if len(self.df) > 1000:
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatL2(embedding_dim), 
                embedding_dim, 
                nlist
            )
            # Train the index
            if hasattr(self.index, 'train'):
                self.index.train(semantic_embeddings.astype('float32'))
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)
        
        if hasattr(self.index, 'add'):
            self.index.add(semantic_embeddings.astype('float32'))
        self.logger.info(f"Semantic index built with {self.index.ntotal} embeddings")

    def _build_bm25_index(self):
        """Build BM25 index for keyword-based retrieval"""
        self.logger.info("Building BM25 index...")
        
        tokenized_corpus = []
        for text in self.df["keyword_text"]:
            tokens = re.findall(r'\b\w+\b', text.lower())
            tokenized_corpus.append(tokens)
        
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.logger.info(f"BM25 index built with {len(tokenized_corpus)} documents")

    def _build_tfidf_index(self):
        """Build TF-IDF index for additional keyword features"""
        self.logger.info("Building TF-IDF index...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df["keyword_text"])
        if self.tfidf_matrix is not None and hasattr(self.tfidf_matrix, 'shape'):
            self.logger.info(f"TF-IDF index built with {self.tfidf_matrix.shape[1]} features")
        else:
            self.logger.info("TF-IDF index built")

    def _build_contextual_index(self):
        """Build contextual index for context-aware retrieval"""
        self.logger.info("Building contextual index...")
        
        # Encode contextual text
        contextual_embeddings = self.model.encode(
            self.df["contextual_text"].tolist(),
            show_progress_bar=True,
            batch_size=32
        )
        
        embedding_dim = contextual_embeddings.shape[1]
        self.contextual_index = faiss.IndexFlatL2(embedding_dim)
        if hasattr(self.contextual_index, 'add'):
            self.contextual_index.add(contextual_embeddings.astype('float32'))
        self.logger.info(f"Contextual index built with {self.contextual_index.ntotal} embeddings")

    def _load_advanced_models(self):
        """Load advanced models including cross-encoder and LLM"""
        self.logger.info("Loading advanced models...")
        
        # Load cross-encoder for re-ranking
        try:
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.logger.info("Cross-encoder loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load cross-encoder: {e}")
            self.cross_encoder = None
        
        # Load LLM
        self._load_llm()

    def _load_llm(self):
        """Load a lightweight LLM for answer generation"""
        try:
            # Using a more reliable and smaller model for better compatibility
            model_name = "microsoft/DialoGPT-small"  # Smaller model for better compatibility
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if it doesn't exist
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            # Move model to CPU to avoid GPU memory issues
            self.llm_model = self.llm_model.cpu()
            
            self.logger.info(f"LLM loaded: {model_name}")
        except Exception as e:
            self.logger.warning(f"Error loading LLM: {e}")
            self.logger.info("Continuing without LLM - will use simple answer formatting")
            self.llm_model = None
            self.llm_tokenizer = None

    def _setup_query_expansion(self):
        """Setup query expansion using WordNet"""
        self.logger.info("Setting up query expansion...")
        self.query_expansion_cache = {}

    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        Expand query using WordNet synonyms and related terms
        
        Args:
            query: Original query
            max_expansions: Maximum number of expansion terms per word
            
        Returns:
            List of expanded query terms
        """
        if query in self.query_expansion_cache:
            return self.query_expansion_cache[query]
        
        expanded_terms = []
        tokens = word_tokenize(query.lower())
        
        for token in tokens:
            if len(token) < 3:  # Skip very short tokens
                continue
                
            # Get synonyms from WordNet
            synonyms = []
            synsets = wordnet.synsets(token)
            for syn in synsets:
                if syn is not None:
                    for lemma in syn.lemmas():
                        if lemma.name() != token and len(lemma.name()) > 2:
                            synonyms.append(lemma.name())
                            if len(synonyms) >= max_expansions:
                                break
                    if len(synonyms) >= max_expansions:
                        break
            
            expanded_terms.extend(synonyms[:max_expansions])
        
        # Add original query
        expanded_terms.insert(0, query)
        
        # Cache the result
        self.query_expansion_cache[query] = expanded_terms
        
        return expanded_terms

    def retrieve_quotes_bm25(self, query, k=5):
        """Retrieve quotes using BM25 keyword search"""
        if self.bm25_index is None:
            return []
            
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
        """Retrieve quotes using semantic search (FAISS) with query expansion"""
        if self.index is None:
            return []
            
        # Expand query for better semantic matching
        expanded_queries = self.expand_query(query)
        
        all_results = []
        
        for expanded_query in expanded_queries:
            query_embedding = self.model.encode([expanded_query])
            if hasattr(self.index, 'search'):
                D, I = self.index.search(np.array(query_embedding).astype("float32"), k)

                for i, score in zip(I[0], D[0]):
                    if i < len(self.df):  # Ensure valid index
                        all_results.append({
                            "quote": self.df.iloc[i]["quote"],
                            "author": self.df.iloc[i]["author"],
                            "tags": self.df.iloc[i]["tags"],
                            "semantic_score": 1 / (1 + score),  # Convert distance to similarity
                            "index": i,
                            "query": expanded_query
                        })
        
        # Remove duplicates and sort by score
        unique_results = {}
        for result in all_results:
            idx = result["index"]
            if idx not in unique_results or result["semantic_score"] > unique_results[idx]["semantic_score"]:
                unique_results[idx] = result
        
        # Sort by score and return top k
        sorted_results = sorted(unique_results.values(), key=lambda x: x["semantic_score"], reverse=True)
        return sorted_results[:k]

    def retrieve_quotes_contextual(self, query, k=5):
        """Retrieve quotes using contextual search"""
        if self.contextual_index is None:
            return []
            
        query_embedding = self.model.encode([query])
        if hasattr(self.contextual_index, 'search'):
            D, I = self.contextual_index.search(np.array(query_embedding).astype("float32"), k)

            retrieved_quotes = []
            for i, score in zip(I[0], D[0]):
                if i < len(self.df):
                    retrieved_quotes.append({
                        "quote": self.df.iloc[i]["quote"],
                        "author": self.df.iloc[i]["author"],
                        "tags": self.df.iloc[i]["tags"],
                        "contextual_score": 1 / (1 + score),
                        "index": i
                    })
            return retrieved_quotes
        return []

    def retrieve_quotes_keyword_advanced(self, query, k=5):
        """Advanced keyword retrieval combining BM25 and TF-IDF"""
        if self.bm25_index is None or self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []
            
        # BM25 retrieval
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # TF-IDF retrieval
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        
        # Combine scores
        combined_scores = 0.6 * bm25_scores + 0.4 * tfidf_similarities
        
        # Get top k indices
        top_indices = np.argsort(combined_scores)[::-1][:k]
        
        results = []
        for i in top_indices:
            if combined_scores[i] > 0:
                results.append({
                    "quote": self.df.iloc[i]["quote"],
                    "author": self.df.iloc[i]["author"],
                    "tags": self.df.iloc[i]["tags"],
                    "keyword_score": combined_scores[i],
                    "bm25_score": bm25_scores[i],
                    "tfidf_score": tfidf_similarities[i],
                    "index": i
                })
        
        return results

    def re_rank_with_cross_encoder(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Re-rank candidates using cross-encoder
        
        Args:
            query: User query
            candidates: List of candidate quotes
            top_k: Number of top results to return
            
        Returns:
            Re-ranked list of quotes
        """
        if self.cross_encoder is None or not candidates:
            return candidates[:top_k]
        
        try:
            # Prepare pairs for cross-encoder
            pairs = []
            for candidate in candidates:
                text = f"{candidate['quote']} by {candidate['author']}"
                pairs.append([query, text])
            
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)
            
            # Add scores to candidates
            for candidate, score in zip(candidates, scores):
                candidate['cross_encoder_score'] = float(score)
            
            # Sort by cross-encoder score
            re_ranked = sorted(candidates, key=lambda x: x['cross_encoder_score'], reverse=True)
            
            return re_ranked[:top_k]
            
        except Exception as e:
            self.logger.warning(f"Error in cross-encoder re-ranking: {e}")
            return candidates[:top_k]

    def retrieve_quotes_hybrid(self, query, k=5, alpha=0.4, beta=0.3, gamma=0.3):
        """
        Advanced hybrid retrieval combining multiple strategies
        
        Args:
            query: User query
            k: Number of quotes to retrieve
            alpha: Weight for semantic search
            beta: Weight for contextual search
            gamma: Weight for keyword search
        """
        self.logger.info(f"Performing advanced hybrid retrieval for query: {query}")
        
        # Get results from all methods
        semantic_results = self.retrieve_quotes_semantic(query, k=k*2)
        contextual_results = self.retrieve_quotes_contextual(query, k=k*2)
        keyword_results = self.retrieve_quotes_keyword_advanced(query, k=k*2)
        
        # Create combined scores dictionary
        combined_scores = {}
        
        # Add semantic scores
        for result in semantic_results:
            idx = result["index"]
            combined_scores[idx] = combined_scores.get(idx, 0) + alpha * result["semantic_score"]
        
        # Add contextual scores
        for result in contextual_results:
            idx = result["index"]
            combined_scores[idx] = combined_scores.get(idx, 0) + beta * result["contextual_score"]
        
        # Add keyword scores
        for result in keyword_results:
            idx = result["index"]
            combined_scores[idx] = combined_scores.get(idx, 0) + gamma * result["keyword_score"]
        
        # Sort by combined score and get top k
        sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:k]
        
        # Build final results
        hybrid_results = []
        for idx in sorted_indices:
            # Get the quote data
            quote_data = self.df.iloc[idx]
            
            # Find the best scores from each method
            semantic_score = next((r["semantic_score"] for r in semantic_results if r["index"] == idx), 0)
            contextual_score = next((r["contextual_score"] for r in contextual_results if r["index"] == idx), 0)
            keyword_score = next((r["keyword_score"] for r in keyword_results if r["index"] == idx), 0)
            
            hybrid_results.append({
                "quote": quote_data["quote"],
                "author": quote_data["author"],
                "tags": quote_data["tags"],
                "hybrid_score": combined_scores[idx],
                "semantic_score": semantic_score,
                "contextual_score": contextual_score,
                "keyword_score": keyword_score,
                "index": idx
            })
        
        return hybrid_results

    def retrieve_quotes(self, query, k=5, method="hybrid"):
        """
        Main retrieval method that can use different strategies
        
        Args:
            query: User query
            k: Number of quotes to retrieve
            method: "semantic", "bm25", "contextual", "keyword_advanced", or "hybrid"
        """
        if method == "semantic":
            results = self.retrieve_quotes_semantic(query, k)
        elif method == "bm25":
            results = self.retrieve_quotes_bm25(query, k)
        elif method == "contextual":
            results = self.retrieve_quotes_contextual(query, k)
        elif method == "keyword_advanced":
            results = self.retrieve_quotes_keyword_advanced(query, k)
        else:  # hybrid
            results = self.retrieve_quotes_hybrid(query, k)
        
        # Apply cross-encoder re-ranking if available
        if self.cross_encoder is not None:
            results = self.re_rank_with_cross_encoder(query, results, k)
        
        return results

    def generate_answer_with_llm(self, query, retrieved_quotes):
        """Generate answer using LLM based on retrieved quotes with enhanced prompting"""
        if self.llm_model is None or self.llm_tokenizer is None:
            return self._format_simple_answer(query, retrieved_quotes)
        
        try:
            # Create enhanced context
            context = self._create_enhanced_context(query, retrieved_quotes)
            
            # Create better prompt
            prompt = self._create_enhanced_prompt(query, context)
            
            # Tokenize and generate with better error handling
            inputs = self.llm_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # Ensure inputs are on CPU
            inputs = inputs.cpu()
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,  # Reduced length for stability
                    num_return_sequences=1,
                    temperature=0.7,  # Reduced temperature for more stable output
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    repetition_penalty=1.1  # Prevent repetition
                )
            
            # Decode the response
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            generated_text = response[len(prompt):].strip()
            
            if generated_text and len(generated_text) > 10:  # Ensure meaningful response
                return f"{generated_text}\n\n" + self._format_simple_answer(query, retrieved_quotes)
            else:
                return self._format_simple_answer(query, retrieved_quotes)
                
        except Exception as e:
            self.logger.warning(f"Error generating LLM response: {e}")
            self.logger.info("Falling back to simple answer formatting")
            return self._format_simple_answer(query, retrieved_quotes)

    def _create_enhanced_context(self, query: str, retrieved_quotes: List[Dict]) -> str:
        """Create enhanced context for LLM generation"""
        context = f"Query: {query}\n\nRelevant Quotes:\n"
        
        for i, quote_info in enumerate(retrieved_quotes[:5], 1):
            context += f"{i}. \"{quote_info['quote']}\" - {quote_info['author']}\n"
            if quote_info['tags']:
                context += f"   Topics: {', '.join(quote_info['tags'])}\n"
            if 'hybrid_score' in quote_info:
                context += f"   Relevance: {quote_info['hybrid_score']:.3f}\n"
            context += "\n"
        
        return context

    def _create_enhanced_prompt(self, query: str, context: str) -> str:
        """Create enhanced prompt for LLM generation"""
        prompt = f"""You are a knowledgeable assistant that provides thoughtful responses based on relevant quotes.

{context}

Based on these quotes, provide a comprehensive and insightful response to the query. 
Consider the themes, authors, and context of the quotes in your response.
Make sure your response is helpful, accurate, and well-structured.

Response:"""
        
        return prompt

    def _format_simple_answer(self, query, retrieved_quotes):
        """Fallback method to format answer without LLM with enhanced information"""
        if not retrieved_quotes:
            return "No relevant quotes found."

        response = f"Here are relevant quotes for your query '{query}':\n\n"
        for i, quote_info in enumerate(retrieved_quotes):
            response += f"{i+1}. \"{quote_info['quote']}\"\n"
            response += f"   - {quote_info['author']}\n"
            response += f"   - Tags: {', '.join(quote_info['tags'])}\n"
            
            # Add detailed score information
            if 'hybrid_score' in quote_info:
                response += f"   - Overall Relevance: {quote_info['hybrid_score']:.3f}\n"
            if 'semantic_score' in quote_info:
                response += f"   - Semantic Similarity: {quote_info['semantic_score']:.3f}\n"
            if 'contextual_score' in quote_info:
                response += f"   - Contextual Relevance: {quote_info['contextual_score']:.3f}\n"
            if 'keyword_score' in quote_info:
                response += f"   - Keyword Match: {quote_info['keyword_score']:.3f}\n"
            if 'cross_encoder_score' in quote_info:
                response += f"   - Cross-Encoder Score: {quote_info['cross_encoder_score']:.3f}\n"
            
            response += "\n"
        
        return response

    def answer_query(self, query, method="hybrid", use_llm=True):
        """
        Main method to answer user queries
        
        Args:
            query: User query
            method: Retrieval method ("semantic", "bm25", "contextual", "keyword_advanced", "hybrid")
            use_llm: Whether to use LLM for answer generation
        """
        # Retrieve relevant quotes
        retrieved_quotes = self.retrieve_quotes(query, method=method)
        
        if use_llm:
            return self.generate_answer_with_llm(query, retrieved_quotes)
        else:
            return self._format_simple_answer(query, retrieved_quotes)

    def get_retrieval_statistics(self) -> Dict:
        """Get statistics about the retrieval system"""
        tfidf_features = 0
        if (hasattr(self, 'tfidf_matrix') and 
            self.tfidf_matrix is not None and 
            hasattr(self.tfidf_matrix, 'shape') and 
            len(self.tfidf_matrix.shape) > 1):
            tfidf_features = self.tfidf_matrix.shape[1]
            
        return {
            "total_quotes": len(self.df),
            "semantic_index_size": self.index.ntotal if hasattr(self, 'index') and self.index is not None else 0,
            "contextual_index_size": self.contextual_index.ntotal if hasattr(self, 'contextual_index') and self.contextual_index is not None else 0,
            "bm25_index_size": self.bm25_index.corpus_size if hasattr(self, 'bm25_index') and self.bm25_index is not None else 0,
            "tfidf_features": tfidf_features,
            "cross_encoder_available": self.cross_encoder is not None,
            "llm_available": self.llm_model is not None
        }

if __name__ == "__main__":
    # Example usage
    print("Initializing Enhanced RAG Pipeline...")
    rag_pipeline = RAGPipelineEnhanced()
    
    # Get statistics
    stats = rag_pipeline.get_retrieval_statistics()
    print("\nRetrieval System Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test queries
    test_queries = [
        "quotes about hope and inspiration",
        "what did Oscar Wilde say about life?",
        "motivational quotes about success",
        "quotes about love and relationships",
        "philosophical quotes about wisdom"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        # Test different methods
        for method in ["semantic", "contextual", "keyword_advanced", "hybrid"]:
            print(f"\n--- {method.upper()} RETRIEVAL ---")
            answer = rag_pipeline.answer_query(query, method=method, use_llm=True)
            print(answer[:500] + "..." if len(answer) > 500 else answer)

