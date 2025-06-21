import pandas as pd
import numpy as np
from datasets import Dataset
from rag_pipeline_enhanced import RAGPipelineEnhanced
import random
import json
import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class CustomRAGEvaluator:
    """
    Custom RAG evaluation framework that doesn't require external API keys.
    Implements custom metrics similar to RAGAS but using local models.
    """
    
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.evaluation_dataset = None
        self.results = None
        # Use a lightweight model for similarity calculations
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def create_evaluation_dataset(self, num_samples=30):
        """
        Create an evaluation dataset for RAG evaluation.
        """
        print(f"Creating evaluation dataset with {num_samples} samples...")
        
        # Diverse evaluation queries
        evaluation_queries = [
            "What are some inspiring quotes about hope and perseverance?",
            "Can you share quotes about love and relationships?",
            "What did famous philosophers say about life and wisdom?",
            "What are some motivational quotes about success and achievement?",
            "Can you find quotes about happiness and joy?",
            "What are some quotes about courage and bravery?",
            "Can you share quotes about friendship and loyalty?",
            "What did great leaders say about leadership and responsibility?",
            "What are some quotes about learning and education?",
            "Can you find quotes about creativity and imagination?",
            "What are some quotes about time and its value?",
            "Can you share quotes about dreams and aspirations?",
            "What are some quotes about change and transformation?",
            "Can you find quotes about peace and tranquility?",
            "What are some quotes about strength and resilience?",
            "What did writers say about books and reading?",
            "Can you share quotes about nature and beauty?",
            "What are some quotes about truth and honesty?",
            "Can you find quotes about freedom and independence?",
            "What are some quotes about family and home?",
            "What did scientists say about knowledge and discovery?",
            "Can you share quotes about art and creativity?",
            "What are some quotes about kindness and compassion?",
            "Can you find quotes about work and dedication?",
            "What are some quotes about faith and belief?",
            "What did poets say about poetry and expression?",
            "Can you share quotes about adventure and exploration?",
            "What are some quotes about forgiveness and mercy?",
            "Can you find quotes about justice and fairness?",
            "What are some quotes about simplicity and minimalism?"
        ]
        
        # Sample queries
        if num_samples < len(evaluation_queries):
            selected_queries = random.sample(evaluation_queries, num_samples)
        else:
            selected_queries = evaluation_queries[:num_samples]
        
        questions = []
        contexts = []
        answers = []
        ground_truths = []
        retrieved_quotes_list = []
        
        for query in selected_queries:
            print(f"Processing query: {query[:50]}...")
            
            # Test different retrieval methods
            methods = ["semantic", "bm25", "hybrid"]
            method = random.choice(methods)
            
            # Get retrieved contexts
            retrieved_quotes = self.rag_pipeline.retrieve_quotes(query, method=method, k=5)
            
            # Format contexts
            context_list = []
            for quote_info in retrieved_quotes:
                context = f'"{quote_info["quote"]}" - {quote_info["author"]}'
                if quote_info["tags"]:
                    context += f" (Tags: {', '.join(quote_info['tags'])})"
                context_list.append(context)
            
            # Generate answer using the RAG pipeline (without LLM to avoid API issues)
            answer = self.rag_pipeline.answer_query(query, method=method, use_llm=False)
            
            # Create a ground truth answer
            ground_truth = self._create_ground_truth(query, retrieved_quotes)
            
            questions.append(query)
            contexts.append(context_list)
            answers.append(answer)
            ground_truths.append(ground_truth)
            retrieved_quotes_list.append(retrieved_quotes)
        
        # Create the evaluation dataset
        eval_data = {
            "question": questions,
            "contexts": contexts,
            "answer": answers,
            "ground_truth": ground_truths,
            "retrieved_quotes": retrieved_quotes_list
        }
        
        self.evaluation_dataset = eval_data
        print(f"Evaluation dataset created with {len(questions)} samples.")
        
        return self.evaluation_dataset
    
    def _create_ground_truth(self, query, retrieved_quotes):
        """
        Create a ground truth answer based on the retrieved quotes.
        """
        if not retrieved_quotes:
            return "No relevant quotes found for this query."
        
        # Create a concise ground truth based on the top quotes
        top_quotes = retrieved_quotes[:3]
        
        ground_truth = f"Relevant quotes for '{query}':\n"
        for i, quote_info in enumerate(top_quotes, 1):
            ground_truth += f"{i}. \"{quote_info['quote']}\" - {quote_info['author']}\n"
        
        return ground_truth.strip()
    
    def calculate_context_relevance(self, question, contexts):
        """
        Calculate how relevant the retrieved contexts are to the question.
        Uses cosine similarity between question and context embeddings.
        """
        if not contexts:
            return 0.0
        
        # Get embeddings
        question_embedding = self.similarity_model.encode([question])
        context_embeddings = self.similarity_model.encode(contexts)
        
        # Calculate similarities
        similarities = cosine_similarity(question_embedding, context_embeddings)[0]
        
        # Return average similarity
        return float(np.mean(similarities))
    
    def calculate_answer_relevance(self, question, answer):
        """
        Calculate how relevant the answer is to the question.
        """
        if not answer or not question:
            return 0.0
        
        # Get embeddings
        question_embedding = self.similarity_model.encode([question])
        answer_embedding = self.similarity_model.encode([answer])
        
        # Calculate similarity
        similarity = cosine_similarity(question_embedding, answer_embedding)[0][0]
        
        return float(similarity)
    
    def calculate_context_precision(self, question, contexts, retrieved_quotes):
        """
        Calculate precision of retrieved contexts.
        Measures how many of the retrieved contexts are relevant.
        """
        if not contexts or not retrieved_quotes:
            return 0.0
        
        # Simple heuristic: contexts with higher similarity scores are more precise
        total_score = 0.0
        for quote_info in retrieved_quotes:
            # Use the retrieval score as a proxy for precision
            if 'hybrid_score' in quote_info:
                total_score += quote_info['hybrid_score']
            elif 'semantic_score' in quote_info:
                total_score += quote_info['semantic_score']
            elif 'bm25_score' in quote_info:
                # Normalize BM25 score
                total_score += min(quote_info['bm25_score'] / 10.0, 1.0)
        
        return total_score / len(retrieved_quotes) if retrieved_quotes else 0.0
    
    def calculate_context_recall(self, ground_truth, contexts):
        """
        Calculate recall of retrieved contexts.
        Measures how much of the ground truth information is covered by contexts.
        """
        if not ground_truth or not contexts:
            return 0.0
        
        # Calculate similarity between ground truth and contexts
        ground_truth_embedding = self.similarity_model.encode([ground_truth])
        context_embeddings = self.similarity_model.encode(contexts)
        
        # Find the maximum similarity (best coverage)
        similarities = cosine_similarity(ground_truth_embedding, context_embeddings)[0]
        
        return float(np.max(similarities)) if len(similarities) > 0 else 0.0
    
    def calculate_faithfulness(self, answer, contexts):
        """
        Calculate faithfulness of the answer to the retrieved contexts.
        Measures how much the answer is supported by the contexts.
        """
        if not answer or not contexts:
            return 0.0
        
        # Calculate similarity between answer and contexts
        answer_embedding = self.similarity_model.encode([answer])
        context_embeddings = self.similarity_model.encode(contexts)
        
        # Calculate similarities and take the average
        similarities = cosine_similarity(answer_embedding, context_embeddings)[0]
        
        return float(np.mean(similarities))
    
    def run_evaluation(self, dataset=None):
        """
        Run custom RAG evaluation on the dataset.
        """
        if dataset is None:
            dataset = self.evaluation_dataset
        
        if dataset is None:
            raise ValueError("No evaluation dataset available. Please create one first.")
        
        print("Starting Custom RAG Evaluation...")
        print(f"Dataset size: {len(dataset['question'])}")
        
        results = {
            'context_relevance': [],
            'answer_relevance': [],
            'context_precision': [],
            'context_recall': [],
            'faithfulness': []
        }
        
        for i in range(len(dataset['question'])):
            question = dataset['question'][i]
            contexts = dataset['contexts'][i]
            answer = dataset['answer'][i]
            ground_truth = dataset['ground_truth'][i]
            retrieved_quotes = dataset['retrieved_quotes'][i]
            
            print(f"Evaluating sample {i+1}/{len(dataset['question'])}: {question[:50]}...")
            
            # Calculate metrics
            context_rel = self.calculate_context_relevance(question, contexts)
            answer_rel = self.calculate_answer_relevance(question, answer)
            context_prec = self.calculate_context_precision(question, contexts, retrieved_quotes)
            context_rec = self.calculate_context_recall(ground_truth, contexts)
            faithful = self.calculate_faithfulness(answer, contexts)
            
            results['context_relevance'].append(context_rel)
            results['answer_relevance'].append(answer_rel)
            results['context_precision'].append(context_prec)
            results['context_recall'].append(context_rec)
            results['faithfulness'].append(faithful)
        
        # Calculate average scores
        avg_results = {}
        for metric, scores in results.items():
            avg_results[metric] = np.mean(scores)
            avg_results[f"{metric}_std"] = np.std(scores)
        
        self.results = {
            'individual_scores': results,
            'average_scores': avg_results,
            'dataset_info': {
                'num_samples': len(dataset['question']),
                'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        print("Custom RAG evaluation completed successfully!")
        return self.results
    
    def analyze_results(self):
        """
        Analyze and display the evaluation results.
        """
        if self.results is None:
            print("No evaluation results available.")
            return
        
        print("\n" + "="*60)
        print("CUSTOM RAG EVALUATION RESULTS")
        print("="*60)
        
        avg_scores = self.results['average_scores']
        
        print(f"Dataset Information:")
        print(f"- Number of samples: {self.results['dataset_info']['num_samples']}")
        print(f"- Evaluation date: {self.results['dataset_info']['evaluation_date']}")
        
        print(f"\nAverage Metric Scores:")
        metrics_info = {
            'context_relevance': 'How relevant retrieved contexts are to the question',
            'answer_relevance': 'How relevant the answer is to the question',
            'context_precision': 'Precision of retrieved contexts',
            'context_recall': 'Recall of retrieved contexts',
            'faithfulness': 'How faithful the answer is to the contexts'
        }
        
        for metric in ['context_relevance', 'answer_relevance', 'context_precision', 'context_recall', 'faithfulness']:
            score = avg_scores[metric]
            std = avg_scores[f"{metric}_std"]
            print(f"- {metric.replace('_', ' ').title()}: {score:.4f} (±{std:.4f})")
            print(f"  {metrics_info[metric]}")
        
        # Overall score
        overall_score = np.mean([avg_scores[m] for m in ['context_relevance', 'answer_relevance', 'context_precision', 'context_recall', 'faithfulness']])
        print(f"\nOverall RAG Performance Score: {overall_score:.4f}")
        
        # Performance interpretation
        if overall_score >= 0.8:
            performance = "Excellent"
        elif overall_score >= 0.7:
            performance = "Good"
        elif overall_score >= 0.6:
            performance = "Fair"
        else:
            performance = "Needs Improvement"
        
        print(f"Performance Level: {performance}")
        
        return self.results
    
    def save_results(self, filename="custom_rag_evaluation_results.json"):
        """
        Save evaluation results to a file.
        """
        if self.results is None:
            print("No results to save.")
            return
        
        # Convert numpy arrays to lists for JSON serialization
        results_to_save = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                results_to_save[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        results_to_save[key][k] = v.tolist()
                    elif isinstance(v, (np.float64, np.float32, np.int64, np.int32)):
                        results_to_save[key][k] = float(v)
                    else:
                        results_to_save[key][k] = v
            elif isinstance(value, list):
                results_to_save[key] = [float(x) if isinstance(x, (np.float64, np.float32, np.int64, np.int32)) else x for x in value]
            else:
                results_to_save[key] = value
        
        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Results saved to {filename}")

def main():
    """
    Main function to run custom RAG evaluation.
    """
    print("Initializing Custom RAG Evaluation...")
    
    # Initialize the RAG pipeline
    rag_pipeline = RAGPipelineEnhanced()
    
    # Initialize the evaluator
    evaluator = CustomRAGEvaluator(rag_pipeline)
    
    # Create evaluation dataset
    dataset = evaluator.create_evaluation_dataset(num_samples=25)
    
    # Run evaluation
    results = evaluator.run_evaluation(dataset)
    
    # Analyze results
    evaluator.analyze_results()
    
    # Save results
    evaluator.save_results()
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\nCustom RAG evaluation completed successfully!")
    else:
        print("\nCustom RAG evaluation encountered issues.")

