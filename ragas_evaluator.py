import pandas as pd
import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    ContextRelevance
)
from rag_pipeline_enhanced import RAGPipelineEnhanced
import random
import json

class RAGASEvaluator:
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.evaluation_dataset = None
        self.results = None
    
    def create_evaluation_dataset(self, num_samples=50):
        """
        Create an evaluation dataset for RAGAS evaluation.
        
        This function generates questions, retrieves contexts, generates answers,
        and creates ground truth answers for evaluation.
        """
        print(f"Creating evaluation dataset with {num_samples} samples...")
        
        # Sample diverse queries for evaluation
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
            "What are some quotes about simplicity and minimalism?",
            "What did musicians say about music and harmony?",
            "Can you share quotes about patience and perseverance?",
            "What are some quotes about gratitude and appreciation?",
            "Can you find quotes about solitude and reflection?",
            "What are some quotes about communication and understanding?",
            "What did athletes say about sports and competition?",
            "Can you share quotes about innovation and progress?",
            "What are some quotes about memory and nostalgia?",
            "Can you find quotes about humor and laughter?",
            "What are some quotes about purpose and meaning?",
            "What did entrepreneurs say about business and success?",
            "Can you share quotes about travel and adventure?",
            "What are some quotes about health and wellness?",
            "Can you find quotes about technology and the future?",
            "What are some quotes about community and society?",
            "What did teachers say about education and learning?",
            "Can you share quotes about balance and harmony?",
            "What are some quotes about risk and opportunity?",
            "Can you find quotes about legacy and impact?",
            "What are some quotes about growth and development?"
        ]
        
        # Randomly sample queries if we need fewer than available
        if num_samples < len(evaluation_queries):
            selected_queries = random.sample(evaluation_queries, num_samples)
        else:
            selected_queries = evaluation_queries[:num_samples]
        
        questions = []
        contexts = []
        answers = []
        ground_truths = []
        
        for query in selected_queries:
            print(f"Processing query: {query[:50]}...")
            
            # Get retrieved contexts using hybrid search
            retrieved_quotes = self.rag_pipeline.retrieve_quotes(query, method="hybrid", k=5)
            
            # Format contexts
            context_list = []
            for quote_info in retrieved_quotes:
                context = f'"{quote_info["quote"]}" - {quote_info["author"]}'
                if quote_info["tags"]:
                    context += f" (Tags: {', '.join(quote_info['tags'])})"
                context_list.append(context)
            
            # Generate answer using the RAG pipeline
            answer = self.rag_pipeline.answer_query(query, method="hybrid", use_llm=True)
            
            # Create a ground truth answer (simplified version focusing on the quotes)
            ground_truth = self._create_ground_truth(query, retrieved_quotes)
            
            questions.append(query)
            contexts.append(context_list)
            answers.append(answer)
            ground_truths.append(ground_truth)
        
        # Create the evaluation dataset
        eval_data = {
            "question": questions,
            "contexts": contexts,
            "answer": answers,
            "ground_truth": ground_truths
        }
        
        self.evaluation_dataset = Dataset.from_dict(eval_data)
        print(f"Evaluation dataset created with {len(questions)} samples.")
        
        return self.evaluation_dataset
    
    def _create_ground_truth(self, query, retrieved_quotes):
        """
        Create a ground truth answer based on the retrieved quotes.
        This is a simplified approach for demonstration purposes.
        """
        if not retrieved_quotes:
            return "No relevant quotes found for this query."
        
        # Create a concise ground truth based on the top quotes
        top_quotes = retrieved_quotes[:3]  # Use top 3 quotes
        
        ground_truth = f"Based on the query '{query}', here are relevant quotes:\n"
        for i, quote_info in enumerate(top_quotes, 1):
            ground_truth += f"{i}. \"{quote_info['quote']}\" - {quote_info['author']}\n"
        
        return ground_truth.strip()
    
    def configure_ragas_metrics(self):
        """
        Configure RAGAS metrics for evaluation.
        """
        metrics = [
            faithfulness,           # Measures factual consistency of the answer
            answer_relevancy,       # Measures how relevant the answer is to the question
            context_precision,      # Measures precision of retrieved context
            context_recall,         # Measures recall of retrieved context
            ContextRelevance()      # Measures relevance of retrieved context
        ]
        
        print("RAGAS metrics configured:")
        for metric in metrics:
            print(f"- {metric.name}")
        
        return metrics
    
    def run_evaluation(self, dataset=None, metrics=None):
        """
        Run RAGAS evaluation on the dataset.
        """
        if dataset is None:
            dataset = self.evaluation_dataset
        
        if dataset is None:
            raise ValueError("No evaluation dataset available. Please create one first.")
        
        if metrics is None:
            metrics = self.configure_ragas_metrics()
        
        print("Starting RAGAS evaluation...")
        print(f"Dataset size: {len(dataset)}")
        print(f"Metrics: {[m.name for m in metrics]}")
        
        try:
            # Run the evaluation
            self.results = evaluate(
                dataset=dataset,
                metrics=metrics,
            )
            
            print("RAGAS evaluation completed successfully!")
            return self.results
            
        except Exception as e:
            print(f"Error during RAGAS evaluation: {e}")
            print("This might be due to API limitations or configuration issues.")
            return None
    
    def analyze_results(self):
        """
        Analyze and display the evaluation results.
        """
        if self.results is None:
            print("No evaluation results available.")
            return
        
        print("\n" + "="*60)
        print("RAGAS EVALUATION RESULTS")
        print("="*60)
        
        # Display overall scores
        for metric_name, score in self.results.items():
            if isinstance(score, (int, float)):
                print(f"{metric_name}: {score:.4f}")
        
        # Convert to DataFrame for detailed analysis
        df_results = self.results.to_pandas()
        
        print(f"\nDetailed Results Summary:")
        print(f"Total samples evaluated: {len(df_results)}")
        
        # Display statistics for each metric
        numeric_columns = df_results.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            print(f"\nMetric Statistics:")
            print(df_results[numeric_columns].describe())
        
        return df_results
    
    def save_results(self, filename="ragas_evaluation_results.json"):
        """
        Save evaluation results to a file.
        """
        if self.results is None:
            print("No results to save.")
            return
        
        # Convert results to a serializable format
        results_dict = {}
        for key, value in self.results.items():
            if isinstance(value, (int, float, str, bool)):
                results_dict[key] = value
            else:
                results_dict[key] = str(value)
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def create_sample_dataset_for_testing(self, num_samples=5):
        """
        Create a small sample dataset for testing purposes.
        """
        print(f"Creating sample dataset with {num_samples} samples for testing...")
        
        # Simple test queries
        test_queries = [
            "What are quotes about love?",
            "Can you find quotes about success?",
            "What did Oscar Wilde say?",
            "What are motivational quotes?",
            "Can you share quotes about life?"
        ]
        
        questions = []
        contexts = []
        answers = []
        ground_truths = []
        
        for i, query in enumerate(test_queries[:num_samples]):
            # Get a simple answer without LLM to avoid potential issues
            retrieved_quotes = self.rag_pipeline.retrieve_quotes(query, method="semantic", k=3)
            
            # Format contexts
            context_list = []
            for quote_info in retrieved_quotes:
                context = f'"{quote_info["quote"]}" - {quote_info["author"]}'
                context_list.append(context)
            
            # Simple answer
            answer = f"Here are some relevant quotes for '{query}':\n"
            for j, quote_info in enumerate(retrieved_quotes[:2], 1):
                answer += f"{j}. \"{quote_info['quote']}\" - {quote_info['author']}\n"
            
            # Ground truth
            ground_truth = f"Relevant quotes about the topic in the query."
            
            questions.append(query)
            contexts.append(context_list)
            answers.append(answer.strip())
            ground_truths.append(ground_truth)
        
        # Create dataset
        eval_data = {
            "question": questions,
            "contexts": contexts,
            "answer": answers,
            "ground_truth": ground_truths
        }
        
        return Dataset.from_dict(eval_data)

def main():
    """
    Main function to run RAGAS evaluation.
    """
    print("Initializing RAGAS Evaluation...")
    
    # Initialize the RAG pipeline
    rag_pipeline = RAGPipelineEnhanced()
    
    # Initialize the evaluator
    evaluator = RAGASEvaluator(rag_pipeline)
    
    # Create evaluation dataset
    try:
        # Try to create a full dataset
        dataset = evaluator.create_evaluation_dataset(num_samples=20)
    except Exception as e:
        print(f"Error creating full dataset: {e}")
        print("Creating a smaller sample dataset for testing...")
        dataset = evaluator.create_sample_dataset_for_testing(num_samples=5)
    
    # Configure metrics
    metrics = evaluator.configure_ragas_metrics()
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation(dataset, metrics)
        
        if results is not None:
            # Analyze results
            df_results = evaluator.analyze_results()
            
            # Save results
            evaluator.save_results()
            
            return True
        else:
            print("Evaluation failed.")
            return False
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("This might be due to API limitations or missing API keys.")
        print("RAGAS evaluation requires access to LLM APIs for some metrics.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nRAGAS evaluation completed successfully!")
    else:
        print("\nRAGAS evaluation encountered issues. Please check the configuration.")

