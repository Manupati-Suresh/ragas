import pandas as pd
import numpy as np
from custom_rag_evaluator import CustomRAGEvaluator
from rag_pipeline_enhanced import RAGPipelineEnhanced
import matplotlib.pyplot as plt
import seaborn as sns

def run_comprehensive_evaluation():
    """
    Run a comprehensive evaluation of the RAG pipeline with different configurations.
    """
    print("="*80)
    print("COMPREHENSIVE RAG PIPELINE EVALUATION")
    print("="*80)
    
    # Initialize the RAG pipeline
    print("Initializing RAG Pipeline...")
    rag_pipeline = RAGPipelineEnhanced()
    
    # Initialize the evaluator
    evaluator = CustomRAGEvaluator(rag_pipeline)
    
    # Create evaluation dataset
    print("\nCreating evaluation dataset...")
    dataset = evaluator.create_evaluation_dataset(num_samples=30)
    
    # Run evaluation
    print("\nRunning comprehensive evaluation...")
    results = evaluator.run_evaluation(dataset)
    
    # Analyze results
    print("\nAnalyzing results...")
    evaluator.analyze_results()
    
    # Save results manually to avoid JSON serialization issues
    save_results_manually(evaluator.results)
    
    # Create visualizations
    create_evaluation_visualizations(evaluator.results)
    
    # Compare different retrieval methods
    compare_retrieval_methods(rag_pipeline)
    
    return evaluator.results

def save_results_manually(results):
    """
    Manually save results to avoid JSON serialization issues.
    """
    print("\nSaving evaluation results...")
    
    # Save as CSV for easy analysis
    if 'individual_scores' in results:
        df = pd.DataFrame(results['individual_scores'])
        df.to_csv('rag_evaluation_individual_scores.csv', index=False)
        print("Individual scores saved to rag_evaluation_individual_scores.csv")
    
    # Save summary as text
    with open('rag_evaluation_summary.txt', 'w') as f:
        f.write("RAG Pipeline Evaluation Summary\n")
        f.write("="*50 + "\n\n")
        
        if 'dataset_info' in results:
            f.write(f"Dataset Information:\n")
            f.write(f"- Number of samples: {results['dataset_info']['num_samples']}\n")
            f.write(f"- Evaluation date: {results['dataset_info']['evaluation_date']}\n\n")
        
        if 'average_scores' in results:
            f.write("Average Metric Scores:\n")
            avg_scores = results['average_scores']
            
            metrics_info = {
                'context_relevance': 'How relevant retrieved contexts are to the question',
                'answer_relevance': 'How relevant the answer is to the question',
                'context_precision': 'Precision of retrieved contexts',
                'context_recall': 'Recall of retrieved contexts',
                'faithfulness': 'How faithful the answer is to the contexts'
            }
            
            for metric in ['context_relevance', 'answer_relevance', 'context_precision', 'context_recall', 'faithfulness']:
                if metric in avg_scores:
                    score = avg_scores[metric]
                    std = avg_scores.get(f"{metric}_std", 0)
                    f.write(f"- {metric.replace('_', ' ').title()}: {score:.4f} (±{std:.4f})\n")
                    f.write(f"  {metrics_info[metric]}\n")
            
            # Overall score
            overall_score = np.mean([avg_scores[m] for m in ['context_relevance', 'answer_relevance', 'context_precision', 'context_recall', 'faithfulness'] if m in avg_scores])
            f.write(f"\nOverall RAG Performance Score: {overall_score:.4f}\n")
            
            # Performance interpretation
            if overall_score >= 0.8:
                performance = "Excellent"
            elif overall_score >= 0.7:
                performance = "Good"
            elif overall_score >= 0.6:
                performance = "Fair"
            else:
                performance = "Needs Improvement"
            
            f.write(f"Performance Level: {performance}\n")
    
    print("Summary saved to rag_evaluation_summary.txt")

def create_evaluation_visualizations(results):
    """
    Create visualizations for the evaluation results.
    """
    print("\nCreating evaluation visualizations...")
    
    if 'individual_scores' not in results:
        print("No individual scores available for visualization.")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RAG Pipeline Evaluation Results', fontsize=16, fontweight='bold')
    
    individual_scores = results['individual_scores']
    
    # Plot 1: Box plot of all metrics
    metrics_data = []
    metric_names = []
    for metric, scores in individual_scores.items():
        metrics_data.extend(scores)
        metric_names.extend([metric.replace('_', ' ').title()] * len(scores))
    
    df_plot = pd.DataFrame({'Metric': metric_names, 'Score': metrics_data})
    sns.boxplot(data=df_plot, x='Metric', y='Score', ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Metric Scores')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Bar plot of average scores
    avg_scores = {k.replace('_', ' ').title(): np.mean(v) for k, v in individual_scores.items()}
    axes[0, 1].bar(avg_scores.keys(), avg_scores.values())
    axes[0, 1].set_title('Average Metric Scores')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim(0, 1)
    
    # Plot 3: Correlation heatmap
    df_corr = pd.DataFrame(individual_scores)
    correlation_matrix = df_corr.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 2])
    axes[0, 2].set_title('Metric Correlations')
    
    # Plot 4: Line plot showing score trends
    sample_indices = range(len(individual_scores['context_relevance']))
    for metric, scores in individual_scores.items():
        axes[1, 0].plot(sample_indices, scores, label=metric.replace('_', ' ').title(), marker='o', markersize=3)
    axes[1, 0].set_title('Score Trends Across Samples')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Histogram of overall performance
    overall_scores = []
    for i in range(len(individual_scores['context_relevance'])):
        sample_scores = [individual_scores[metric][i] for metric in individual_scores.keys()]
        overall_scores.append(np.mean(sample_scores))
    
    axes[1, 1].hist(overall_scores, bins=10, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Distribution of Overall Sample Scores')
    axes[1, 1].set_xlabel('Overall Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.mean(overall_scores), color='red', linestyle='--', label=f'Mean: {np.mean(overall_scores):.3f}')
    axes[1, 1].legend()
    
    # Plot 6: Radar chart of average metrics
    metrics = list(individual_scores.keys())
    values = [np.mean(individual_scores[metric]) for metric in metrics]
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]
    
    axes[1, 2].plot(angles, values, 'o-', linewidth=2)
    axes[1, 2].fill(angles, values, alpha=0.25)
    axes[1, 2].set_xticks(angles[:-1])
    axes[1, 2].set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title('Radar Chart of Average Metrics')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('rag_evaluation_visualizations.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to rag_evaluation_visualizations.png")
    plt.close()

def compare_retrieval_methods(rag_pipeline):
    """
    Compare different retrieval methods (semantic, BM25, hybrid).
    """
    print("\nComparing retrieval methods...")
    
    test_queries = [
        "quotes about love and happiness",
        "what did Oscar Wilde say about life?",
        "motivational quotes about success",
        "quotes about wisdom and knowledge",
        "inspiring quotes about courage"
    ]
    
    methods = ["semantic", "bm25", "hybrid"]
    comparison_results = {method: [] for method in methods}
    
    for query in test_queries:
        print(f"Testing query: {query}")
        
        for method in methods:
            # Get retrieval results
            retrieved_quotes = rag_pipeline.retrieve_quotes(query, method=method, k=5)
            
            # Calculate a simple relevance score based on the retrieval scores
            if retrieved_quotes:
                if method == "semantic":
                    avg_score = np.mean([q.get('semantic_score', 0) for q in retrieved_quotes])
                elif method == "bm25":
                    # Normalize BM25 scores
                    bm25_scores = [min(q.get('bm25_score', 0) / 10.0, 1.0) for q in retrieved_quotes]
                    avg_score = np.mean(bm25_scores)
                else:  # hybrid
                    avg_score = np.mean([q.get('hybrid_score', 0) for q in retrieved_quotes])
            else:
                avg_score = 0.0
            
            comparison_results[method].append(avg_score)
    
    # Create comparison visualization
    plt.figure(figsize=(12, 8))
    
    # Bar plot comparison
    plt.subplot(2, 2, 1)
    method_averages = {method: np.mean(scores) for method, scores in comparison_results.items()}
    plt.bar(method_averages.keys(), method_averages.values())
    plt.title('Average Retrieval Performance by Method')
    plt.ylabel('Average Score')
    
    # Line plot for each query
    plt.subplot(2, 2, 2)
    for method, scores in comparison_results.items():
        plt.plot(range(len(scores)), scores, marker='o', label=method.title())
    plt.title('Retrieval Performance by Query')
    plt.xlabel('Query Index')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot comparison
    plt.subplot(2, 2, 3)
    data_for_box = [scores for scores in comparison_results.values()]
    plt.boxplot(data_for_box, labels=list(comparison_results.keys()))
    plt.title('Score Distribution by Method')
    plt.ylabel('Score')
    
    # Heatmap of method performance per query
    plt.subplot(2, 2, 4)
    heatmap_data = np.array([comparison_results[method] for method in methods])
    sns.heatmap(heatmap_data, 
                xticklabels=[f"Q{i+1}" for i in range(len(test_queries))],
                yticklabels=[method.title() for method in methods],
                annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Method Performance Heatmap')
    
    plt.tight_layout()
    plt.savefig('retrieval_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("Method comparison saved to retrieval_methods_comparison.png")
    plt.close()
    
    # Save comparison results
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.index = [f"Query_{i+1}" for i in range(len(test_queries))]
    comparison_df.to_csv('retrieval_methods_comparison.csv')
    print("Comparison data saved to retrieval_methods_comparison.csv")

if __name__ == "__main__":
    results = run_comprehensive_evaluation()
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("- rag_evaluation_individual_scores.csv")
    print("- rag_evaluation_summary.txt")
    print("- rag_evaluation_visualizations.png")
    print("- retrieval_methods_comparison.png")
    print("- retrieval_methods_comparison.csv")

