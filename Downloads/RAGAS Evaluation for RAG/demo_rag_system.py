#!/usr/bin/env python3
"""
Demo script for RAG system with open source models.
This script demonstrates the capabilities of the RAG pipeline.
"""

from rag_pipeline_enhanced import RAGPipelineEnhanced
import time

def demo_rag_system():
    """Demonstrate the RAG system capabilities."""
    print("🎯 RAG SYSTEM DEMO WITH OPEN SOURCE MODELS")
    print("="*60)
    
    # Initialize the RAG pipeline
    print("Initializing RAG Pipeline...")
    start_time = time.time()
    rag_pipeline = RAGPipelineEnhanced()
    init_time = time.time() - start_time
    print(f"✓ Pipeline initialized in {init_time:.2f} seconds")
    
    # Show system statistics
    stats = rag_pipeline.get_retrieval_statistics()
    print(f"\n📊 System Statistics:")
    print(f"  - Total quotes: {stats['total_quotes']}")
    print(f"  - Semantic index: {stats['semantic_index_size']} embeddings")
    print(f"  - Contextual index: {stats['contextual_index_size']} embeddings")
    print(f"  - BM25 index: {stats['bm25_index_size']} documents")
    print(f"  - TF-IDF features: {stats['tfidf_features']}")
    print(f"  - Cross-encoder: {'Available' if stats['cross_encoder_available'] else 'Not available'}")
    print(f"  - LLM: {'Available' if stats['llm_available'] else 'Not available'}")
    
    # Demo queries
    demo_queries = [
        "What are some inspiring quotes about hope?",
        "Can you find quotes about love and relationships?",
        "What did famous philosophers say about life?",
        "Show me motivational quotes about success",
        "What are some quotes about courage and bravery?"
    ]
    
    print(f"\n🔍 Testing Different Retrieval Methods")
    print("="*60)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 40)
        
        # Test different retrieval methods
        methods = ["semantic", "bm25", "hybrid"]
        
        for method in methods:
            try:
                start_time = time.time()
                results = rag_pipeline.retrieve_quotes(query, method=method, k=2)
                retrieval_time = time.time() - start_time
                
                print(f"  {method.upper()}: {len(results)} results in {retrieval_time:.3f}s")
                if results:
                    top_result = results[0]
                    score_key = f"{method}_score" if f"{method}_score" in top_result else "hybrid_score"
                    score = top_result.get(score_key, "N/A")
                    print(f"    Top: '{top_result['quote'][:60]}...' (Score: {score:.3f})")
                    
            except Exception as e:
                print(f"  {method.upper()}: Error - {e}")
    
    print(f"\n💬 Testing Answer Generation")
    print("="*60)
    
    test_query = "What are some quotes about wisdom and knowledge?"
    print(f"Query: '{test_query}'")
    
    # Test simple answer generation
    print("\n📝 Simple Answer (without LLM):")
    try:
        start_time = time.time()
        simple_answer = rag_pipeline.answer_query(test_query, method="hybrid", use_llm=False)
        gen_time = time.time() - start_time
        print(f"Generated in {gen_time:.3f} seconds")
        print(simple_answer[:300] + "..." if len(simple_answer) > 300 else simple_answer)
    except Exception as e:
        print(f"Error: {e}")
    
    # Test LLM answer generation
    print("\n🤖 LLM Answer (with open source model):")
    try:
        start_time = time.time()
        llm_answer = rag_pipeline.answer_query(test_query, method="hybrid", use_llm=True)
        gen_time = time.time() - start_time
        print(f"Generated in {gen_time:.3f} seconds")
        print(llm_answer[:300] + "..." if len(llm_answer) > 300 else llm_answer)
    except Exception as e:
        print(f"Error: {e}")
        print("(This is expected if the LLM model is not available)")
    
    print(f"\n🎉 Demo completed successfully!")
    print("The RAG system is working with open source models.")

def interactive_demo():
    """Interactive demo where users can ask questions."""
    print("\n" + "="*60)
    print("🎮 INTERACTIVE DEMO")
    print("="*60)
    print("You can now ask questions about quotes!")
    print("Type 'quit' to exit.")
    
    # Initialize pipeline
    rag_pipeline = RAGPipelineEnhanced()
    
    while True:
        try:
            query = input("\n❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print("\n🔍 Searching...")
            start_time = time.time()
            
            # Get answer
            answer = rag_pipeline.answer_query(query, method="hybrid", use_llm=True)
            
            search_time = time.time() - start_time
            print(f"\n⏱️  Found in {search_time:.3f} seconds")
            print("\n💡 Answer:")
            print(answer)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main demo function."""
    try:
        # Run automated demo
        demo_rag_system()
        
        # Ask if user wants interactive demo
        print("\n" + "="*60)
        response = input("Would you like to try the interactive demo? (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            interactive_demo()
        else:
            print("Thanks for trying the demo!")
            
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Please check that all dependencies are installed and data files are available.")

if __name__ == "__main__":
    main() 