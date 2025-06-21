import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline_enhanced import RAGPipelineEnhanced

class TestRAGPipelineEnhanced(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        print("Setting up RAG Pipeline Enhanced for testing...")
        cls.rag_pipeline = RAGPipelineEnhanced()
    
    def test_initialization(self):
        """Test that the RAG pipeline initializes correctly."""
        self.assertIsNotNone(self.rag_pipeline.model)
        self.assertIsNotNone(self.rag_pipeline.df)
        self.assertIsNotNone(self.rag_pipeline.index)
        self.assertIsNotNone(self.rag_pipeline.bm25_index)
        print("✓ Initialization test passed")
    
    def test_semantic_retrieval(self):
        """Test semantic retrieval functionality."""
        query = "quotes about love"
        results = self.rag_pipeline.retrieve_quotes_semantic(query, k=3)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)
        
        if results:
            # Check that each result has required fields
            for result in results:
                self.assertIn('quote', result)
                self.assertIn('author', result)
                self.assertIn('tags', result)
                self.assertIn('semantic_score', result)
                self.assertIn('index', result)
        
        print(f"✓ Semantic retrieval test passed - Retrieved {len(results)} quotes")
    
    def test_bm25_retrieval(self):
        """Test BM25 retrieval functionality."""
        query = "success motivation"
        results = self.rag_pipeline.retrieve_quotes_bm25(query, k=3)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)
        
        if results:
            # Check that each result has required fields
            for result in results:
                self.assertIn('quote', result)
                self.assertIn('author', result)
                self.assertIn('tags', result)
                self.assertIn('bm25_score', result)
                self.assertIn('index', result)
        
        print(f"✓ BM25 retrieval test passed - Retrieved {len(results)} quotes")
    
    def test_hybrid_retrieval(self):
        """Test hybrid retrieval functionality."""
        query = "wisdom philosophy"
        results = self.rag_pipeline.retrieve_quotes_hybrid(query, k=5)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 5)
        
        if results:
            # Check that each result has required fields
            for result in results:
                self.assertIn('quote', result)
                self.assertIn('author', result)
                self.assertIn('tags', result)
                self.assertIn('hybrid_score', result)
                self.assertIn('index', result)
        
        print(f"✓ Hybrid retrieval test passed - Retrieved {len(results)} quotes")
    
    def test_answer_generation_without_llm(self):
        """Test answer generation without LLM."""
        query = "quotes about happiness"
        answer = self.rag_pipeline.answer_query(query, use_llm=False)
        
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
        self.assertIn(query, answer)
        
        print("✓ Answer generation (without LLM) test passed")
    
    def test_answer_generation_with_llm(self):
        """Test answer generation with LLM."""
        query = "quotes about courage"
        answer = self.rag_pipeline.answer_query(query, use_llm=True)
        
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
        
        print("✓ Answer generation (with LLM) test passed")
    
    def test_different_retrieval_methods(self):
        """Test that different retrieval methods work."""
        query = "life wisdom"
        
        # Test semantic method
        semantic_answer = self.rag_pipeline.answer_query(query, method="semantic", use_llm=False)
        self.assertIsInstance(semantic_answer, str)
        
        # Test BM25 method
        bm25_answer = self.rag_pipeline.answer_query(query, method="bm25", use_llm=False)
        self.assertIsInstance(bm25_answer, str)
        
        # Test hybrid method
        hybrid_answer = self.rag_pipeline.answer_query(query, method="hybrid", use_llm=False)
        self.assertIsInstance(hybrid_answer, str)
        
        print("✓ Different retrieval methods test passed")
    
    def test_empty_query(self):
        """Test behavior with empty query."""
        query = ""
        results = self.rag_pipeline.retrieve_quotes(query, k=3)
        
        # Should handle empty query gracefully
        self.assertIsInstance(results, list)
        
        print("✓ Empty query test passed")
    
    def test_query_with_special_characters(self):
        """Test behavior with special characters in query."""
        query = "what's life? @#$%"
        results = self.rag_pipeline.retrieve_quotes(query, k=3)
        
        # Should handle special characters gracefully
        self.assertIsInstance(results, list)
        
        print("✓ Special characters query test passed")

def run_integration_tests():
    """Run comprehensive integration tests."""
    print("="*60)
    print("RUNNING RAG PIPELINE ENHANCED INTEGRATION TESTS")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRAGPipelineEnhanced)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")
    
    return success

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)

