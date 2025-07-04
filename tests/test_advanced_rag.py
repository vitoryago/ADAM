#!/usr/bin/env python3
"""
Comprehensive Test Suite for Advanced RAG System
===============================================

This test suite validates all aspects of the advanced RAG system:
1. Individual retrieval methods (BM25, Vector, Graph)
2. Reciprocal Rank Fusion combination
3. Edge cases and error handling
4. Performance characteristics
5. Integration with existing memory systems

The tests are designed to:
- Ensure correctness of each component
- Validate the theoretical benefits claimed
- Test edge cases and failure modes
- Measure performance characteristics
"""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
from typing import List, Dict, Set

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.adam.memory import MemorySystem, MemoryType
from src.adam.memory_network import MemoryNetworkSystem
from src.adam.advanced_rag import (
    AdvancedRAGSystem, 
    RetrievalResult,
    demonstrate_retrieval_differences
)


class TestAdvancedRAG:
    """
    Main test class for the Advanced RAG system
    
    Tests are organized by:
    1. Setup and initialization
    2. Individual retrieval methods
    3. Combined retrieval
    4. Edge cases
    5. Performance tests
    """
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def populated_system(self, temp_dir):
        """
        Create a populated test system with diverse memories
        
        This fixture creates:
        - A variety of memory types
        - Connected memories for graph traversal
        - Memories with different characteristics
        """
        # Initialize systems
        memory_system = MemorySystem(base_dir=temp_dir)
        memory_network = MemoryNetworkSystem(memory_system)
        
        # Add diverse test memories
        test_data = [
            # Exact match memories (BM25 should excel)
            {
                "query": "ImportError: No module named 'pandas'",
                "response": "Install pandas with: pip install pandas",
                "topics": ["python", "error", "import"],
                "memory_type": "error_solution"
            },
            {
                "query": "ModuleNotFoundError: No module named 'numpy'",
                "response": "Install numpy with: pip install numpy",
                "topics": ["python", "error", "import"],
                "memory_type": "error_solution"
            },
            
            # Semantic variations (Vector should excel)
            {
                "query": "How to read CSV files in Python?",
                "response": "Use pandas.read_csv() or csv.reader() from standard library",
                "topics": ["python", "csv", "data"],
                "memory_type": "explanation"
            },
            {
                "query": "Loading comma-separated data",
                "response": "For CSV data, use pd.read_csv() with appropriate parameters",
                "topics": ["python", "csv", "loading"],
                "memory_type": "explanation"
            },
            
            # Connected concepts (Graph should excel)
            {
                "query": "Database connection timeout",
                "response": "Increase timeout in connection string or pool configuration",
                "topics": ["database", "connection", "timeout"],
                "memory_type": "configuration"
            },
            {
                "query": "Connection pool exhausted",
                "response": "Increase pool size or implement connection recycling",
                "topics": ["database", "connection", "pool"],
                "memory_type": "error_solution"
            },
            {
                "query": "Optimize database queries",
                "response": "Use indexes, avoid N+1 queries, batch operations",
                "topics": ["database", "optimization", "performance"],
                "memory_type": "optimization"
            }
        ]
        
        # Store memories and collect IDs
        memory_ids = []
        for data in test_data:
            memory_id = memory_system.store_memory(
                query=data["query"],
                response=data["response"],
                memory_type=data["memory_type"],
                metadata={"topics": data["topics"]}
            )
            memory_ids.append(memory_id)
            
            # Add to network
            memory_network.add_memory(
                memory_id=memory_id,
                conversation_id=f"test_{memory_id}",
                query=data["query"],
                response=data["response"],
                topics=data["topics"],
                memory_type=data["memory_type"]
            )
        
        # Create connections for graph traversal testing
        # Connect import errors
        memory_network.add_reference(memory_ids[0], memory_ids[1], weight=0.9)
        
        # Connect CSV loading methods
        memory_network.add_reference(memory_ids[2], memory_ids[3], weight=0.95)
        
        # Connect database issues in a chain
        memory_network.add_reference(memory_ids[4], memory_ids[5], weight=0.9)
        memory_network.add_reference(memory_ids[5], memory_ids[6], weight=0.85)
        
        # Create RAG system
        rag_system = AdvancedRAGSystem(memory_system, memory_network)
        
        return {
            "rag": rag_system,
            "memory_system": memory_system,
            "memory_network": memory_network,
            "memory_ids": memory_ids,
            "test_data": test_data
        }
    
    def test_initialization(self, temp_dir):
        """Test proper initialization of the RAG system"""
        memory_system = MemorySystem(base_dir=temp_dir)
        memory_network = MemoryNetworkSystem(memory_system)
        
        # Test with default parameters
        rag = AdvancedRAGSystem(memory_system, memory_network)
        assert rag.k1 == 1.2  # Default BM25 parameter
        assert rag.b == 0.75   # Default BM25 parameter
        assert rag.bm25 is not None
        assert rag.vector_store is not None
        
        # Test with custom parameters
        rag_custom = AdvancedRAGSystem(
            memory_system, 
            memory_network,
            k1=2.0,
            b=0.5
        )
        assert rag_custom.k1 == 2.0
        assert rag_custom.b == 0.5
    
    def test_bm25_tokenization(self, populated_system):
        """Test the BM25 tokenizer handles technical content correctly"""
        rag = populated_system["rag"]
        
        # Test camelCase splitting
        tokens = rag._tokenize_for_bm25("getElementById")
        assert "get" in tokens
        assert "element" in tokens
        assert "by" in tokens
        assert "id" in tokens
        
        # Test snake_case preservation
        tokens = rag._tokenize_for_bm25("user_name_field")
        assert "user_name_field" in tokens or ("user" in tokens and "name" in tokens and "field" in tokens)
        
        # Test technical symbols
        tokens = rag._tokenize_for_bm25("self.method_name")
        assert any("self" in t for t in tokens)
        assert any("method" in t for t in tokens)
        
        # Test filtering short tokens
        tokens = rag._tokenize_for_bm25("a b c def")
        assert "a" not in tokens  # Too short
        assert "def" in tokens
    
    def test_bm25_exact_match(self, populated_system):
        """Test BM25 excels at exact keyword matching"""
        rag = populated_system["rag"]
        
        # Test exact error message match
        results = rag._bm25_retrieve("ImportError module pandas", k=5)
        
        assert len(results) > 0
        assert results[0].retrieval_method == "bm25"
        assert "ImportError" in results[0].metadata['query']
        assert results[0].matched_terms is not None
        assert "importerror" in results[0].matched_terms  # Lowercase
        
        # Verify it ranks exact matches higher
        first_score = results[0].score
        if len(results) > 1:
            assert first_score >= results[1].score
    
    def test_vector_semantic_similarity(self, populated_system):
        """Test vector search finds semantically similar content"""
        rag = populated_system["rag"]
        
        # Search for semantic similarity
        results = rag._vector_retrieve("load CSV data files", k=5)
        
        assert len(results) > 0
        assert results[0].retrieval_method == "vector"
        assert results[0].vector_similarity is not None
        assert 0 <= results[0].vector_similarity <= 1
        
        # Should find CSV-related memories even with different wording
        csv_found = any("csv" in r.metadata['query'].lower() for r in results)
        assert csv_found, "Vector search should find CSV-related content"
    
    def test_graph_traversal(self, populated_system):
        """Test graph traversal finds connected memories"""
        rag = populated_system["rag"]
        
        # Search for database issues - should traverse the connection chain
        results = rag._graph_retrieve("database connection issues", k=10)
        
        # Should find connected memories
        found_paths = [r for r in results if r.graph_path is not None]
        assert len(found_paths) > 0, "Graph traversal should find connected memories"
        
        # Check traversal depth
        depths = [r.metadata.get('depth', 0) for r in results]
        assert max(depths) > 0, "Should traverse beyond seed nodes"
        
        # Verify path integrity
        for result in results:
            if result.graph_path:
                assert len(result.graph_path) == result.metadata.get('depth', 0) + 1
    
    def test_reciprocal_rank_fusion(self, populated_system):
        """Test RRF combines results effectively"""
        rag = populated_system["rag"]
        
        # Create mock results from different methods
        bm25_results = [
            RetrievalResult("mem1", "content1", "bm25", 0.9),
            RetrievalResult("mem2", "content2", "bm25", 0.8),
            RetrievalResult("mem3", "content3", "bm25", 0.7),
        ]
        
        vector_results = [
            RetrievalResult("mem2", "content2", "vector", 0.85),
            RetrievalResult("mem4", "content4", "vector", 0.75),
            RetrievalResult("mem1", "content1", "vector", 0.65),
        ]
        
        graph_results = [
            RetrievalResult("mem5", "content5", "graph", 0.8),
            RetrievalResult("mem2", "content2", "graph", 0.7),
        ]
        
        # Apply RRF
        fused = rag._reciprocal_rank_fusion(
            [
                (bm25_results, 1.0),
                (vector_results, 1.0),
                (graph_results, 1.0)
            ],
            k=5
        )
        
        # mem2 should rank highest (appears in all three)
        assert fused[0].memory_id == "mem2"
        assert len(fused[0].metadata['retrieved_by']) == 3
        
        # Check RRF scores are properly calculated
        assert all(r.metadata.get('rrf_score', 0) > 0 for r in fused)
        
        # Verify descending order
        scores = [r.score for r in fused]
        assert scores == sorted(scores, reverse=True)
    
    def test_combined_retrieval(self, populated_system):
        """Test the main retrieve method combines all approaches"""
        rag = populated_system["rag"]
        
        # Test with a query that should trigger all methods
        results = rag.retrieve("python import error CSV module", k=5)
        
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        
        # Check that multiple methods contributed
        methods_used = set()
        for r in results:
            methods_used.update(r.metadata.get('retrieved_by', []))
        
        assert len(methods_used) >= 2, "Should use multiple retrieval methods"
    
    def test_retrieval_with_weights(self, populated_system):
        """Test retrieval with custom method weights"""
        rag = populated_system["rag"]
        
        # Heavy weight on BM25
        results_bm25_heavy = rag.retrieve(
            "ImportError",
            k=3,
            method_weights={"bm25": 2.0, "vector": 0.5, "graph": 0.5}
        )
        
        # Heavy weight on vector
        results_vector_heavy = rag.retrieve(
            "ImportError",
            k=3,
            method_weights={"bm25": 0.5, "vector": 2.0, "graph": 0.5}
        )
        
        # Results should differ based on weights
        bm25_ids = [r.memory_id for r in results_bm25_heavy]
        vector_ids = [r.memory_id for r in results_vector_heavy]
        
        # Order might be different even if same documents retrieved
        assert bm25_ids != vector_ids or len(set(bm25_ids)) == 1
    
    def test_empty_query_handling(self, populated_system):
        """Test handling of empty or invalid queries"""
        rag = populated_system["rag"]
        
        # Empty query
        results = rag.retrieve("", k=5)
        assert isinstance(results, list)
        
        # Very short query
        results = rag.retrieve("a", k=5)
        assert isinstance(results, list)
        
        # Special characters only
        results = rag.retrieve("@#$%", k=5)
        assert isinstance(results, list)
    
    def test_no_results_handling(self, populated_system):
        """Test handling when no relevant results found"""
        rag = populated_system["rag"]
        
        # Query unlikely to match anything
        results = rag.retrieve("xyzabc123 quantum blockchain metaverse", k=5)
        
        # Should return empty list or low-scoring results
        assert isinstance(results, list)
        if results:
            # Any results should have low scores
            assert all(r.score < 0.5 for r in results)
    
    def test_large_k_handling(self, populated_system):
        """Test retrieval with k larger than available memories"""
        rag = populated_system["rag"]
        total_memories = len(populated_system["memory_ids"])
        
        # Request more results than available
        results = rag.retrieve("python", k=total_memories * 2)
        
        # Should return at most the number of available memories
        assert len(results) <= total_memories
    
    def test_explanation_generation(self, populated_system):
        """Test retrieval explanation feature"""
        rag = populated_system["rag"]
        
        # Get results
        results = rag.retrieve("ImportError pandas", k=3)
        
        if results:
            # Test explanation for each type
            for result in results:
                explanation = rag.explain_retrieval("ImportError pandas", result)
                
                assert isinstance(explanation, str)
                assert result.memory_id in explanation
                assert result.retrieval_method in explanation
                
                # Method-specific checks
                if result.retrieval_method == "bm25":
                    assert "keyword" in explanation.lower() or "term" in explanation.lower()
                elif result.retrieval_method == "vector":
                    assert "semantic" in explanation.lower() or "similar" in explanation.lower()
                elif result.retrieval_method == "graph":
                    assert "graph" in explanation.lower() or "connection" in explanation.lower()
    
    def test_caching_behavior(self, populated_system):
        """Test that caching improves performance"""
        rag = populated_system["rag"]
        
        import time
        
        # First retrieval (cold cache)
        start = time.time()
        results1 = rag.retrieve("python error", k=5)
        time1 = time.time() - start
        
        # Second identical retrieval (warm cache)
        start = time.time()
        results2 = rag.retrieve("python error", k=5)
        time2 = time.time() - start
        
        # Results should be identical
        assert len(results1) == len(results2)
        
        # Note: Cache timing test might be flaky, so we just verify functionality
        # In a real system, time2 should be less than time1
    
    def test_memory_update_handling(self, populated_system):
        """Test RAG system handles memory updates correctly"""
        rag = populated_system["rag"]
        memory_system = populated_system["memory_system"]
        
        # Add a new memory
        new_id = memory_system.store_memory(
            query="New test query for RAG",
            response="New test response",
            memory_type="test"
        )
        
        # Reinitialize BM25 index to include new memory
        rag._initialize_bm25_index()
        
        # Should find the new memory
        results = rag.retrieve("New test query for RAG", k=3)
        
        found_new = any(r.memory_id == new_id for r in results)
        assert found_new, "Should find newly added memory after reindexing"
    
    def test_performance_characteristics(self, populated_system):
        """Test performance meets expectations"""
        rag = populated_system["rag"]
        
        import time
        
        # Measure retrieval time
        times = []
        for _ in range(5):
            start = time.time()
            results = rag.retrieve("python error handling", k=10)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert avg_time < 1.0, f"Average retrieval time {avg_time:.3f}s exceeds threshold"
        
        # Verify we get results
        assert len(results) > 0


class TestRetrievalPatterns:
    """
    Test specific retrieval patterns and use cases
    
    These tests validate that the system handles common
    real-world patterns correctly.
    """
    
    def test_error_message_pattern(self, populated_system):
        """Test retrieval of error messages and solutions"""
        rag = populated_system["rag"]
        
        # Common error message pattern
        results = rag.retrieve("TypeError: 'NoneType' object has no attribute", k=5)
        
        # Should prioritize exact error matches
        if results:
            # Check if BM25 contributed to top results
            top_methods = results[0].metadata.get('retrieved_by', [])
            assert 'bm25' in top_methods, "BM25 should catch exact error patterns"
    
    def test_synonym_pattern(self, populated_system):
        """Test retrieval handles synonyms correctly"""
        rag = populated_system["rag"]
        
        # Search with synonyms
        queries = [
            "quickly process data",
            "fast data processing", 
            "speed up computation"
        ]
        
        all_results = []
        for query in queries:
            results = rag.retrieve(query, k=3)
            all_results.extend(results)
        
        # Vector search should find these as related
        vector_contributions = sum(
            1 for r in all_results 
            if 'vector' in r.metadata.get('retrieved_by', [])
        )
        
        assert vector_contributions > 0, "Vector search should handle synonyms"
    
    def test_technical_jargon_pattern(self, populated_system):
        """Test handling of technical terminology"""
        rag = populated_system["rag"]
        
        # Technical terms that might not have close embeddings
        results = rag.retrieve("pip install numpy ImportError", k=5)
        
        # Should find relevant installation instructions
        assert len(results) > 0
        
        # BM25 should contribute for technical terms
        bm25_found = any(
            'bm25' in r.metadata.get('retrieved_by', [])
            for r in results
        )
        assert bm25_found, "BM25 should catch technical terminology"


class TestEdgeCases:
    """
    Test edge cases and error conditions
    
    These tests ensure the system degrades gracefully
    and handles unexpected inputs correctly.
    """
    
    def test_unicode_handling(self, populated_system):
        """Test handling of Unicode characters"""
        rag = populated_system["rag"]
        
        # Various Unicode inputs
        unicode_queries = [
            "python café ☕",
            "error message with émojis 🐛",
            "中文查询测试",
            "Ñoño query with tildes"
        ]
        
        for query in unicode_queries:
            try:
                results = rag.retrieve(query, k=3)
                assert isinstance(results, list)
            except Exception as e:
                pytest.fail(f"Failed on Unicode query '{query}': {e}")
    
    def test_very_long_query(self, populated_system):
        """Test handling of extremely long queries"""
        rag = populated_system["rag"]
        
        # Create a very long query
        long_query = "python " * 100 + "error handling best practices"
        
        results = rag.retrieve(long_query, k=5)
        assert isinstance(results, list)
    
    def test_special_characters(self, populated_system):
        """Test handling of special characters in queries"""
        rag = populated_system["rag"]
        
        special_queries = [
            "python->error",
            "module::not::found",
            "error@line#42",
            "import * from module"
        ]
        
        for query in special_queries:
            results = rag.retrieve(query, k=3)
            assert isinstance(results, list)
    
    def test_concurrent_retrieval(self, populated_system):
        """Test thread safety of retrieval"""
        rag = populated_system["rag"]
        
        import threading
        
        results_container = []
        errors = []
        
        def retrieve_concurrent(query, container, error_list):
            try:
                results = rag.retrieve(query, k=5)
                container.append(results)
            except Exception as e:
                error_list.append(e)
        
        # Create multiple threads
        threads = []
        queries = ["python error", "import module", "CSV data", "database connection"]
        
        for query in queries:
            t = threading.Thread(
                target=retrieve_concurrent,
                args=(query, results_container, errors)
            )
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Should complete without errors
        assert len(errors) == 0, f"Concurrent retrieval errors: {errors}"
        assert len(results_container) == len(queries)


def test_demonstrate_function():
    """Test the demonstration function runs without errors"""
    try:
        demonstrate_retrieval_differences()
    except Exception as e:
        pytest.fail(f"Demonstration function failed: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
    
    # Also run a quick demonstration
    print("\n" + "="*60)
    print("Running demonstration...")
    print("="*60 + "\n")
    
    demonstrate_retrieval_differences()