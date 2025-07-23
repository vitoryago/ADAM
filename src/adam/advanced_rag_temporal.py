"""
Enhanced Advanced RAG with Temporal Scoring Integration

This shows how to integrate natural time-based scoring into the existing
three-stage retrieval system (BM25, Vector, Graph) without breaking anything.
"""

from typing import List, Dict, Tuple, Optional, Any
from adam.advanced_rag import AdvancedRAGSystem, RetrievalResult
from adam.temporal_memory_scoring import TemporalMemoryScorer, TemporalScoringConfig

class TemporalAdvancedRAGSystem(AdvancedRAGSystem):
    """
    Enhanced RAG system that adds temporal awareness to the existing
    three-stage retrieval without replacing any functionality.
    """
    
    def __init__(self, *args, temporal_config: TemporalScoringConfig = None, **kwargs):
        """
        Initialize with all existing parameters plus temporal scoring config
        """
        super().__init__(*args, **kwargs)
        self.temporal_scorer = TemporalMemoryScorer(temporal_config)
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        method_weights: Dict[str, float] = None,
        filters: Optional[Dict[str, Any]] = None,
        enable_temporal: bool = True
    ) -> List[RetrievalResult]:
        """
        Enhanced retrieve that adds temporal scoring after RRF fusion
        
        The process:
        1. BM25 retrieval (keyword matching)
        2. Vector retrieval (semantic similarity)
        3. Graph retrieval (following connections)
        4. Reciprocal Rank Fusion (combines all three)
        5. NEW: Temporal re-ranking (adds time awareness)
        """
        # Get results from the original three-stage retrieval
        combined_results = super().retrieve(query, k, method_weights, filters)
        
        # Apply temporal scoring if enabled
        if enable_temporal:
            combined_results = self._apply_temporal_scoring(combined_results)
        
        return combined_results[:k]  # Return top k after temporal re-ranking
    
    def _apply_temporal_scoring(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Apply temporal scoring to the already-fused results
        
        This preserves the original three-stage scores while adding
        time awareness as a final re-ranking step.
        """
        # Convert RetrievalResult objects to dict format for scorer
        memories = []
        for result in results:
            memory = {
                'memory_id': result.memory_id,
                'content': result.content,
                'similarity': result.score,  # RRF score becomes base similarity
                'metadata': result.metadata,
                'original_result': result  # Keep original for reconstruction
            }
            memories.append(memory)
        
        # Apply temporal scoring
        reranked_memories = self.temporal_scorer.rerank_memories(memories)
        
        # Convert back to RetrievalResult objects
        reranked_results = []
        for memory in reranked_memories:
            original_result = memory['original_result']
            
            # Create new result with updated score and metadata
            new_result = RetrievalResult(
                memory_id=original_result.memory_id,
                content=original_result.content,
                retrieval_method=original_result.retrieval_method,
                score=memory['combined_score'],  # New temporal-aware score
                metadata={
                    **original_result.metadata,
                    'original_rrf_score': original_result.score,
                    'semantic_score': memory['semantic_score'],
                    'temporal_score': memory['temporal_score'],
                    'combined_score': memory['combined_score']
                },
                matched_terms=original_result.matched_terms,
                vector_similarity=original_result.vector_similarity
            )
            reranked_results.append(new_result)
        
        return reranked_results
    
    def retrieve_with_explanation(
        self,
        query: str,
        k: int = 10,
        **kwargs
    ) -> Tuple[List[RetrievalResult], str]:
        """
        Retrieve with detailed explanation of scoring
        
        This helps debug why certain memories ranked where they did.
        """
        results = self.retrieve(query, k, **kwargs)
        
        explanation = f"Query: '{query}'\n\n"
        explanation += "Top Results with Scoring Breakdown:\n"
        explanation += "=" * 60 + "\n\n"
        
        for i, result in enumerate(results[:5]):  # Explain top 5
            explanation += f"{i+1}. Memory ID: {result.memory_id}\n"
            explanation += f"   Method: {result.retrieval_method}\n"
            
            # Extract scores from metadata
            meta = result.metadata
            explanation += f"   Original RRF Score: {meta.get('original_rrf_score', 'N/A'):.3f}\n"
            explanation += f"   Semantic Score: {meta.get('semantic_score', 'N/A'):.3f} (70% weight)\n"
            explanation += f"   Temporal Score: {meta.get('temporal_score', 'N/A'):.3f} (30% weight)\n"
            explanation += f"   Final Score: {result.score:.3f}\n"
            
            # Show content preview
            content_preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
            explanation += f"   Content: {content_preview}\n\n"
        
        return results, explanation


# Example showing your DAG scenario
if __name__ == "__main__":
    from datetime import datetime, timedelta
    
    # Simulate your scenario:
    # 1. You created a DAG 50 days ago
    # 2. You haven't created any DAGs since
    # 3. You ask "bring me back that DAG"
    
    print("Scenario: Retrieving a 50-day old DAG\n")
    
    # Simulate search results
    mock_results = [
        RetrievalResult(
            memory_id="dag_50days",
            content="DAG with new_fee_repricing_user and MARKETING_ANALYTICS",
            retrieval_method="vector",
            score=0.8,  # High semantic match
            metadata={
                "timestamp": (datetime.now() - timedelta(days=50)).isoformat(),
                "type": "code_pattern"
            }
        ),
        RetrievalResult(
            memory_id="recent_chat",
            content="Random conversation from yesterday",
            retrieval_method="bm25",
            score=0.4,  # Lower semantic match but recent
            metadata={
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                "type": "conversation"
            }
        )
    ]
    
    # Create temporal scorer
    scorer = TemporalMemoryScorer()
    
    # Convert to memory format and apply temporal scoring
    memories = [
        {
            'memory_id': r.memory_id,
            'content': r.content,
            'similarity': r.score,
            'metadata': r.metadata,
            'original_result': r
        }
        for r in mock_results
    ]
    
    reranked = scorer.rerank_memories(memories)
    
    print("Results after temporal scoring:")
    print("-" * 60)
    
    for i, memory in enumerate(reranked):
        print(f"\n{i+1}. {memory['content'][:50]}...")
        print(f"   Original score: {memory['similarity']:.3f}")
        print(f"   Temporal score: {memory['temporal_score']:.3f}")
        print(f"   Combined score: {memory['combined_score']:.3f}")
        
        # Calculate age
        timestamp_str = memory['metadata'].get('timestamp', '')
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str)
            age_days = (datetime.now() - timestamp).days
            print(f"   Age: {age_days} days")
    
    print("\nConclusion: The 50-day old DAG still ranks #1 because its high")
    print("semantic relevance (0.8) outweighs the temporal advantage of")
    print("the recent conversation (0.4), even with temporal scoring.")