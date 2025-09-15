"""
Temporal Memory Scoring - Natural recency decay for memory retrieval

This module implements time-aware memory scoring that works alongside
BM25, vector search, and graph traversal without hard cutoffs.
"""
import math
from datetime import datetime
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class TemporalScoringConfig:
    """Configuration for temporal scoring"""
    # Weight given to temporal score vs semantic score
    temporal_weight: float = 0.3  # 30% time, 70% semantic
    
    # Half-life in days - after this many days, temporal score is halved
    half_life_days: float = 30.0
    
    # Minimum temporal score (prevents very old memories from getting 0)
    min_temporal_score: float = 0.1


class TemporalMemoryScorer:
    """
    Adds natural time decay to memory scoring without hard cutoffs.
    
    This ensures recent memories are preferred but doesn't exclude
    older memories that might be the only relevant ones.
    """
    
    def __init__(self, config: TemporalScoringConfig = None):
        self.config = config or TemporalScoringConfig()
    
    def calculate_temporal_score(self, timestamp: datetime) -> float:
        """
        Calculate temporal relevance score using exponential decay.
        
        Recent memories get scores close to 1.0, older memories decay
        exponentially but never go below min_temporal_score.
        """
        now = datetime.now()
        if timestamp.tzinfo:
            now = datetime.now(timestamp.tzinfo)
        
        age_days = max(0, (now - timestamp).total_seconds() / 86400)
        
        # Exponential decay: score = e^(-λt) where λ = ln(2)/half_life
        decay_rate = 0.693 / self.config.half_life_days  # ln(2) ≈ 0.693
        temporal_score = math.exp(-decay_rate * age_days)
        
        # Apply minimum score to prevent very old memories from being completely ignored
        return max(temporal_score, self.config.min_temporal_score)
    
    def combine_scores(self, semantic_score: float, temporal_score: float) -> float:
        """
        Combine semantic and temporal scores with configured weights.
        """
        return (
            (1 - self.config.temporal_weight) * semantic_score +
            self.config.temporal_weight * temporal_score
        )
    
    def score_memory(self, memory: Dict[str, Any]) -> float:
        """
        Score a single memory considering both semantic relevance and recency.
        """
        # Get base semantic score (from BM25, vector search, or graph)
        semantic_score = memory.get('similarity', 0.5)
        
        # Special handling for memories with negative similarity
        # These are often important memories that have decayed in strength
        if semantic_score < 0:
            # Convert negative similarity to a small positive value
            # This ensures temporal scoring can still help surface them
            semantic_score = 0.1 + (semantic_score + 1) * 0.05  # Maps [-1, 0] to [0.1, 0.15]
        
        # Get timestamp and calculate temporal score
        metadata = memory.get('metadata', {})
        timestamp_str = metadata.get('timestamp', '')
        
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                temporal_score = self.calculate_temporal_score(timestamp)
            except:
                temporal_score = self.config.min_temporal_score
        else:
            temporal_score = self.config.min_temporal_score
        
        # Store individual scores for transparency
        memory['semantic_score'] = semantic_score
        memory['temporal_score'] = temporal_score
        
        # Return combined score
        return self.combine_scores(semantic_score, temporal_score)
    
    def rerank_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank memories using temporal scoring.
        
        This is applied AFTER the three-stage retrieval (BM25, vector, graph)
        to add time awareness without breaking existing functionality.
        """
        # Score each memory
        for memory in memories:
            memory['combined_score'] = self.score_memory(memory)
        
        # Sort by combined score
        memories.sort(key=lambda m: m['combined_score'], reverse=True)
        
        return memories
    
    def explain_scoring(self, memory: Dict[str, Any]) -> str:
        """
        Explain why a memory received its score - useful for debugging.
        """
        semantic = memory.get('semantic_score', 0)
        temporal = memory.get('temporal_score', 0)
        combined = memory.get('combined_score', 0)
        
        metadata = memory.get('metadata', {})
        timestamp_str = metadata.get('timestamp', 'unknown')
        
        if timestamp_str != 'unknown':
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                age_days = (datetime.now(timestamp.tzinfo) - timestamp).days
                age_str = f"{age_days} days old"
            except:
                age_str = "unknown age"
        else:
            age_str = "unknown age"
        
        return (
            f"Score breakdown:\n"
            f"  Semantic: {semantic:.3f} (70% weight)\n"
            f"  Temporal: {temporal:.3f} (30% weight, {age_str})\n"
            f"  Combined: {combined:.3f}"
        )


# Example usage showing how it handles your DAG scenario
if __name__ == "__main__":
    from datetime import timedelta
    
    scorer = TemporalMemoryScorer()
    
    # Simulate memories at different ages
    test_memories = [
        {
            "content": "DAG with new_fee_repricing_user",
            "similarity": 0.9,  # Very relevant
            "metadata": {"timestamp": (datetime.now() - timedelta(days=50)).isoformat()}
        },
        {
            "content": "Random recent conversation",
            "similarity": 0.3,  # Not very relevant
            "metadata": {"timestamp": (datetime.now() - timedelta(days=1)).isoformat()}
        }
    ]
    
    # Score and rank
    ranked = scorer.rerank_memories(test_memories)
    
    print("Memory ranking:")
    for i, memory in enumerate(ranked):
        print(f"\n{i+1}. {memory['content']}")
        print(scorer.explain_scoring(memory))
    
    # Result: The 50-day old DAG ranks #1 because its high semantic relevance
    # (0.9) outweighs the recency advantage of the 1-day old memory (0.3)