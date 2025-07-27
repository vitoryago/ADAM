"""
Advanced memory service for ADAM v2.0
Integrates the best features from v1.0's advanced memory system
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import json
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict
import numpy as np

# Add parent directory to path for ADAM imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from existing memory service
from .memory_service import ProjectMemoryService, MemorySearchResult, MemoryType

# Try to import ADAM v1 components
try:
    from adam.llm.query_analyzer import QueryAnalyzer, QueryComplexity
    from adam.memory import ADAMMemoryAdvanced
    ADAM_V1_AVAILABLE = True
except ImportError:
    ADAM_V1_AVAILABLE = False
    QueryComplexity = None

# Try to import BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logging.warning("BM25 not available. Install with: pip install rank-bm25")

logger = logging.getLogger(__name__)


class MemoryWorthiness(Enum):
    """Memory worthiness levels"""
    ESSENTIAL = "essential"      # Must store (errors, expensive)
    VALUABLE = "valuable"        # Should store (complex, patterns)
    MODERATE = "moderate"        # Maybe store (useful info)
    TRIVIAL = "trivial"         # Don't store (simple facts)


@dataclass
class EnhancedSearchResult(MemorySearchResult):
    """Enhanced search result with additional metadata"""
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    combined_score: float = 0.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    success_rate: float = 1.0


class AdvancedMemoryService(ProjectMemoryService):
    """
    Advanced memory service with features from ADAM v1.0:
    - BM25 + Semantic search fusion
    - Intelligent memory evaluation
    - Memory versioning and success tracking
    """
    
    def __init__(self, project_id: str, project_name: str):
        super().__init__(project_id, project_name)
        
        # Initialize query analyzer if available
        if ADAM_V1_AVAILABLE:
            self.query_analyzer = QueryAnalyzer()
        else:
            self.query_analyzer = None
        
        # Cache for BM25 index
        self._bm25_index = None
        self._bm25_docs = []
        self._bm25_ids = []
        self._last_index_update = datetime.now()
    
    async def evaluate_memory_worthiness(
        self,
        query: str,
        response: str,
        cost: float,
        model: str
    ) -> Tuple[MemoryWorthiness, Dict[str, Any]]:
        """
        Evaluate if a query-response pair is worth storing.
        Based on ADAM v1.0's sophisticated evaluation.
        """
        metadata = {
            "evaluated_at": datetime.now().isoformat(),
            "factors": {}
        }
        
        # Factor 1: Cost (most objective)
        if cost > 0.01:
            metadata["factors"]["cost"] = "expensive"
            return MemoryWorthiness.ESSENTIAL, metadata
        elif cost > 0.001:
            metadata["factors"]["cost"] = "moderate"
            cost_score = 0.5
        else:
            metadata["factors"]["cost"] = "cheap"
            cost_score = 0.1
        
        # Factor 2: Query complexity
        complexity_score = 0.3
        if self.query_analyzer:
            complexity, _ = self.query_analyzer.analyze_query(query)
            metadata["factors"]["complexity"] = complexity.value
            
            if complexity == QueryComplexity.HIGH:
                complexity_score = 0.9
            elif complexity == QueryComplexity.MEDIUM:
                complexity_score = 0.6
            else:
                complexity_score = 0.2
        
        # Factor 3: Response characteristics
        response_score = 0.3
        response_length = len(response.split())
        
        # Check for code blocks
        if "```" in response:
            metadata["factors"]["has_code"] = True
            response_score += 0.3
        
        # Check for structured content
        if any(marker in response for marker in ["1.", "- ", "* ", "##"]):
            metadata["factors"]["structured"] = True
            response_score += 0.2
        
        # Length consideration
        if response_length > 200:
            metadata["factors"]["detailed"] = True
            response_score += 0.2
        elif response_length < 20:
            metadata["factors"]["brief"] = True
            response_score -= 0.2
        
        # Factor 4: Query patterns
        query_lower = query.lower()
        
        # Error/debugging queries are always valuable
        if any(word in query_lower for word in ["error", "bug", "fix", "issue", "problem"]):
            metadata["factors"]["error_related"] = True
            return MemoryWorthiness.ESSENTIAL, metadata
        
        # Implementation queries are valuable
        if any(word in query_lower for word in ["implement", "create", "build", "design"]):
            metadata["factors"]["implementation"] = True
            response_score += 0.3
        
        # Calculate final score
        final_score = (cost_score * 0.4 + 
                      complexity_score * 0.3 + 
                      response_score * 0.3)
        
        metadata["factors"]["final_score"] = round(final_score, 2)
        
        # Determine worthiness
        if final_score >= 0.7:
            return MemoryWorthiness.ESSENTIAL, metadata
        elif final_score >= 0.5:
            return MemoryWorthiness.VALUABLE, metadata
        elif final_score >= 0.3:
            return MemoryWorthiness.MODERATE, metadata
        else:
            return MemoryWorthiness.TRIVIAL, metadata
    
    async def store_memory_with_evaluation(
        self,
        query: str,
        response: str,
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
        cost: Optional[float] = None,
        model: Optional[str] = None
    ) -> Optional[str]:
        """
        Store memory only if it's worth storing.
        Returns memory ID if stored, None if rejected.
        """
        # Evaluate worthiness
        worthiness, eval_metadata = await self.evaluate_memory_worthiness(
            query, response, cost or 0.0, model or "unknown"
        )
        
        # Don't store trivial memories
        if worthiness == MemoryWorthiness.TRIVIAL:
            logger.debug(f"Rejected trivial memory: {query[:50]}...")
            return None
        
        # Prepare enhanced metadata
        enhanced_metadata = metadata or {}
        enhanced_metadata.update({
            "worthiness": worthiness.value,
            "evaluation": eval_metadata,
            "model": model,
            "query_length": len(query.split()),
            "response_length": len(response.split()),
            "stored_at": datetime.now().isoformat()
        })
        
        # Store with combined content for better search
        content = f"Q: {query}\n\nA: {response}"
        
        memory_id = await self.store_memory(
            content=content,
            memory_type=memory_type,
            metadata=enhanced_metadata,
            conversation_id=conversation_id,
            cost=cost
        )
        
        # Invalidate BM25 cache
        self._bm25_index = None
        
        logger.info(f"Stored {worthiness.value} memory: {memory_id}")
        return memory_id
    
    async def advanced_search(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
        min_relevance: float = 0.3,
        conversation_id: Optional[str] = None,
        use_bm25: bool = True,
        use_semantic: bool = True
    ) -> List[EnhancedSearchResult]:
        """
        Advanced search using BM25 + Semantic fusion (inspired by v1.0's RAG)
        """
        results = []
        
        # Get both BM25 and semantic results
        bm25_results = {}
        semantic_results = {}
        
        # Semantic search (using ChromaDB)
        if use_semantic:
            semantic_search = await self.search_memories(
                query=query,
                limit=limit * 2,  # Get more for fusion
                memory_types=memory_types,
                min_relevance=min_relevance * 0.7,  # Lower threshold for fusion
                conversation_id=conversation_id
            )
            
            for result in semantic_search:
                semantic_results[result.id] = result
        
        # BM25 search
        if use_bm25 and BM25_AVAILABLE:
            bm25_search = await self._bm25_search(
                query=query,
                limit=limit * 2,
                memory_types=memory_types,
                conversation_id=conversation_id
            )
            
            for result in bm25_search:
                bm25_results[result.id] = result
        
        # Reciprocal Rank Fusion (RRF)
        all_ids = set(semantic_results.keys()) | set(bm25_results.keys())
        k = 60  # RRF constant
        
        for memory_id in all_ids:
            # Calculate RRF score
            semantic_rank = 0
            bm25_rank = 0
            
            # Find ranks
            if memory_id in semantic_results:
                semantic_list = list(semantic_results.keys())
                semantic_rank = semantic_list.index(memory_id) + 1 if memory_id in semantic_list else len(semantic_list) + 1
            
            if memory_id in bm25_results:
                bm25_list = list(bm25_results.keys())
                bm25_rank = bm25_list.index(memory_id) + 1 if memory_id in bm25_list else len(bm25_list) + 1
            
            # RRF formula
            rrf_score = 0
            if semantic_rank > 0:
                rrf_score += 1 / (k + semantic_rank)
            if bm25_rank > 0:
                rrf_score += 1 / (k + bm25_rank)
            
            # Get the result data
            if memory_id in semantic_results:
                base_result = semantic_results[memory_id]
            else:
                base_result = bm25_results[memory_id]
            
            # Create enhanced result
            enhanced = EnhancedSearchResult(
                id=base_result.id,
                content=base_result.content,
                metadata=base_result.metadata,
                relevance_score=base_result.relevance_score,
                memory_type=base_result.memory_type,
                timestamp=base_result.timestamp,
                semantic_score=semantic_results.get(memory_id, base_result).relevance_score if memory_id in semantic_results else 0,
                bm25_score=bm25_results[memory_id].relevance_score if memory_id in bm25_results else 0,
                combined_score=rrf_score,
                access_count=base_result.metadata.get("access_count", 0),
                last_accessed=datetime.fromisoformat(base_result.metadata["last_accessed"]) if "last_accessed" in base_result.metadata else None,
                success_rate=base_result.metadata.get("success_rate", 1.0)
            )
            
            # Apply minimum relevance filter on combined score
            if enhanced.combined_score >= min_relevance * 0.01:  # Adjust threshold for RRF scale
                results.append(enhanced)
        
        # Sort by combined score and limit
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results[:limit]
    
    async def _bm25_search(
        self,
        query: str,
        limit: int,
        memory_types: Optional[List[MemoryType]] = None,
        conversation_id: Optional[str] = None
    ) -> List[MemorySearchResult]:
        """BM25 keyword-based search"""
        # Build or update index
        await self._update_bm25_index()
        
        if not self._bm25_index or not self._bm25_docs:
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self._bm25_index.get_scores(query_tokens)
        
        # Get top results
        top_indices = np.argsort(scores)[::-1][:limit * 2]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                memory_id = self._bm25_ids[idx]
                
                # Get full memory data
                memory_data = self.collection.get(ids=[memory_id])
                
                if memory_data and memory_data["ids"]:
                    metadata = memory_data["metadatas"][0]
                    
                    # Apply filters
                    if memory_types:
                        type_values = [t.value if hasattr(t, 'value') else str(t) for t in memory_types]
                        if metadata.get("memory_type") not in type_values:
                            continue
                    
                    if conversation_id and metadata.get("conversation_id") != conversation_id:
                        continue
                    
                    result = MemorySearchResult(
                        id=memory_id,
                        content=memory_data["documents"][0],
                        metadata=metadata,
                        relevance_score=float(scores[idx]) / (max(scores) + 1e-6),  # Normalize
                        memory_type=metadata.get("memory_type", "conversation"),
                        timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat()))
                    )
                    results.append(result)
        
        return results[:limit]
    
    async def _update_bm25_index(self):
        """Update BM25 index if needed"""
        # Only update if it's been more than 5 minutes
        if (self._bm25_index and 
            (datetime.now() - self._last_index_update).seconds < 300):
            return
        
        if not BM25_AVAILABLE:
            return
        
        # Get all memories
        all_memories = self.collection.get()
        
        if not all_memories or not all_memories["ids"]:
            return
        
        # Prepare documents for BM25
        self._bm25_docs = []
        self._bm25_ids = []
        
        for i, doc in enumerate(all_memories["documents"]):
            # Tokenize and lowercase
            tokens = doc.lower().split()
            self._bm25_docs.append(tokens)
            self._bm25_ids.append(all_memories["ids"][i])
        
        # Build BM25 index
        self._bm25_index = BM25Okapi(self._bm25_docs)
        self._last_index_update = datetime.now()
    
    async def update_memory_success(
        self,
        memory_id: str,
        success: bool,
        feedback: Optional[str] = None
    ):
        """Update memory success rate based on user feedback"""
        try:
            # Get current memory
            result = self.collection.get(ids=[memory_id])
            
            if result and result["metadatas"]:
                metadata = result["metadatas"][0]
                
                # Update success tracking
                total_uses = metadata.get("total_uses", 0) + 1
                successful_uses = metadata.get("successful_uses", 0) + (1 if success else 0)
                
                metadata.update({
                    "total_uses": total_uses,
                    "successful_uses": successful_uses,
                    "success_rate": successful_uses / total_uses,
                    "last_feedback": feedback,
                    "last_feedback_at": datetime.now().isoformat()
                })
                
                # Update in collection
                self.collection.update(
                    ids=[memory_id],
                    metadatas=[metadata]
                )
                
                logger.info(f"Updated memory {memory_id} success rate: {metadata['success_rate']:.2f}")
                
        except Exception as e:
            logger.error(f"Failed to update memory success: {e}")
    
    async def get_memory_versions(self, original_id: str) -> List[MemorySearchResult]:
        """Get all versions of a memory (for tracking improvements)"""
        # Search for memories with parent_id
        where_clause = {"parent_id": original_id}
        
        results = self.collection.query(
            query_texts=[""],  # Empty query to get all
            n_results=10,
            where=where_clause
        )
        
        versions = []
        if results and results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                
                version = MemorySearchResult(
                    id=memory_id,
                    content=results["documents"][0][i],
                    metadata=metadata,
                    relevance_score=1.0,
                    memory_type=metadata.get("memory_type", "conversation"),
                    timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat()))
                )
                versions.append(version)
        
        # Sort by version number or timestamp
        versions.sort(key=lambda x: x.timestamp)
        return versions