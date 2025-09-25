"""
Fast Intelligent Routing using Claude 3.5 Haiku
Ultra-low latency routing decisions (~0.5s) 
"""

import asyncio
import logging
import json
from typing import Dict, Tuple, Optional
from functools import lru_cache
import hashlib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from adam.llm.client import UnifiedLLMClient
from adam.llm.config import LLMConfig

logger = logging.getLogger(__name__)


class FastRoutingService:
    """
    Ultra-fast routing decisions using Claude 3.5 Haiku
    Includes caching for repeated queries
    """
    
    def __init__(self):
        self.client = UnifiedLLMClient()
        self.config = LLMConfig()
        
        # Cache for routing decisions (query hash -> decision)
        self._routing_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
    def _hash_query(self, query: str) -> str:
        """Create a hash of the query for caching"""
        # Normalize the query (lowercase, strip whitespace)
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    async def route_query(self, query: str) -> Dict[str, any]:
        """
        Route a query to the appropriate model and determine if knowledge bases are needed
        Returns in ~0.5s with Claude Haiku
        """
        # Check cache first
        query_hash = self._hash_query(query)
        if query_hash in self._routing_cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit for query: '{query[:50]}...'")
            return self._routing_cache[query_hash]
        
        self._cache_misses += 1
        
        # Build a concise routing prompt
        routing_prompt = f"""Analyze this query and provide routing information in JSON:
Query: "{query}"

Route strictly based on these rules:
- "reasoning": Code generation, debugging, complex algorithms, system architecture, multi-step problem solving
- "standard": DBT models, SQL queries, PDT conversion, data analysis, documentation, explanations, any technical work
- "fast": ONLY greetings (hi/hello/thanks), yes/no questions, simple acknowledgments

IMPORTANT routing priorities:
1. If mentions DBT, PDT, SQL, model, table, query, data → ALWAYS "standard" (use grok-4-fast-non-reasoning)
2. If asks to create/write/implement code or debug → "reasoning" (use grok-4-fast-reasoning)
3. If technical question or requires analysis → "standard"
4. ONLY use "fast" for trivial greetings or one-word answers

Respond with ONLY valid JSON:
{{"model_tier": "fast|standard|reasoning", "needs_dbt": true/false, "needs_sql": true/false, "confidence": 0.0-1.0}}"""

        try:
            # Use Claude Haiku for ultra-fast routing
            response = await self.client.complete(
                prompt=routing_prompt,
                model="claude-3.5-haiku",
                temperature=0.0,  # Deterministic
                max_tokens=100,   # Small response
                stream=False      # No streaming for routing
            )
            
            # Parse response
            result = json.loads(response.content.strip())
            
            # Map to our models - using fast Grok models
            model_mapping = {
                "fast": "grok-4-fast-non-reasoning",      # Fast non-reasoning for simple responses
                "standard": "grok-4-fast-non-reasoning",  # Fast non-reasoning for medium complexity
                "reasoning": "grok-4-fast-reasoning"       # Fast reasoning for code/complex tasks
            }
            
            decision = {
                "model": model_mapping.get(result["model_tier"], "grok-4-fast-non-reasoning"),  # Default to fast non-reasoning if tier unknown
                "needs_dbt": result.get("needs_dbt", False),
                "needs_sql": result.get("needs_sql", False),
                "confidence": result.get("confidence", 0.8),
                "tier": result["model_tier"]
            }
            
            # Cache the decision
            self._routing_cache[query_hash] = decision
            
            logger.info(f"Routed to {decision['model']} (confidence: {decision['confidence']:.2f}, "
                       f"DBT: {decision['needs_dbt']}, SQL: {decision['needs_sql']})")
            
            return decision
            
        except Exception as e:
            logger.error(f"Haiku routing failed: {e}, falling back to keyword detection")
            
            # Fallback to simple keyword detection
            return self._fallback_routing(query)
    
    def _fallback_routing(self, query: str) -> Dict[str, any]:
        """Simple keyword-based fallback routing"""
        query_lower = query.lower()
        
        # Better complexity detection - default to standard for technical queries
        if any(word in query_lower for word in ['write code', 'create function', 'implement', 'debug', 'fix', 'algorithm', 'develop']):
            model = "grok-4-reasoning"  # Code generation and debugging
            tier = "reasoning"
        elif any(word in query_lower for word in ['dbt', 'pdt', 'sql', 'model', 'table', 'query', 'convert', 'document', 'explain', 'analyze', 'help me']):
            model = "grok-4"  # DBT/SQL/technical work
            tier = "standard"
        elif len(query) < 20 and any(word in query_lower for word in ['hi', 'hello', 'thanks', 'ok', 'yes', 'no']):
            model = "gpt-4o-mini"  # Only for very simple greetings
            tier = "fast"
        else:
            # Default to standard for anything technical
            model = "grok-4"  # Default to standard model for safety
            tier = "standard"
        
        # Detect knowledge needs
        needs_dbt = any(word in query_lower for word in ['dbt', 'model', 'transformation', 'staging', 'mart'])
        needs_sql = any(word in query_lower for word in ['sql', 'query', 'select', 'join', 'table', 'database'])
        
        return {
            "model": model,
            "needs_dbt": needs_dbt,
            "needs_sql": needs_sql,
            "confidence": 0.5,  # Lower confidence for fallback
            "tier": tier
        }
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._routing_cache),
            "hit_rate": hit_rate
        }
    
    def clear_cache(self):
        """Clear the routing cache"""
        self._routing_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Routing cache cleared")