"""
Intelligent Routing Service for ADAM v2
========================================

Integrates the modern LLM router with the existing ADAM system
"""

import logging
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add parent path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from adam.llm_router import SpecializedRouter, RoutingDecision, ModelTier
from adam.llm.config import LLMConfig

logger = logging.getLogger(__name__)


class IntelligentRoutingService:
    """
    Service that integrates LLM-based routing with ADAM's existing infrastructure
    """
    
    def __init__(self):
        # Initialize the router (can switch between Claude Haiku and GPT-4-mini)
        self.router = SpecializedRouter(use_anthropic=True)
        
        # Map model tiers to actual model names in ADAM's config
        self.model_mapping = {
            ModelTier.REASONING: "grok-4-reasoning",  # grok-4-reasoning for complex reasoning and code
            ModelTier.STANDARD: "grok-4",  # grok-4 for standard/complex queries
            ModelTier.FAST: "gpt-4o-mini"   # gpt-4o-mini for fast/basic queries (LOW latency)
        }
        
        # Track routing metrics
        self.metrics = {
            "total_routes": 0,
            "tier_usage": {tier: 0 for tier in ModelTier},
            "avg_confidence": 0.0,
            "total_cost": 0.0
        }
        
    async def route_and_configure(self, 
                                  query: str, 
                                  project_id: Optional[str] = None,
                                  conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Route a query and return configuration for LLM processing
        
        Args:
            query: User's query
            project_id: Current project ID for context
            conversation_id: Current conversation for context
            
        Returns:
            Configuration dict with model selection and parameters
        """
        # Build context for router
        context = {}
        if project_id:
            context["project_id"] = project_id
        if conversation_id:
            context["conversation_id"] = conversation_id
            
        # Get routing decision from LLM
        decision = await self.router.route_query(query, context)
        
        # Update metrics
        self._update_metrics(decision)
        
        # Build configuration for ADAM
        config = {
            "model": self.model_mapping[decision.model_tier],
            "reasoning": decision.reasoning,
            "confidence": decision.confidence,
            "memory_config": {
                "enabled": decision.requires_memory,
                "depth": decision.memory_depth,
                "include_cross_project": decision.memory_depth > 10  # Deep search includes other projects
            },
            "specialized_domain": decision.specialized_domain,
            "estimated_cost": decision.estimated_cost
        }
        
        # Add specialized prompts based on domain
        if decision.specialized_domain:
            config["system_prompt_addon"] = self._get_specialized_prompt(decision.specialized_domain)
            
        logger.info(f"Routed to {config['model']} (confidence: {decision.confidence:.2f}, "
                   f"domain: {decision.specialized_domain})")
        
        return config
    
    def _get_specialized_prompt(self, domain: str) -> str:
        """
        Get specialized system prompt based on domain
        """
        prompts = {
            "sql": """You are a SQL expert. Focus on:
                     - Query optimization and performance
                     - Proper indexing strategies
                     - CTE usage and best practices
                     - EXPLAIN ANALYZE interpretation""",
                     
            "dbt": """You are a dbt expert. Focus on:
                     - Model materialization strategies
                     - Incremental model patterns
                     - Testing and documentation
                     - Macro development
                     - Following dbt best practices (staging, intermediate, marts)""",
                     
            "react": """You are a React expert. Focus on:
                      - Modern React patterns (hooks, functional components)
                      - Performance optimization
                      - State management best practices
                      - TypeScript integration""",
                      
            "python": """You are a Python expert. Focus on:
                       - Pythonic code patterns
                       - Type hints and modern Python features
                       - Performance optimization
                       - Testing with pytest"""
        }
        
        return prompts.get(domain, "")
    
    def _update_metrics(self, decision: RoutingDecision):
        """Update routing metrics for monitoring"""
        self.metrics["total_routes"] += 1
        self.metrics["tier_usage"][decision.model_tier] += 1
        
        # Update rolling average confidence
        n = self.metrics["total_routes"]
        prev_avg = self.metrics["avg_confidence"]
        self.metrics["avg_confidence"] = (prev_avg * (n-1) + decision.confidence) / n
        
        # Track costs
        self.metrics["total_cost"] += decision.estimated_cost
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics for monitoring"""
        return self.metrics
    
    async def should_escalate(self, 
                              query: str, 
                              current_response: str,
                              user_feedback: Optional[str] = None) -> bool:
        """
        Determine if we should escalate to a more powerful model
        based on response quality or user feedback
        
        Args:
            query: Original query
            current_response: Response from current model
            user_feedback: Optional user feedback indicating dissatisfaction
            
        Returns:
            True if should escalate to reasoning model
        """
        # Quick check for explicit escalation triggers
        if user_feedback and any(word in user_feedback.lower() 
                                 for word in ["wrong", "incorrect", "not what", "try again"]):
            logger.info("Escalating due to negative feedback")
            return True
            
        # Use router to re-evaluate with context about failure
        context = {
            "previous_response": current_response[:500],  # First 500 chars
            "user_feedback": user_feedback
        }
        
        new_decision = await self.router.route_query(query, context)
        
        # Escalate if router now suggests a higher tier
        return new_decision.model_tier == ModelTier.REASONING