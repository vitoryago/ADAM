#!/usr/bin/env python3
"""
Modern LLM-based Query Router for ADAM
======================================

This module uses a lightweight LLM (like Claude Haiku or GPT-4-mini) to intelligently
route queries to the appropriate model based on complexity and requirements.

Instead of hardcoded rules, we let an LLM understand the query and decide:
1. Which model to use (reasoning, standard, or fast)
2. Whether specialized knowledge is needed (SQL, dbt, etc.)
3. How much context/memory to retrieve
"""

import json
import logging
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass
from enum import Enum
import os
from dotenv import load_dotenv

# For making LLM calls
import openai
from anthropic import Anthropic

load_dotenv()
logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Available model tiers with associated models"""
    REASONING = "grok-4-reasoning"  # Most expensive, highest capability
    STANDARD = "grok-4"             # Balanced performance
    FAST = "gpt-4o-mini"       # Cheapest, fastest (LOW latency)
    
    # Additional options for router
    ROUTER = "claude-3-5-haiku-20241022"  # Fast, cheap router
    ROUTER_BACKUP = "gpt-4o-mini-2024-07-18"  # Alternative router


@dataclass
class RoutingDecision:
    """
    Represents the routing decision made by the LLM router
    """
    model_tier: ModelTier
    reasoning: str
    confidence: float
    requires_memory: bool
    memory_depth: int  # How many memories to retrieve (0-20)
    specialized_domain: Optional[str] = None  # e.g., "sql", "dbt", "react"
    estimated_tokens: int = 0
    estimated_cost: float = 0.0
    

class LLMRouter:
    """
    Modern LLM-based query router that uses AI to make routing decisions
    """
    
    ROUTER_PROMPT = """You are a query routing expert for ADAM, an AI assistant. 
Analyze the user's query and decide which model tier to use.

Available tiers:
1. REASONING - Use for complex tasks requiring deep analysis, multi-step reasoning, 
   code architecture, debugging complex issues, or mathematical proofs
2. STANDARD - Use for moderate tasks like general coding, explanations, refactoring,
   data analysis, or technical discussions
3. FAST - Use for simple tasks like greetings, basic questions, syntax lookups,
   or straightforward responses

Also determine:
- requires_memory: Should we search previous conversations? (true/false)
- memory_depth: How many relevant memories to retrieve (0-20)
- specialized_domain: Is this about a specific domain? (sql/dbt/react/python/null)

Respond ONLY with a JSON object:
{
  "model_tier": "REASONING" | "STANDARD" | "FAST",
  "reasoning": "Brief explanation of why this tier was chosen",
  "confidence": 0.0-1.0,
  "requires_memory": true/false,
  "memory_depth": 0-20,
  "specialized_domain": "sql" | "dbt" | "react" | "python" | null
}

Query: {query}"""

    def __init__(self, use_anthropic: bool = True):
        """
        Initialize the router with chosen LLM provider
        
        Args:
            use_anthropic: Use Claude Haiku if True, else use GPT-4-mini
        """
        self.use_anthropic = use_anthropic
        
        if use_anthropic:
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.router_model = ModelTier.ROUTER.value
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.client = openai
            self.router_model = ModelTier.ROUTER_BACKUP.value
            
        # Cost tracking for router calls
        self.router_costs = {
            ModelTier.ROUTER.value: 0.00025,  # per 1K tokens
            ModelTier.ROUTER_BACKUP.value: 0.00015  # per 1K tokens
        }
        
    async def route_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """
        Route a query using LLM intelligence
        
        Args:
            query: The user's query to route
            context: Optional context about the conversation/project
            
        Returns:
            RoutingDecision with model selection and parameters
        """
        try:
            # Build the routing prompt
            prompt = self.ROUTER_PROMPT.format(query=query)
            
            # Add context if provided
            if context:
                prompt += f"\n\nContext: {json.dumps(context, indent=2)}"
            
            # Make the routing decision using LLM
            if self.use_anthropic:
                response = await self._route_with_anthropic(prompt)
            else:
                response = await self._route_with_openai(prompt)
            
            # Clean the response before parsing
            response = response.strip()
            
            # Remove code blocks if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            # Find JSON object in response
            if not response.startswith('{'):
                json_start = response.find('{')
                if json_start != -1:
                    response = response[json_start:]
                else:
                    raise ValueError(f"No JSON object found in response: {response}")
            
            # Find matching closing brace
            if response.startswith('{'):
                brace_count = 0
                end_pos = 0
                for i, char in enumerate(response):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                if end_pos > 0:
                    response = response[:end_pos]
                
            # Parse the JSON response
            decision_data = json.loads(response)
            
            # Map string to enum
            model_mapping = {
                "REASONING": ModelTier.REASONING,
                "STANDARD": ModelTier.STANDARD,
                "FAST": ModelTier.FAST
            }
            
            decision = RoutingDecision(
                model_tier=model_mapping[decision_data["model_tier"]],
                reasoning=decision_data["reasoning"],
                confidence=float(decision_data["confidence"]),
                requires_memory=decision_data["requires_memory"],
                memory_depth=int(decision_data["memory_depth"]),
                specialized_domain=decision_data.get("specialized_domain")
            )
            
            # Estimate costs
            decision.estimated_tokens = len(query.split()) * 2  # Rough estimate
            decision.estimated_cost = self._estimate_cost(decision, decision.estimated_tokens)
            
            logger.info(f"Routing decision: {decision.model_tier.value} "
                       f"(confidence: {decision.confidence:.2f})")
            
            return decision
            
        except Exception as e:
            logger.error(f"Router failed, falling back to rule-based: {e}")
            return self._fallback_routing(query)
    
    async def _route_with_anthropic(self, prompt: str) -> str:
        """Route using Claude Haiku"""
        response = self.client.messages.create(
            model=self.router_model,
            max_tokens=200,
            temperature=0.1,  # Low temperature for consistent routing
            messages=[{"role": "user", "content": prompt}],
            system="You are a query router. Always respond with valid JSON only, no additional text."
        )
        content = response.content[0].text.strip()
        
        # Extract JSON more robustly
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Clean up common issues
        content = content.strip()
        if not content.startswith('{'):
            # Find the first { and start from there
            json_start = content.find('{')
            if json_start != -1:
                content = content[json_start:]
        
        # Find matching closing brace
        if content.startswith('{'):
            brace_count = 0
            end_pos = 0
            for i, char in enumerate(content):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            if end_pos > 0:
                content = content[:end_pos]
        
        return content
    
    async def _route_with_openai(self, prompt: str) -> str:
        """Route using GPT-4-mini"""
        response = await self.client.chat.completions.create(
            model=self.router_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        return response.choices[0].message.content
    
    def _fallback_routing(self, query: str) -> RoutingDecision:
        """
        Fallback to simple rule-based routing if LLM fails
        """
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Simple heuristics
        if word_count < 10 and any(word in query_lower for word in ["hi", "hello", "thanks"]):
            return RoutingDecision(
                model_tier=ModelTier.FAST,
                reasoning="Simple greeting or acknowledgment",
                confidence=0.7,
                requires_memory=False,
                memory_depth=0
            )
        elif word_count > 50 or "complex" in query_lower or "architect" in query_lower:
            return RoutingDecision(
                model_tier=ModelTier.REASONING,
                reasoning="Long or complex query detected",
                confidence=0.6,
                requires_memory=True,
                memory_depth=10
            )
        else:
            return RoutingDecision(
                model_tier=ModelTier.STANDARD,
                reasoning="Standard query, using default tier",
                confidence=0.5,
                requires_memory=True,
                memory_depth=5
            )
    
    def _estimate_cost(self, decision: RoutingDecision, estimated_tokens: int) -> float:
        """
        Estimate the cost of the query based on routing decision
        """
        # Model costs per 1K tokens (example rates)
        model_costs = {
            ModelTier.REASONING: 0.015,
            ModelTier.STANDARD: 0.005,
            ModelTier.FAST: 0.0015
        }
        
        base_cost = (estimated_tokens / 1000) * model_costs[decision.model_tier]
        
        # Add memory retrieval cost if needed
        if decision.requires_memory:
            memory_cost = decision.memory_depth * 0.0001  # Small cost per memory
            base_cost += memory_cost
            
        return base_cost
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about routing decisions
        """
        # This would connect to a database in production
        return {
            "total_routes": 0,
            "tier_distribution": {
                "REASONING": 0,
                "STANDARD": 0,
                "FAST": 0
            },
            "average_confidence": 0.0,
            "total_cost": 0.0
        }


class SpecializedRouter(LLMRouter):
    """
    Extended router with specialized domain detection for SQL/dbt
    """
    
    SQL_DBT_PROMPT_ADDITION = """
Special attention for SQL/dbt queries:
- If query mentions: SELECT, JOIN, CREATE TABLE, indexes, CTEs -> specialized_domain: "sql"
- If query mentions: dbt, models, refs, sources, macros, incremental -> specialized_domain: "dbt"
- If query asks to convert SQL to dbt or optimize queries -> use REASONING tier
"""
    
    def __init__(self, use_anthropic: bool = True):
        super().__init__(use_anthropic)
        # Extend the router prompt
        self.ROUTER_PROMPT += self.SQL_DBT_PROMPT_ADDITION


# Example usage
async def main():
    """Example of using the modern LLM router"""
    router = SpecializedRouter(use_anthropic=True)
    
    test_queries = [
        "Hi there!",
        "Create a complex dbt incremental model with custom tests",
        "What's 2+2?",
        "Help me optimize this SQL query with window functions and CTEs"
    ]
    
    for query in test_queries:
        decision = await router.route_query(query)
        print(f"\nQuery: {query}")
        print(f"Decision: {decision.model_tier.value}")
        print(f"Reasoning: {decision.reasoning}")
        print(f"Domain: {decision.specialized_domain}")
        print(f"Memory needed: {decision.requires_memory} (depth: {decision.memory_depth})")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())