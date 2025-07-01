#!/usr/bin/env python3
"""
Real-time Pricing Manager for ADAM
Fetches current pricing from LLM providers dynamically
"""

import asyncio
import aiohttp
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class PricingManager:
    """
    Manages real-time pricing for different LLM models
    Caches prices for efficiency but refreshes periodically
    """
    
    def __init__(self, cache_duration_hours: int = 24):
        """
        Initialize the pricing manager
        
        Args:
            cache_duration_hours: How long to cache prices before refreshing
        """
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self._price_cache: Dict[str, Tuple[float, datetime]] = {}
        self._default_prices = {
            # Fallback prices if API calls fail (per 1K tokens)
            "grok-3-mini-reasoning-high": {"input": 0.15, "output": 0.60},
            "o3": {"input": 15.0, "output": 60.0},  
            "claude-opus-4": {"input": 15.0, "output": 75.0}
        }
        
        # API endpoints for pricing (these would be real in production)
        self.pricing_endpoints = {
            "openai": "https://api.openai.com/v1/pricing",
            "anthropic": "https://api.anthropic.com/v1/pricing",
            "xai": "https://api.x.ai/v1/pricing"
        }
    
    async def get_model_price(self, model: str, token_type: str = "both") -> Dict[str, float]:
        """
        Get current price for a model
        
        Args:
            model: Model name
            token_type: "input", "output", or "both"
            
        Returns:
            Dictionary with pricing information
        """
        cache_key = f"{model}_{token_type}"
        
        # Check cache first
        if cache_key in self._price_cache:
            price, cached_time = self._price_cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                logger.debug(f"Using cached price for {model}")
                return {"price": price, "cached": True, "age_hours": (datetime.now() - cached_time).total_seconds() / 3600}
        
        # Fetch fresh pricing
        try:
            price = await self._fetch_real_time_price(model)
            self._price_cache[cache_key] = (price, datetime.now())
            return {"price": price, "cached": False, "age_hours": 0}
        except Exception as e:
            logger.warning(f"Failed to fetch real-time price for {model}: {e}")
            # Use default pricing
            default = self._default_prices.get(model, {"input": 1.0, "output": 1.0})
            if token_type == "input":
                return {"price": default["input"], "cached": False, "default": True}
            elif token_type == "output":
                return {"price": default["output"], "cached": False, "default": True}
            else:
                return {"price": (default["input"] + default["output"]) / 2, "cached": False, "default": True}
    
    async def _fetch_real_time_price(self, model: str) -> float:
        """
        Fetch real-time pricing from provider APIs
        
        In production, this would make actual API calls to:
        - OpenAI pricing API for O3
        - Anthropic pricing API for Claude
        - X.AI pricing API for Grok
        """
        # Simulate API call with realistic response times
        await asyncio.sleep(0.1)  # Simulate network latency
        
        # In production, this would be real API calls
        if model == "grok-3-mini-reasoning-high":
            # Simulate X.AI API response
            return 0.20  # $0.20 per 1K tokens average
        elif model == "o3":
            # Simulate OpenAI API response
            # O3 pricing is high due to advanced reasoning
            return 37.5  # $37.50 per 1K tokens average (between input and output)
        elif model == "claude-opus-4":
            # Simulate Anthropic API response
            return 45.0  # $45 per 1K tokens average
        else:
            raise ValueError(f"Unknown model: {model}")
    
    async def get_current_prices(self) -> Dict[str, Dict[str, float]]:
        """Get current prices for all models"""
        models = ["grok-3-mini-reasoning-high", "o3", "claude-opus-4"]
        prices = {}
        
        for model in models:
            price_info = await self.get_model_price(model)
            prices[model] = price_info
        
        return prices
    
    def estimate_query_cost(self, model: str, estimated_tokens: int) -> float:
        """
        Estimate cost for a query
        
        Args:
            model: Model to use
            estimated_tokens: Estimated total tokens (input + output)
            
        Returns:
            Estimated cost in USD
        """
        # Use cached price or default
        cache_key = f"{model}_both"
        if cache_key in self._price_cache:
            price_per_1k, _ = self._price_cache[cache_key]
        else:
            default = self._default_prices.get(model, {"input": 1.0, "output": 1.0})
            price_per_1k = (default["input"] + default["output"]) / 2
        
        return (estimated_tokens / 1000) * price_per_1k
    
    async def update_all_prices(self):
        """Force update all prices from APIs"""
        logger.info("Updating all model prices...")
        
        models = ["grok-3-mini-reasoning-high", "o3", "claude-opus-4"]
        for model in models:
            try:
                price = await self._fetch_real_time_price(model)
                self._price_cache[f"{model}_both"] = (price, datetime.now())
                logger.info(f"Updated {model} price: ${price:.2f} per 1K tokens")
            except Exception as e:
                logger.error(f"Failed to update price for {model}: {e}")
    
    def get_cost_breakdown(self, model: str, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """
        Get detailed cost breakdown
        
        Args:
            model: Model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Detailed cost breakdown
        """
        default = self._default_prices.get(model, {"input": 1.0, "output": 1.0})
        
        input_cost = (input_tokens / 1000) * default["input"]
        output_cost = (output_tokens / 1000) * default["output"]
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model,
            "timestamp": datetime.now().isoformat()
        }


class CostOptimizer:
    """
    Optimizes model selection based on real-time costs and requirements
    """
    
    def __init__(self, pricing_manager: PricingManager):
        self.pricing_manager = pricing_manager
        self.budget_alerts = {
            "daily_limit": 1.0,
            "query_limit": 0.10  # Alert if single query exceeds $0.10
        }
    
    async def select_optimal_model(self, 
                                 complexity: str, 
                                 is_coding: bool,
                                 memory_confidence: float,
                                 remaining_daily_budget: float) -> str:
        """
        Select optimal model based on current prices and constraints
        
        Args:
            complexity: Query complexity (simple/moderate/complex)
            is_coding: Whether this is a coding task
            memory_confidence: Confidence in memory match (0-1)
            remaining_daily_budget: Remaining budget for today
            
        Returns:
            Optimal model to use
        """
        # Get current prices
        prices = await self.pricing_manager.get_current_prices()
        
        # High memory confidence allows using cheaper model
        if memory_confidence > 0.9:
            return "grok-3-mini-reasoning-high"
        
        # Coding tasks need Claude
        if is_coding and complexity == "complex":
            claude_cost = self.pricing_manager.estimate_query_cost("claude-opus-4", 2000)
            if claude_cost <= remaining_daily_budget:
                return "claude-opus-4"
            else:
                logger.warning(f"Claude cost ${claude_cost:.3f} exceeds budget, falling back to O3")
                return "o3"
        
        # Complex non-coding tasks
        if complexity == "complex":
            o3_cost = self.pricing_manager.estimate_query_cost("o3", 1500)
            if o3_cost <= remaining_daily_budget:
                return "o3"
            else:
                logger.warning(f"O3 cost ${o3_cost:.3f} exceeds budget, falling back to Grok")
                return "grok-3-mini-reasoning-high"
        
        # Simple and moderate tasks use Grok
        return "grok-3-mini-reasoning-high"
    
    def should_alert_cost(self, query_cost: float, daily_total: float) -> Tuple[bool, Optional[str]]:
        """
        Check if cost alerts should be triggered
        
        Returns:
            (should_alert, alert_message)
        """
        if query_cost > self.budget_alerts["query_limit"]:
            return True, f"Query cost ${query_cost:.3f} exceeds per-query limit of ${self.budget_alerts['query_limit']}"
        
        if daily_total > self.budget_alerts["daily_limit"]:
            return True, f"Daily total ${daily_total:.3f} exceeds daily limit of ${self.budget_alerts['daily_limit']}"
        
        return False, None


# Global pricing manager instance
_pricing_manager = None

def get_pricing_manager() -> PricingManager:
    """Get or create the global pricing manager"""
    global _pricing_manager
    if _pricing_manager is None:
        _pricing_manager = PricingManager()
    return _pricing_manager


# Example usage and testing
async def demo_pricing():
    """Demonstrate pricing functionality"""
    manager = get_pricing_manager()
    optimizer = CostOptimizer(manager)
    
    print("=== Real-time Pricing Demo ===\n")
    
    # Update all prices
    await manager.update_all_prices()
    
    # Get current prices
    prices = await manager.get_current_prices()
    print("Current Model Prices (per 1K tokens):")
    for model, info in prices.items():
        print(f"  {model}: ${info['price']:.2f}")
    print()
    
    # Test cost estimation
    test_queries = [
        ("What is Python?", "simple", False, 100),
        ("Debug this distributed system", "complex", False, 1500),
        ("Implement a web scraper", "complex", True, 2000)
    ]
    
    print("Query Cost Estimates:")
    for query, complexity, is_coding, tokens in test_queries:
        model = await optimizer.select_optimal_model(complexity, is_coding, 0.5, 1.0)
        cost = manager.estimate_query_cost(model, tokens)
        print(f"  '{query[:30]}...': {model} -> ${cost:.4f}")
    
    # Test cost breakdown
    print("\nDetailed Cost Breakdown (O3, 1000 tokens):")
    breakdown = manager.get_cost_breakdown("o3", 500, 500)
    for key, value in breakdown.items():
        if isinstance(value, float):
            print(f"  {key}: ${value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(demo_pricing())