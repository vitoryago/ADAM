"""
Unified LLM Router for ADAM
============================

Consolidates routing logic from:
- llm_router.py (LLMRouter, SpecializedRouter, RoutingDecision, ModelTier)
- fast_routing_service.py (caching, keyword-based fast pre-filter)
- intelligent_routing_service.py (specialized prompts, metrics, escalation)

Uses a lightweight LLM (Claude Haiku or GPT-4-mini) to intelligently route
queries to the appropriate model tier. Falls back to rule-based routing
when LLM APIs are unavailable.
"""

import json
import logging
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Available model tiers with associated models"""
    REASONING = "grok-4.20-0309-reasoning"        # Reasoning for complex tasks and code
    STANDARD = "grok-4.20-0309-non-reasoning"     # Non-reasoning for intermediate tasks
    FAST = "grok-4.20-0309-non-reasoning"         # Non-reasoning for basic queries

    # Additional options for router
    ROUTER = "claude-haiku-4-5"               # Fast, cheap router
    ROUTER_BACKUP = "gpt-4o-mini-2024-07-18"  # Alternative router


@dataclass
class RoutingDecision:
    """
    Represents the routing decision made by the LLM router.

    Attributes:
        model_tier: The tier/model selected for the query
        reasoning: Human-readable explanation of why this tier was chosen
        confidence: Confidence score (0.0 - 1.0)
        requires_memory: Whether memory retrieval is needed
        memory_depth: How many memories to retrieve (0-20)
        specialized_domain: Detected domain (e.g., "sql", "dbt", "react", "python")
        estimated_tokens: Rough token estimate for the query
        estimated_cost: Estimated cost for the query
        needs_dbt: Whether dbt knowledge is needed
        needs_sql: Whether SQL knowledge is needed
        system_prompt_addon: Specialized system prompt for the domain
    """
    model_tier: ModelTier
    reasoning: str
    confidence: float
    requires_memory: bool
    memory_depth: int  # How many memories to retrieve (0-20)
    specialized_domain: Optional[str] = None  # e.g., "sql", "dbt", "react"
    estimated_tokens: int = 0
    estimated_cost: float = 0.0
    needs_dbt: bool = False
    needs_sql: bool = False
    system_prompt_addon: Optional[str] = None


# Specialized domain system prompts (from intelligent_routing_service.py)
DOMAIN_PROMPTS = {
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


class LLMRouter:
    """
    Unified LLM-based query router with caching, rule-based fallback,
    specialized domain detection, and metrics tracking.

    Merges functionality from:
    - LLMRouter / SpecializedRouter (LLM-based routing with Anthropic/OpenAI)
    - FastRoutingService (query hash caching, keyword-based pre-filter)
    - IntelligentRoutingService (model mapping, metrics, escalation, domain prompts)
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

Special attention for SQL/dbt queries:
- If query mentions: SELECT, JOIN, CREATE TABLE, indexes, CTEs -> specialized_domain: "sql"
- If query mentions: dbt, models, refs, sources, macros, incremental -> specialized_domain: "dbt"
- If query asks to convert SQL to dbt or optimize queries -> use REASONING tier

Also determine:
- requires_memory: Should we search previous conversations? (true/false)
- memory_depth: How many relevant memories to retrieve (0-20)
- specialized_domain: Is this about a specific domain? (sql/dbt/react/python/null)

Respond ONLY with a JSON object:
{{
  "model_tier": "REASONING" | "STANDARD" | "FAST",
  "reasoning": "Brief explanation of why this tier was chosen",
  "confidence": 0.0-1.0,
  "requires_memory": true/false,
  "memory_depth": 0-20,
  "specialized_domain": "sql" | "dbt" | "react" | "python" | null
}}

Query: {query}"""

    # Model tier to actual model name mapping
    MODEL_MAPPING = {
        ModelTier.REASONING: "grok-4.20-0309-reasoning",
        ModelTier.STANDARD: "grok-4.20-0309-non-reasoning",
        ModelTier.FAST: "grok-4.20-0309-non-reasoning",
    }

    def __init__(self, use_anthropic: bool = True):
        """
        Initialize the router with chosen LLM provider.

        Args:
            use_anthropic: Use Claude Haiku if True, else use GPT-4-mini
        """
        self.use_anthropic = use_anthropic
        self.client = None
        self.router_model = None

        # Try to initialize the LLM client for routing
        try:
            if use_anthropic:
                from anthropic import Anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.client = Anthropic(api_key=api_key)
                    self.router_model = ModelTier.ROUTER.value
            else:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    openai.api_key = api_key
                    self.client = openai
                    self.router_model = ModelTier.ROUTER_BACKUP.value
        except ImportError:
            logger.warning("Router LLM SDK not available, will use rule-based fallback")
        except Exception as e:
            logger.warning(f"Failed to initialize router LLM client: {e}")

        # Router call cost tracking
        self.router_costs = {
            ModelTier.ROUTER.value: 0.00025,        # per 1K tokens
            ModelTier.ROUTER_BACKUP.value: 0.00015,  # per 1K tokens
        }

        # Query hash cache (from FastRoutingService)
        self._routing_cache: Dict[str, RoutingDecision] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Routing metrics (from IntelligentRoutingService)
        self.metrics = {
            "total_routes": 0,
            "tier_usage": {tier: 0 for tier in ModelTier},
            "avg_confidence": 0.0,
            "total_cost": 0.0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Route a query using LLM intelligence with caching.

        Checks the cache first, then tries the LLM router, and falls back
        to rule-based routing on failure.

        Args:
            query: The user's query to route
            context: Optional context (project_id, conversation_id, etc.)

        Returns:
            RoutingDecision with model selection and parameters
        """
        # Check cache first (from FastRoutingService)
        query_hash = self._hash_query(query)
        if query_hash in self._routing_cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit for query: '{query[:50]}...'")
            return self._routing_cache[query_hash]

        self._cache_misses += 1

        # Try LLM-based routing
        decision = await self._route_with_llm(query, context)

        # Enrich decision with domain prompts (from IntelligentRoutingService)
        if decision.specialized_domain:
            decision.system_prompt_addon = DOMAIN_PROMPTS.get(
                decision.specialized_domain, ""
            )

        # Detect knowledge needs via keywords
        query_lower = query.lower()
        decision.needs_dbt = any(
            w in query_lower
            for w in ["dbt", "model", "transformation", "staging", "mart"]
        )
        decision.needs_sql = any(
            w in query_lower
            for w in ["sql", "query", "select", "join", "table", "database"]
        )

        # Cache the decision
        self._routing_cache[query_hash] = decision

        # Update metrics
        self._update_metrics(decision)

        return decision

    def route_and_configure(
        self,
        decision: RoutingDecision,
    ) -> Dict[str, Any]:
        """
        Build a configuration dict from a RoutingDecision for downstream use.

        Args:
            decision: A RoutingDecision from route_query()

        Returns:
            Configuration dict with model, memory config, domain info, etc.
        """
        config = {
            "model": self.MODEL_MAPPING.get(decision.model_tier, decision.model_tier.value),
            "reasoning": decision.reasoning,
            "confidence": decision.confidence,
            "memory_config": {
                "enabled": decision.requires_memory,
                "depth": decision.memory_depth,
                "include_cross_project": decision.memory_depth > 10,
            },
            "specialized_domain": decision.specialized_domain,
            "estimated_cost": decision.estimated_cost,
            "needs_dbt": decision.needs_dbt,
            "needs_sql": decision.needs_sql,
        }
        if decision.system_prompt_addon:
            config["system_prompt_addon"] = decision.system_prompt_addon
        return config

    # ------------------------------------------------------------------
    # LLM-based routing
    # ------------------------------------------------------------------

    async def _route_with_llm(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Attempt LLM-based routing; fall back on failure."""
        if self.client is None:
            logger.debug("No router LLM client available, using fallback")
            return self._fallback_routing(query)

        try:
            prompt = self.ROUTER_PROMPT.format(query=query)
            if context:
                prompt += f"\n\nContext: {json.dumps(context, indent=2)}"

            if self.use_anthropic:
                response_text = await self._call_anthropic(prompt)
            else:
                response_text = await self._call_openai(prompt)

            response_text = self._extract_json(response_text)
            decision_data = json.loads(response_text)

            model_mapping = {
                "REASONING": ModelTier.REASONING,
                "STANDARD": ModelTier.STANDARD,
                "FAST": ModelTier.FAST,
            }

            decision = RoutingDecision(
                model_tier=model_mapping[decision_data["model_tier"]],
                reasoning=decision_data["reasoning"],
                confidence=float(decision_data["confidence"]),
                requires_memory=decision_data["requires_memory"],
                memory_depth=int(decision_data["memory_depth"]),
                specialized_domain=decision_data.get("specialized_domain"),
            )

            # Estimate costs
            decision.estimated_tokens = len(query.split()) * 2
            decision.estimated_cost = self._estimate_cost(decision, decision.estimated_tokens)

            logger.info(
                f"Routing decision: {decision.model_tier.value} "
                f"(confidence: {decision.confidence:.2f})"
            )
            return decision

        except Exception as e:
            logger.error(f"Router failed, falling back to rule-based: {e}")
            return self._fallback_routing(query)

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Claude Haiku for routing."""
        response = self.client.messages.create(
            model=self.router_model,
            max_tokens=200,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
            system="You are a query router. Always respond with valid JSON only, no additional text.",
        )
        return response.content[0].text.strip()

    async def _call_openai(self, prompt: str) -> str:
        """Call GPT-4-mini for routing."""
        response = await self.client.chat.completions.create(
            model=self.router_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Rule-based fallback (merged from llm_router + fast_routing_service)
    # ------------------------------------------------------------------

    def _fallback_routing(self, query: str) -> RoutingDecision:
        """
        Rule-based routing fallback when LLM router is unavailable.

        Merges heuristics from the original LLMRouter._fallback_routing()
        and FastRoutingService._fallback_routing() for comprehensive
        keyword-based detection.
        """
        query_lower = query.lower()
        word_count = len(query.split())

        # --- Tier detection (combined rules) ---

        # Code generation / debugging -> REASONING
        if any(
            w in query_lower
            for w in [
                "write code", "create function", "implement", "debug",
                "fix", "algorithm", "develop", "architect",
            ]
        ):
            tier = ModelTier.REASONING
            reasoning = "Code generation or debugging detected"
            confidence = 0.7
            requires_memory = True
            memory_depth = 10

        # Very long or explicitly complex queries -> REASONING
        elif word_count > 50 or "complex" in query_lower:
            tier = ModelTier.REASONING
            reasoning = "Long or complex query detected"
            confidence = 0.6
            requires_memory = True
            memory_depth = 10

        # Technical / data / dbt / sql work -> STANDARD
        elif any(
            w in query_lower
            for w in [
                "dbt", "pdt", "sql", "model", "table", "query",
                "convert", "document", "explain", "analyze", "help me",
            ]
        ):
            tier = ModelTier.STANDARD
            reasoning = "Technical or data-related query"
            confidence = 0.6
            requires_memory = True
            memory_depth = 5

        # Short greetings / acknowledgments -> FAST
        elif word_count < 10 and any(
            w in query_lower
            for w in ["hi", "hello", "thanks", "ok", "yes", "no"]
        ):
            tier = ModelTier.FAST
            reasoning = "Simple greeting or acknowledgment"
            confidence = 0.7
            requires_memory = False
            memory_depth = 0

        # Default -> STANDARD
        else:
            tier = ModelTier.STANDARD
            reasoning = "Standard query, using default tier"
            confidence = 0.5
            requires_memory = True
            memory_depth = 5

        # --- Domain detection (from FastRoutingService) ---
        needs_dbt = any(
            w in query_lower
            for w in ["dbt", "model", "transformation", "staging", "mart"]
        )
        needs_sql = any(
            w in query_lower
            for w in ["sql", "query", "select", "join", "table", "database"]
        )

        specialized_domain = None
        if needs_dbt:
            specialized_domain = "dbt"
        elif needs_sql:
            specialized_domain = "sql"

        system_prompt_addon = DOMAIN_PROMPTS.get(specialized_domain, None)

        return RoutingDecision(
            model_tier=tier,
            reasoning=reasoning,
            confidence=confidence,
            requires_memory=requires_memory,
            memory_depth=memory_depth,
            specialized_domain=specialized_domain,
            needs_dbt=needs_dbt,
            needs_sql=needs_sql,
            system_prompt_addon=system_prompt_addon,
        )

    # ------------------------------------------------------------------
    # Escalation (from IntelligentRoutingService)
    # ------------------------------------------------------------------

    async def should_escalate(
        self,
        query: str,
        current_response: str,
        user_feedback: Optional[str] = None,
    ) -> bool:
        """
        Determine if we should escalate to a more powerful model
        based on response quality or user feedback.

        Args:
            query: Original query
            current_response: Response from current model
            user_feedback: Optional user feedback indicating dissatisfaction

        Returns:
            True if should escalate to REASONING tier
        """
        if user_feedback and any(
            word in user_feedback.lower()
            for word in ["wrong", "incorrect", "not what", "try again"]
        ):
            logger.info("Escalating due to negative feedback")
            return True

        # Re-evaluate with failure context
        context = {
            "previous_response": current_response[:500],
            "user_feedback": user_feedback,
        }
        new_decision = await self.route_query(query, context)
        return new_decision.model_tier == ModelTier.REASONING

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_query(query: str) -> str:
        """Create a normalized hash of the query for caching."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract a JSON object from potentially noisy LLM output."""
        text = text.strip()

        # Remove code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Find the opening brace
        if not text.startswith("{"):
            json_start = text.find("{")
            if json_start == -1:
                raise ValueError(f"No JSON object found in response: {text}")
            text = text[json_start:]

        # Find the matching closing brace
        brace_count = 0
        end_pos = 0
        for i, char in enumerate(text):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        if end_pos > 0:
            text = text[:end_pos]

        return text

    def _estimate_cost(self, decision: RoutingDecision, estimated_tokens: int) -> float:
        """Estimate the cost of the query based on routing decision."""
        model_costs = {
            ModelTier.REASONING: 0.015,
            ModelTier.STANDARD: 0.005,
            ModelTier.FAST: 0.0015,
        }
        base_cost = (estimated_tokens / 1000) * model_costs.get(decision.model_tier, 0.005)

        if decision.requires_memory:
            base_cost += decision.memory_depth * 0.0001
        return base_cost

    def _update_metrics(self, decision: RoutingDecision):
        """Update routing metrics for monitoring."""
        self.metrics["total_routes"] += 1
        self.metrics["tier_usage"][decision.model_tier] += 1

        n = self.metrics["total_routes"]
        prev_avg = self.metrics["avg_confidence"]
        self.metrics["avg_confidence"] = (prev_avg * (n - 1) + decision.confidence) / n
        self.metrics["total_cost"] += decision.estimated_cost

    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics for monitoring."""
        return self.metrics

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._routing_cache),
            "hit_rate": hit_rate,
        }

    def clear_cache(self):
        """Clear the routing cache."""
        self._routing_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Routing cache cleared")

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get combined routing and cache statistics."""
        stats = self.get_metrics()
        stats["cache"] = self.get_cache_stats()
        return stats
