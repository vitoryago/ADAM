"""Tests for LLM layer imports and basic structure."""


def test_unified_client_importable():
    from adam.llm.client import UnifiedLLMClient
    assert UnifiedLLMClient is not None


def test_router_importable():
    from adam.llm.router import LLMRouter, RoutingDecision
    assert LLMRouter is not None
    assert RoutingDecision is not None


def test_router_fallback():
    """Test that rule-based fallback works without API keys."""
    from adam.llm.router import LLMRouter
    router = LLMRouter.__new__(LLMRouter)  # Skip __init__ (needs API keys)
    decision = router._fallback_routing("Hello, how are you?")
    assert decision is not None
    assert hasattr(decision, 'model_tier')


def test_router_fallback_greeting():
    """Test fallback routes greetings to FAST tier."""
    from adam.llm.router import LLMRouter, ModelTier
    router = LLMRouter.__new__(LLMRouter)
    decision = router._fallback_routing("hi there")
    assert decision.model_tier == ModelTier.FAST
    assert decision.requires_memory is False


def test_router_fallback_complex():
    """Test fallback routes complex queries to REASONING tier."""
    from adam.llm.router import LLMRouter, ModelTier
    router = LLMRouter.__new__(LLMRouter)
    decision = router._fallback_routing("implement a distributed caching algorithm with consistent hashing that handles node failures gracefully and scales to handle millions of requests")
    assert decision.model_tier == ModelTier.REASONING
    assert decision.requires_memory is True


def test_router_fallback_dbt():
    """Test fallback detects dbt domain."""
    from adam.llm.router import LLMRouter
    router = LLMRouter.__new__(LLMRouter)
    decision = router._fallback_routing("help me create a dbt staging model")
    assert decision.needs_dbt is True
    assert decision.specialized_domain == "dbt"


def test_routing_decision_dataclass():
    """Test RoutingDecision has all expected fields."""
    from adam.llm.router import RoutingDecision, ModelTier
    decision = RoutingDecision(
        model_tier=ModelTier.STANDARD,
        reasoning="test",
        confidence=0.8,
        requires_memory=True,
        memory_depth=5,
    )
    assert decision.model_tier == ModelTier.STANDARD
    assert decision.confidence == 0.8
    assert decision.needs_dbt is False
    assert decision.needs_sql is False
    assert decision.system_prompt_addon is None


def test_llm_init_exports():
    """Test that the llm package exports UnifiedLLMClient."""
    from adam.llm import UnifiedLLMClient
    assert UnifiedLLMClient is not None
