"""Tests for multi-agent integration into the chat pipeline.

Verifies that:
- The orchestrator correctly identifies complex vs simple queries.
- LLMService exposes the generate_with_agents method.
- The messages router correctly detects multi-agent triggers.
- Force-trigger phrases ("deep think", "multi-agent", "reason deeply") work.
- Simple queries are NOT sent to multi-agent.
- The API health endpoint is reachable (sanity check for app wiring).
"""

import pytest
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Mock helpers (reused from test_orchestrator.py pattern)
# ---------------------------------------------------------------------------

@dataclass
class MockLLMResponse:
    content: str
    model: str = "mock-model"
    cost: float = 0.01
    total_tokens: int = 100


class MockLLMClient:
    def __init__(self, responses=None, default_content="mock response"):
        self.responses = list(responses or [])
        self.default_content = default_content
        self.call_count = 0

    async def complete(self, *, prompt, model=None, system_prompt=None,
                       temperature=0.7, max_tokens=4000, stream=False):
        self.call_count += 1
        if self.responses:
            return self.responses.pop(0)
        return MockLLMResponse(content=self.default_content)


# ---------------------------------------------------------------------------
# Tests: multi-agent detection triggers
# ---------------------------------------------------------------------------

class TestMultiAgentDetectionTriggers:
    """Orchestrator correctly identifies complex queries."""

    def test_design_microservices_triggers(self):
        from adam.agents.orchestrator import Orchestrator
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.should_use_multi_agent("Design a microservices architecture") is True

    def test_simple_question_does_not_trigger(self):
        from adam.agents.orchestrator import Orchestrator
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.should_use_multi_agent("What is Python?") is False

    def test_complexity_hint_forces_true(self):
        from adam.agents.orchestrator import Orchestrator
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.should_use_multi_agent("What is Python?", "complex") is True

    def test_complexity_hint_simple_with_trigger_word(self):
        from adam.agents.orchestrator import Orchestrator
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.should_use_multi_agent("Implement a REST API", "simple") is True

    def test_greeting_does_not_trigger(self):
        from adam.agents.orchestrator import Orchestrator
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.should_use_multi_agent("Hello there!") is False


# ---------------------------------------------------------------------------
# Tests: LLMService has generate_with_agents method
# ---------------------------------------------------------------------------

class TestLLMServiceAgentMethod:
    """LLMService has generate_with_agents method."""

    def test_method_exists(self):
        from adam.services.llm_service import LLMService
        assert hasattr(LLMService, "generate_with_agents")

    def test_method_is_coroutine(self):
        import inspect
        from adam.services.llm_service import LLMService
        assert inspect.iscoroutinefunction(LLMService.generate_with_agents)

    @pytest.mark.asyncio
    async def test_generate_with_agents_returns_llm_response(self):
        """The method produces an LLMResponse with multi_agent metadata."""
        from adam.services.llm_service import LLMService, LLMResponse

        service = LLMService.__new__(LLMService)
        # Wire up a mock client directly
        service.llm_client = MockLLMClient(responses=[
            MockLLMResponse(content="Plan: step 1, step 2", cost=0.02, total_tokens=80),
            MockLLMResponse(content="def foo(): pass", cost=0.03, total_tokens=120),
            MockLLMResponse(content="Looks good", cost=0.01, total_tokens=60),
            MockLLMResponse(content="Final answer with all steps", cost=0.01, total_tokens=100),
        ])
        service.memory_service = None

        result = await service.generate_with_agents("Implement a sorting algorithm")

        assert isinstance(result, LLMResponse)
        assert result.metadata["multi_agent"] is True
        assert result.metadata["pattern"] in ("sequential", "debate")
        assert "multi-agent" in result.model_used
        assert result.cost > 0


# ---------------------------------------------------------------------------
# Tests: _should_use_multi_agent helper in messages router
# ---------------------------------------------------------------------------

class TestMessageRouterMultiAgentDetection:
    """The _should_use_multi_agent helper in messages.py works correctly."""

    def test_force_trigger_deep_think(self):
        from adam.api.routers.messages import _should_use_multi_agent
        assert _should_use_multi_agent("Please deep think about this problem") is True

    def test_force_trigger_multi_agent(self):
        from adam.api.routers.messages import _should_use_multi_agent
        assert _should_use_multi_agent("Use multi-agent to solve this") is True

    def test_force_trigger_reason_deeply(self):
        from adam.api.routers.messages import _should_use_multi_agent
        assert _should_use_multi_agent("I need you to reason deeply about caching") is True

    def test_complex_query_triggers(self):
        from adam.api.routers.messages import _should_use_multi_agent
        assert _should_use_multi_agent("Design a distributed task queue system") is True

    def test_simple_query_does_not_trigger(self):
        from adam.api.routers.messages import _should_use_multi_agent
        assert _should_use_multi_agent("What time is it?") is False

    def test_hello_does_not_trigger(self):
        from adam.api.routers.messages import _should_use_multi_agent
        assert _should_use_multi_agent("Hello, how are you?") is False


# ---------------------------------------------------------------------------
# Tests: API health endpoint (verifies app wiring is intact)
# ---------------------------------------------------------------------------

class TestAPIEndpointExists:
    """The messages API can handle multi-agent queries (app-level sanity)."""

    def test_health_endpoint(self):
        from adam.api.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_api_health_endpoint(self):
        from adam.api.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/health")
        assert resp.status_code == 200
