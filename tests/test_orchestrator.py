"""Tests for the multi-agent orchestrator."""

import pytest
from dataclasses import dataclass
from typing import Optional, Dict

from adam.agents.orchestrator import Orchestrator, OrchestrationPattern
from adam.agents.scratchpad import Scratchpad, ScratchpadEntry, AgentRole, EntryType
from adam.agents.synthesis import FallbackSynthesizer


# ---------------------------------------------------------------------------
# Mock LLM helpers
# ---------------------------------------------------------------------------

@dataclass
class MockLLMResponse:
    """Realistic mock response with the attributes Agent.think() reads."""
    content: str
    model: str = "mock-model-v1"
    cost: float = 0.01
    total_tokens: int = 100


class MockLLMClient:
    """Mock LLM client that returns canned responses sequentially.

    Each call to ``complete()`` pops the next response from the list.
    When the list is exhausted a default response is returned.
    """

    def __init__(self, responses: Optional[list] = None, default_content: str = "default response"):
        self.responses = list(responses or [])
        self.default_content = default_content
        self.call_count = 0
        self.calls = []

    async def complete(self, *, prompt, model=None, system_prompt=None,
                       temperature=0.7, max_tokens=4000, stream=False):
        self.call_count += 1
        self.calls.append({
            "prompt": prompt,
            "model": model,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        })

        if stream:
            return self._stream_response()

        if self.responses:
            return self.responses.pop(0)
        return MockLLMResponse(content=self.default_content)

    async def _stream_response(self):
        """Return an async generator that yields content word-by-word."""
        if self.responses:
            resp = self.responses.pop(0)
            content = resp.content
        else:
            content = self.default_content

        for word in content.split():
            yield word + " "


def _make_sequential_responses():
    """Create four responses for the sequential pipeline (Reasoner->Coder->Critic->Synthesizer)."""
    return [
        MockLLMResponse(
            content="Plan: 1. Parse input 2. Process 3. Return result",
            model="reasoning-model",
            cost=0.02,
            total_tokens=150,
        ),
        MockLLMResponse(
            content="def process(data):\n    return sorted(data)",
            model="coding-model",
            cost=0.03,
            total_tokens=250,
        ),
        MockLLMResponse(
            content="Correctness: OK. Edge cases handled. Confidence: HIGH.",
            model="critic-model",
            cost=0.02,
            total_tokens=120,
        ),
        MockLLMResponse(
            content="Here is the complete solution with edge case handling.",
            model="synth-model",
            cost=0.01,
            total_tokens=180,
        ),
    ]


def _make_debate_responses():
    """Create three responses for the debate pipeline (DebaterA, DebaterB, Reconciler)."""
    return [
        MockLLMResponse(
            content="Perspective A: Use approach X for its performance benefits.",
            model="debater-a-model",
            cost=0.01,
            total_tokens=100,
        ),
        MockLLMResponse(
            content="Perspective B: Use approach Y for its simplicity.",
            model="debater-b-model",
            cost=0.01,
            total_tokens=100,
        ),
        MockLLMResponse(
            content="After weighing both perspectives, the best approach combines X and Y.",
            model="reconciler-model",
            cost=0.01,
            total_tokens=150,
        ),
    ]


# ---------------------------------------------------------------------------
# Tests: should_use_multi_agent
# ---------------------------------------------------------------------------

class TestShouldUseMultiAgent:
    """Tests for the complexity gate that decides when multi-agent is needed."""

    def test_should_use_multi_agent_complex(self):
        """'Design a microservices architecture' should trigger multi-agent."""
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.should_use_multi_agent("Design a microservices architecture") is True

    def test_should_not_use_multi_agent_simple(self):
        """'What is Python?' is too simple for multi-agent."""
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.should_use_multi_agent("What is Python?") is False

    def test_complexity_hint_complex_forces_true(self):
        """When caller says complexity='complex', always return True."""
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.should_use_multi_agent("What is Python?", complexity="complex") is True

    def test_implement_triggers_multi_agent(self):
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.should_use_multi_agent("Implement a REST API with authentication") is True

    def test_debug_triggers_multi_agent(self):
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.should_use_multi_agent("Debug this memory leak in the server") is True

    def test_compare_triggers_multi_agent(self):
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.should_use_multi_agent("Compare React vs Vue for large SPAs") is True

    def test_architect_triggers_multi_agent(self):
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.should_use_multi_agent("Architect a scalable data pipeline") is True

    def test_simple_greeting_does_not_trigger(self):
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.should_use_multi_agent("Hello, how are you?") is False

    def test_simple_fact_does_not_trigger(self):
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.should_use_multi_agent("What year was Python created?") is False


# ---------------------------------------------------------------------------
# Tests: select_pattern
# ---------------------------------------------------------------------------

class TestSelectPattern:
    """Tests for pattern selection logic."""

    def test_select_pattern_sequential(self):
        """'Implement a REST API' should select SEQUENTIAL."""
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.select_pattern("Implement a REST API") == OrchestrationPattern.SEQUENTIAL

    def test_select_pattern_debate(self):
        """'Compare React vs Vue pros and cons' should select DEBATE."""
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.select_pattern("Compare React vs Vue pros and cons") == OrchestrationPattern.DEBATE

    def test_versus_triggers_debate(self):
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.select_pattern("PostgreSQL versus MongoDB for this use case") == OrchestrationPattern.DEBATE

    def test_which_is_better_triggers_debate(self):
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.select_pattern("Which is better: REST or GraphQL?") == OrchestrationPattern.DEBATE

    def test_tradeoffs_triggers_debate(self):
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.select_pattern("What are the trade-offs of microservices?") == OrchestrationPattern.DEBATE

    def test_design_triggers_sequential(self):
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.select_pattern("Design a caching layer") == OrchestrationPattern.SEQUENTIAL

    def test_build_triggers_sequential(self):
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.select_pattern("Build a web scraper") == OrchestrationPattern.SEQUENTIAL


# ---------------------------------------------------------------------------
# Tests: orchestrate (non-streaming)
# ---------------------------------------------------------------------------

class TestOrchestrate:
    """Tests for the main orchestrate() method."""

    @pytest.mark.asyncio
    async def test_orchestrate_sequential(self):
        """Sequential orchestration produces a result with pattern='sequential'."""
        client = MockLLMClient(responses=_make_sequential_responses())
        orch = Orchestrator(client)

        result = await orch.orchestrate("Implement a REST API with auth")

        assert result["pattern"] == "sequential"
        assert len(result["agents_used"]) > 0
        assert result["total_cost"] > 0

    @pytest.mark.asyncio
    async def test_orchestrate_debate(self):
        """Debate orchestration produces a result with pattern='debate'."""
        client = MockLLMClient(responses=_make_debate_responses())
        orch = Orchestrator(client)

        result = await orch.orchestrate("Compare React vs Vue for large apps")

        assert result["pattern"] == "debate"
        assert len(result["agents_used"]) > 0

    @pytest.mark.asyncio
    async def test_orchestrate_returns_response(self):
        """The result has a 'response' key with actual content."""
        client = MockLLMClient(responses=_make_sequential_responses())
        orch = Orchestrator(client)

        result = await orch.orchestrate("Implement a sorting algorithm")

        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    @pytest.mark.asyncio
    async def test_orchestrate_result_structure(self):
        """Verify all expected keys are present in the result."""
        client = MockLLMClient(responses=_make_sequential_responses())
        orch = Orchestrator(client)

        result = await orch.orchestrate("Design a logging system")

        expected_keys = {"response", "pattern", "total_cost", "agents_used", "scratchpad"}
        assert set(result.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_orchestrate_scratchpad_in_result(self):
        """The scratchpad dict should have the expected structure."""
        client = MockLLMClient(responses=_make_sequential_responses())
        orch = Orchestrator(client)

        result = await orch.orchestrate("Build a task queue")

        pad = result["scratchpad"]
        assert "query" in pad
        assert "entries" in pad
        assert "total_cost" in pad
        assert pad["query"] == "Build a task queue"


# ---------------------------------------------------------------------------
# Tests: orchestrate_stream
# ---------------------------------------------------------------------------

class TestOrchestrateStream:
    """Tests for the streaming orchestrate method."""

    @pytest.mark.asyncio
    async def test_orchestrate_stream_events(self):
        """Streaming yields an orchestration_start event followed by pattern events."""
        client = MockLLMClient(responses=_make_sequential_responses())
        orch = Orchestrator(client)

        events = []
        async for event in orch.orchestrate_stream("Implement a cache layer"):
            events.append(event)

        # First event is orchestration_start
        assert events[0]["type"] == "orchestration_start"
        assert events[0]["pattern"] == "sequential"

        # Last event from the pipeline is "done"
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_orchestrate_stream_debate_pattern(self):
        """Streaming debate orchestration starts with debate pattern."""
        client = MockLLMClient(responses=_make_debate_responses())
        orch = Orchestrator(client)

        events = []
        async for event in orch.orchestrate_stream("Compare Python vs Go"):
            events.append(event)

        assert events[0]["type"] == "orchestration_start"
        assert events[0]["pattern"] == "debate"

        # Should have a debate_start event
        types = [e["type"] for e in events]
        assert "debate_start" in types

    @pytest.mark.asyncio
    async def test_orchestrate_stream_has_agent_events(self):
        """Streaming emits agent_start, agent_chunk, and agent_done events."""
        client = MockLLMClient(responses=_make_sequential_responses())
        orch = Orchestrator(client)

        types = set()
        async for event in orch.orchestrate_stream("Build a REST API"):
            types.add(event["type"])

        assert "orchestration_start" in types
        assert "agent_start" in types
        assert "agent_chunk" in types
        assert "agent_done" in types
        assert "done" in types


# ---------------------------------------------------------------------------
# Tests: budget enforcement
# ---------------------------------------------------------------------------

class TestBudgetPassedToScratchpad:
    """Verify that budget is properly threaded through to the scratchpad."""

    @pytest.mark.asyncio
    async def test_budget_passed_to_scratchpad(self):
        """The scratchpad receives the budget specified in orchestrate()."""
        client = MockLLMClient(responses=_make_sequential_responses())
        orch = Orchestrator(client)

        result = await orch.orchestrate("Design a system", budget=0.10)

        # The scratchpad dict should reflect the budget
        assert result["scratchpad"]["budget"] == 0.10

    @pytest.mark.asyncio
    async def test_small_budget_limits_agents(self):
        """A very small budget should result in fewer agents running."""
        # With budget=0.02, only the first agent can run (cost=0.02)
        responses = [
            MockLLMResponse(content="plan", cost=0.02, total_tokens=50),
            MockLLMResponse(content="code", cost=0.03, total_tokens=100),
            MockLLMResponse(content="critique", cost=0.02, total_tokens=80),
            MockLLMResponse(content="synthesis", cost=0.01, total_tokens=60),
        ]
        client = MockLLMClient(responses=responses)
        orch = Orchestrator(client)

        result = await orch.orchestrate("Build something", budget=0.02)

        # Only 1 agent should have run (budget exhausted after first)
        assert len(result["agents_used"]) == 1

    @pytest.mark.asyncio
    async def test_generous_budget_runs_all_agents(self):
        """A large budget lets all agents run."""
        client = MockLLMClient(responses=_make_sequential_responses())
        orch = Orchestrator(client)

        result = await orch.orchestrate("Build a feature", budget=10.0)

        assert len(result["agents_used"]) == 4


# ---------------------------------------------------------------------------
# Tests: fallback synthesis
# ---------------------------------------------------------------------------

class TestFallbackWhenNoSynthesis:
    """Verify FallbackSynthesizer is used when synthesis entry is missing."""

    @pytest.mark.asyncio
    async def test_fallback_when_no_synthesis(self):
        """When budget runs out before synthesizer, FallbackSynthesizer is used."""
        # Budget allows only Reasoner + Coder (0.02 + 0.02 = 0.04)
        responses = [
            MockLLMResponse(content="Step 1: Parse input. Step 2: Sort.", cost=0.02, total_tokens=50),
            MockLLMResponse(content="def sort(arr): return sorted(arr)", cost=0.02, total_tokens=100),
            MockLLMResponse(content="Review: looks good", cost=0.02, total_tokens=80),
            MockLLMResponse(content="Final synthesis", cost=0.01, total_tokens=60),
        ]
        client = MockLLMClient(responses=responses)
        orch = Orchestrator(client)

        result = await orch.orchestrate("Implement sorting", budget=0.04)

        # No synthesis entry because budget ran out
        # FallbackSynthesizer should produce a response from available entries
        assert "response" in result
        assert len(result["response"]) > 0
        # The response should contain content from the entries that did run
        assert "Step 1" in result["response"] or "sort" in result["response"]

    @pytest.mark.asyncio
    async def test_fallback_uses_available_entries(self):
        """FallbackSynthesizer combines plan and code when synthesis is missing."""
        # Only 1 agent runs
        responses = [
            MockLLMResponse(
                content="Plan: validate input, transform data, return result",
                cost=0.05,
                total_tokens=100,
            ),
        ]
        client = MockLLMClient(responses=responses)
        orch = Orchestrator(client)

        result = await orch.orchestrate("Build a data pipeline", budget=0.05)

        # Response should exist and contain the plan content
        assert "response" in result
        assert "Plan:" in result["response"] or "validate" in result["response"]


# ---------------------------------------------------------------------------
# Tests: memory service (optional dependency)
# ---------------------------------------------------------------------------

class TestMemoryService:
    """Verify memory_service is stored correctly."""

    def test_memory_service_stored(self):
        client = MockLLMClient()
        memory = object()  # placeholder
        orch = Orchestrator(client, memory_service=memory)
        assert orch.memory_service is memory

    def test_memory_service_defaults_to_none(self):
        client = MockLLMClient()
        orch = Orchestrator(client)
        assert orch.memory_service is None
