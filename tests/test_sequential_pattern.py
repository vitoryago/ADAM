"""Tests for the sequential pipeline pattern (Reasoner -> Coder -> Critic -> Synthesizer)."""

import pytest
import asyncio
from dataclasses import dataclass
from typing import Optional

from adam.agents.scratchpad import Scratchpad, AgentRole, EntryType
from adam.agents.patterns.sequential import (
    run_sequential_pipeline,
    stream_sequential_pipeline,
)


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
    """Mock LLM client that returns pre-configured responses per call.

    Each call to ``complete()`` returns the next response in *responses*.
    If *responses* is exhausted, a default response is returned.
    """

    def __init__(self, responses: Optional[list] = None, default_content: str = "default response"):
        self.responses = list(responses or [])
        self.default_content = default_content
        self.call_count = 0
        self.calls = []  # record every call for assertions

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


class FailingStreamClient(MockLLMClient):
    """Client whose streaming always fails, but non-streaming works."""

    async def complete(self, *, prompt, model=None, system_prompt=None,
                       temperature=0.7, max_tokens=4000, stream=False):
        self.call_count += 1
        self.calls.append({
            "prompt": prompt,
            "model": model,
            "stream": stream,
        })

        if stream:
            raise RuntimeError("Streaming not supported")

        if self.responses:
            return self.responses.pop(0)
        return MockLLMResponse(content=self.default_content)


class TotallyBrokenClient:
    """Client where every call raises an exception."""

    async def complete(self, **kwargs):
        raise ConnectionError("LLM service unavailable")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

AGENT_RESPONSES = [
    MockLLMResponse(
        content="1. Parse input\n2. Sort using quicksort\n3. Return result",
        model="reasoning-model",
        cost=0.02,
        total_tokens=150,
    ),
    MockLLMResponse(
        content="def quicksort(arr):\n    if len(arr) <= 1: return arr\n    ...",
        model="coding-model",
        cost=0.03,
        total_tokens=250,
    ),
    MockLLMResponse(
        content="Correctness: OK. Edge case: empty list handled. Confidence: HIGH.",
        model="critic-model",
        cost=0.02,
        total_tokens=120,
    ),
    MockLLMResponse(
        content="Here is a quicksort implementation that handles edge cases...",
        model="synth-model",
        cost=0.01,
        total_tokens=180,
    ),
]


@pytest.fixture
def scratchpad():
    return Scratchpad(query="Implement a quicksort function in Python", budget=0.50)


@pytest.fixture
def mock_client():
    return MockLLMClient(responses=list(AGENT_RESPONSES))


# ---------------------------------------------------------------------------
# Tests: run_sequential_pipeline
# ---------------------------------------------------------------------------

class TestRunSequentialPipeline:
    """Tests for the synchronous (non-streaming) pipeline."""

    @pytest.mark.asyncio
    async def test_all_four_agents_run(self, scratchpad, mock_client):
        result = await run_sequential_pipeline(scratchpad, mock_client)

        assert len(result.entries) == 4
        assert mock_client.call_count == 4

    @pytest.mark.asyncio
    async def test_entries_in_correct_order(self, scratchpad, mock_client):
        result = await run_sequential_pipeline(scratchpad, mock_client)

        roles = [e.agent_role for e in result.entries]
        assert roles == [
            AgentRole.REASONER,
            AgentRole.CODER,
            AgentRole.CRITIC,
            AgentRole.SYNTHESIZER,
        ]

    @pytest.mark.asyncio
    async def test_entry_types_match_roles(self, scratchpad, mock_client):
        result = await run_sequential_pipeline(scratchpad, mock_client)

        types = [e.entry_type for e in result.entries]
        assert types == [
            EntryType.PLAN,
            EntryType.CODE,
            EntryType.CRITIQUE,
            EntryType.SYNTHESIS,
        ]

    @pytest.mark.asyncio
    async def test_scratchpad_accumulates_cost(self, scratchpad, mock_client):
        result = await run_sequential_pipeline(scratchpad, mock_client)

        expected_cost = sum(r.cost for r in AGENT_RESPONSES)
        assert result.total_cost == pytest.approx(expected_cost)

    @pytest.mark.asyncio
    async def test_content_is_recorded(self, scratchpad, mock_client):
        result = await run_sequential_pipeline(scratchpad, mock_client)

        assert "quicksort" in result.entries[0].content
        assert "def quicksort" in result.entries[1].content
        assert "Confidence: HIGH" in result.entries[2].content

    @pytest.mark.asyncio
    async def test_model_used_is_recorded(self, scratchpad, mock_client):
        result = await run_sequential_pipeline(scratchpad, mock_client)

        models = [e.model_used for e in result.entries]
        assert models == ["reasoning-model", "coding-model", "critic-model", "synth-model"]

    @pytest.mark.asyncio
    async def test_returns_same_scratchpad_instance(self, scratchpad, mock_client):
        result = await run_sequential_pipeline(scratchpad, mock_client)
        assert result is scratchpad

    @pytest.mark.asyncio
    async def test_model_assignments_passed_to_agents(self):
        """Per-role model overrides reach the LLM client."""
        client = MockLLMClient(responses=list(AGENT_RESPONSES))
        pad = Scratchpad(query="test", budget=1.0)

        assignments = {
            AgentRole.REASONER: "custom-reasoner",
            AgentRole.CODER: "custom-coder",
        }
        await run_sequential_pipeline(pad, client, model_assignments=assignments)

        models_requested = [c["model"] for c in client.calls]
        assert models_requested[0] == "custom-reasoner"
        assert models_requested[1] == "custom-coder"
        # Critic and Synthesizer have no override, so they get None.
        assert models_requested[2] is None
        assert models_requested[3] is None


# ---------------------------------------------------------------------------
# Tests: budget enforcement
# ---------------------------------------------------------------------------

class TestBudgetEnforcement:
    """Budget checks happen before each agent call."""

    @pytest.mark.asyncio
    async def test_budget_exceeded_skips_remaining(self):
        """When budget runs out after Reasoner, only 1 agent runs."""
        # Budget is 0.02; Reasoner costs 0.02 -> exactly at budget.
        pad = Scratchpad(query="test", budget=0.02)
        responses = [
            MockLLMResponse(content="plan", cost=0.02, total_tokens=50),
            MockLLMResponse(content="code", cost=0.03, total_tokens=100),
            MockLLMResponse(content="crit", cost=0.02, total_tokens=80),
            MockLLMResponse(content="synth", cost=0.01, total_tokens=60),
        ]
        client = MockLLMClient(responses=responses)

        result = await run_sequential_pipeline(pad, client)

        # Only the Reasoner ran; the rest were skipped because is_over_budget()
        # returns True when total_cost >= budget.
        assert len(result.entries) == 1
        assert result.entries[0].agent_role == AgentRole.REASONER
        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_budget_allows_partial_pipeline(self):
        """Budget allows Reasoner + Coder but not Critic."""
        pad = Scratchpad(query="test", budget=0.04)
        responses = [
            MockLLMResponse(content="plan", cost=0.02, total_tokens=50),
            MockLLMResponse(content="code", cost=0.02, total_tokens=100),
            MockLLMResponse(content="crit", cost=0.02, total_tokens=80),
            MockLLMResponse(content="synth", cost=0.01, total_tokens=60),
        ]
        client = MockLLMClient(responses=responses)

        result = await run_sequential_pipeline(pad, client)

        assert len(result.entries) == 2
        assert result.entries[0].agent_role == AgentRole.REASONER
        assert result.entries[1].agent_role == AgentRole.CODER
        assert client.call_count == 2

    @pytest.mark.asyncio
    async def test_zero_budget_runs_nothing(self):
        """A zero budget means no agents run at all."""
        pad = Scratchpad(query="test", budget=0.0)
        client = MockLLMClient(responses=list(AGENT_RESPONSES))

        result = await run_sequential_pipeline(pad, client)

        assert len(result.entries) == 0
        assert client.call_count == 0

    @pytest.mark.asyncio
    async def test_large_budget_runs_all(self):
        """A generous budget lets all 4 agents run."""
        pad = Scratchpad(query="test", budget=10.0)
        client = MockLLMClient(responses=list(AGENT_RESPONSES))

        result = await run_sequential_pipeline(pad, client)

        assert len(result.entries) == 4


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Agent failures must not crash the pipeline."""

    @pytest.mark.asyncio
    async def test_broken_client_does_not_crash(self):
        """If the LLM is totally unavailable the pipeline returns gracefully."""
        pad = Scratchpad(query="test", budget=1.0)
        client = TotallyBrokenClient()

        result = await run_sequential_pipeline(pad, client)

        # No entries because every call raised, but the function returned.
        assert len(result.entries) == 0
        assert result.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_partial_failure_still_continues(self):
        """If one agent fails, the next agents still attempt."""

        class FailOnSecondCall(MockLLMClient):
            async def complete(self, **kwargs):
                self.call_count += 1
                if self.call_count == 2:
                    raise RuntimeError("Coder exploded")
                if self.responses:
                    return self.responses.pop(0)
                return MockLLMResponse(content="fallback")

        responses = list(AGENT_RESPONSES)
        client = FailOnSecondCall(responses=responses)
        pad = Scratchpad(query="test", budget=1.0)

        result = await run_sequential_pipeline(pad, client)

        # Reasoner succeeded, Coder failed, Critic succeeded, Synthesizer succeeded
        roles = [e.agent_role for e in result.entries]
        assert AgentRole.REASONER in roles
        assert AgentRole.CODER not in roles  # failed
        assert AgentRole.CRITIC in roles
        assert AgentRole.SYNTHESIZER in roles
        assert len(result.entries) == 3


# ---------------------------------------------------------------------------
# Tests: stream_sequential_pipeline
# ---------------------------------------------------------------------------

class TestStreamSequentialPipeline:
    """Tests for the streaming variant."""

    @pytest.mark.asyncio
    async def test_yields_correct_event_sequence(self, scratchpad, mock_client):
        """Events should follow: agent_start -> agent_chunk(s) -> agent_done, repeated, then done."""
        events = []
        async for event in stream_sequential_pipeline(scratchpad, mock_client):
            events.append(event)

        # The last event is always "done".
        assert events[-1]["type"] == "done"

        # Verify the pattern for each agent.
        event_types = [e["type"] for e in events]

        # We should see 4 agent_start events.
        assert event_types.count("agent_start") == 4
        # We should see 4 agent_done events.
        assert event_types.count("agent_done") == 4
        # At least 4 agent_chunk events (one per agent minimum).
        assert event_types.count("agent_chunk") >= 4

    @pytest.mark.asyncio
    async def test_agent_start_has_name_and_role(self, scratchpad, mock_client):
        events = []
        async for event in stream_sequential_pipeline(scratchpad, mock_client):
            events.append(event)

        starts = [e for e in events if e["type"] == "agent_start"]
        assert len(starts) == 4

        names = [s["agent"] for s in starts]
        assert names == ["Reasoner", "Coder", "Critic", "Synthesizer"]

        roles = [s["role"] for s in starts]
        assert roles == ["reasoner", "coder", "critic", "synthesizer"]

    @pytest.mark.asyncio
    async def test_agent_done_has_cost_and_tokens(self, scratchpad, mock_client):
        events = []
        async for event in stream_sequential_pipeline(scratchpad, mock_client):
            events.append(event)

        dones = [e for e in events if e["type"] == "agent_done"]
        for d in dones:
            assert "cost" in d
            assert "tokens" in d
            assert isinstance(d["cost"], (int, float))
            assert isinstance(d["tokens"], (int, float))

    @pytest.mark.asyncio
    async def test_done_event_contains_scratchpad(self, scratchpad, mock_client):
        events = []
        async for event in stream_sequential_pipeline(scratchpad, mock_client):
            events.append(event)

        done = events[-1]
        assert done["type"] == "done"
        assert "total_cost" in done
        assert "agents_used" in done
        assert "scratchpad" in done
        assert done["agents_used"] == 4

    @pytest.mark.asyncio
    async def test_chunks_contain_content(self, scratchpad, mock_client):
        chunks = []
        async for event in stream_sequential_pipeline(scratchpad, mock_client):
            if event["type"] == "agent_chunk":
                chunks.append(event)

        assert len(chunks) >= 4
        for c in chunks:
            assert "content" in c
            assert len(c["content"]) > 0
            assert "agent" in c

    @pytest.mark.asyncio
    async def test_stream_budget_exceeded_yields_event(self):
        """When budget is exhausted, a budget_exceeded event is emitted.

        Streaming does not inherently track cost on the agent (the LLM
        response is chunked text), so we use FailingStreamClient which
        falls back to think() where cost is properly recorded.
        """
        pad = Scratchpad(query="test", budget=0.02)
        responses = [
            MockLLMResponse(content="plan only", cost=0.02, total_tokens=50),
            MockLLMResponse(content="never reached", cost=0.01, total_tokens=40),
        ]
        client = FailingStreamClient(responses=responses)

        events = []
        async for event in stream_sequential_pipeline(pad, client):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "budget_exceeded" in event_types
        # Only Reasoner ran, then budget_exceeded, then done.
        assert event_types.count("agent_start") == 1
        assert event_types.count("agent_done") == 1

    @pytest.mark.asyncio
    async def test_stream_budget_preexceeded(self):
        """When the scratchpad already exceeds budget, no agents run."""
        pad = Scratchpad(query="test", budget=0.05)
        # Pre-load cost to exhaust the budget.
        pad.total_cost = 0.05
        client = MockLLMClient(responses=list(AGENT_RESPONSES))

        events = []
        async for event in stream_sequential_pipeline(pad, client):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "budget_exceeded" in event_types
        assert event_types.count("agent_start") == 0
        assert events[-1]["type"] == "done"
        assert events[-1]["agents_used"] == 0

    @pytest.mark.asyncio
    async def test_stream_fallback_on_stream_failure(self):
        """If streaming fails, the pipeline falls back to think() transparently."""
        responses = list(AGENT_RESPONSES)
        client = FailingStreamClient(responses=responses)
        pad = Scratchpad(query="test", budget=1.0)

        events = []
        async for event in stream_sequential_pipeline(pad, client):
            events.append(event)

        # All 4 agents should still produce results via fallback.
        starts = [e for e in events if e["type"] == "agent_start"]
        dones = [e for e in events if e["type"] == "agent_done"]
        assert len(starts) == 4
        assert len(dones) == 4

        # The final done event should show 4 agents used.
        assert events[-1]["type"] == "done"
        assert events[-1]["agents_used"] == 4

    @pytest.mark.asyncio
    async def test_stream_order_is_sequential(self, scratchpad, mock_client):
        """Agent events never interleave -- each agent completes before the next starts."""
        events = []
        async for event in stream_sequential_pipeline(scratchpad, mock_client):
            events.append(event)

        # Extract agent-related events (exclude the final "done").
        agent_events = [e for e in events if e["type"] != "done"]

        current_agent = None
        for ev in agent_events:
            if ev["type"] == "agent_start":
                current_agent = ev["agent"]
            elif ev["type"] in ("agent_chunk", "agent_done"):
                assert ev["agent"] == current_agent, (
                    f"Expected events for {current_agent}, got {ev['agent']}"
                )

    @pytest.mark.asyncio
    async def test_stream_scratchpad_populated(self, scratchpad, mock_client):
        """After streaming, the scratchpad has entries from all agents."""
        async for _ in stream_sequential_pipeline(scratchpad, mock_client):
            pass

        assert len(scratchpad.entries) == 4
        roles = [e.agent_role for e in scratchpad.entries]
        assert roles == [
            AgentRole.REASONER,
            AgentRole.CODER,
            AgentRole.CRITIC,
            AgentRole.SYNTHESIZER,
        ]
