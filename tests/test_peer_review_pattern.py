"""Tests for the peer review pattern (5-step pipeline with rebuttal loop)."""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Dict

import pytest

from adam.agents.scratchpad import Scratchpad, ScratchpadEntry, AgentRole, EntryType
from adam.agents.patterns.peer_review import (
    run_peer_review_pipeline,
    stream_peer_review_pipeline,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

@dataclass
class MockLLMResponse:
    """Realistic mock response with the attributes Agent.think() reads."""
    content: str
    cost: float = 0.01
    total_tokens: int = 100
    model: str = "mock-model"


class MockLLMClient:
    """LLM client that returns canned responses keyed by system prompt substring.

    Each call to ``complete()`` matches the system_prompt against *responses*
    keys and returns the matching content. If no key matches, uses default.
    """

    def __init__(self, responses=None, fail_for=None, delay: float = 0.0):
        self.responses = responses or {}
        self.fail_for = fail_for or set()
        self.calls = []
        self.delay = delay

    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        stream: bool = False,
    ):
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "stream": stream,
            "model": model,
        })

        # Check for simulated failures
        for key in self.fail_for:
            if key in system_prompt:
                raise RuntimeError(f"Simulated failure for '{key}'")

        if self.delay:
            await asyncio.sleep(self.delay)

        content = "default response"
        for key, value in self.responses.items():
            if key in system_prompt:
                content = value
                break

        if stream:
            return self._stream_chunks(content)

        return MockLLMResponse(content=content)

    async def _stream_chunks(self, content: str):
        """Yield content word by word as an async generator."""
        words = content.split()
        for w in words:
            yield w + " "


class EmptyStreamFallbackClient(MockLLMClient):
    """Return empty content for streaming calls and normal content for fallback calls."""

    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        stream: bool = False,
    ):
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "stream": stream,
            "model": model,
        })

        content = "default response"
        for key, value in self.responses.items():
            if key in system_prompt:
                content = value
                break

        if stream:
            return self._stream_chunks("")

        return MockLLMResponse(content=content)


class EmptyEverywhereClient(EmptyStreamFallbackClient):
    """Return empty content for both streaming and non-streaming calls."""

    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        stream: bool = False,
    ):
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "stream": stream,
            "model": model,
        })
        if stream:
            return self._stream_chunks("")
        return MockLLMResponse(content="")


# ---------------------------------------------------------------------------
# System prompt identifiers (used to key mock responses)
# ---------------------------------------------------------------------------

# Use the full opening line of each system prompt so there's no ambiguity.
# The Synthesizer prompt mentions all specialist names, so we need the
# "You are a ..." prefix which is unique to each role.
REASONER_KEY = "You are a Reasoning Specialist"
CODER_KEY = "You are a Code Specialist"
CRITIC_KEY = "You are a Code Review Specialist"
SYNTHESIZER_KEY = "You are a Synthesis Specialist"


# ---------------------------------------------------------------------------
# Tests -- run_peer_review_pipeline: full pipeline
# ---------------------------------------------------------------------------

class TestRunPeerReviewPipeline:
    """Tests for the non-streaming peer review pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_produces_7_entries(self):
        """5 steps should produce exactly 7 entries: PLAN + CODE + CRITIQUE + REBUTTAL + REACTION + REACTION + SYNTHESIS."""
        client = MockLLMClient(responses={
            REASONER_KEY: "Step-by-step plan for the problem",
            CODER_KEY: "Code review of the plan",
            CRITIC_KEY: "Critique of the plan",
            SYNTHESIZER_KEY: "Final synthesis of all contributions",
        })
        pad = Scratchpad(query="How to implement a cache?", budget=1.0)
        result = await run_peer_review_pipeline(pad, client)

        assert len(result.entries) == 7

        types = [e.entry_type for e in result.entries]
        assert types[0] == EntryType.PLAN       # Step 1: Reasoner
        assert types[1] == EntryType.CODE        # Step 2: Coder review
        assert types[2] == EntryType.CRITIQUE    # Step 2: Critic review
        assert types[3] == EntryType.REBUTTAL    # Step 3: Reasoner rebuttal
        assert types[4] == EntryType.REACTION    # Step 4: Coder reaction
        assert types[5] == EntryType.REACTION    # Step 4: Critic reaction
        assert types[6] == EntryType.SYNTHESIS   # Step 5: Synthesizer

    @pytest.mark.asyncio
    async def test_budget_stops_pipeline_and_still_synthesizes(self):
        """When budget is exhausted mid-pipeline, skip to synthesis on partial results."""
        # Budget of 0.03: Reasoner costs 0.01, Coder costs 0.01, Critic costs 0.01 = 0.03.
        # After Step 2, budget is exhausted. Pipeline should skip to synthesis.
        client = MockLLMClient(responses={
            REASONER_KEY: "plan output",
            CODER_KEY: "code review output",
            CRITIC_KEY: "critique output",
            SYNTHESIZER_KEY: "synthesis from partial data",
        })
        pad = Scratchpad(query="test budget", budget=0.03)
        result = await run_peer_review_pipeline(pad, client)

        # Should have: PLAN + CODE + CRITIQUE + SYNTHESIS (skipped steps 3-4)
        types = [e.entry_type for e in result.entries]
        assert EntryType.PLAN in types
        assert EntryType.SYNTHESIS in types
        # The total should be less than 7 (full pipeline)
        assert len(result.entries) < 7

    @pytest.mark.asyncio
    async def test_reviews_run_in_parallel(self):
        """Step 2 (reviews) should run in parallel via asyncio.gather.

        We verify this by adding a delay and checking total time is less than
        serial execution would take.
        """
        client = MockLLMClient(
            responses={
                REASONER_KEY: "quick plan",
                CODER_KEY: "code review",
                CRITIC_KEY: "critique review",
                SYNTHESIZER_KEY: "synthesis",
            },
            delay=0.1,  # 100ms per call
        )
        pad = Scratchpad(query="test parallel", budget=1.0)

        start = time.monotonic()
        await run_peer_review_pipeline(pad, client)
        elapsed = time.monotonic() - start

        # 5 sequential steps but steps 2 and 4 are parallel (2 agents each).
        # Serial would be 7 * 0.1 = 0.7s.  Parallel saves ~0.2s -> should be ~0.5s.
        # We allow generous margin: just check < 0.65s (less than fully serial).
        assert elapsed < 0.65, f"Took {elapsed:.2f}s -- reviews may not be parallel"

    @pytest.mark.asyncio
    async def test_agent_failure_does_not_crash(self):
        """If one agent fails, the pipeline continues with remaining agents."""
        client = MockLLMClient(
            responses={
                REASONER_KEY: "plan",
                CODER_KEY: "code review",
                SYNTHESIZER_KEY: "synthesis from partial",
            },
            fail_for={CRITIC_KEY},
        )
        pad = Scratchpad(query="test failure handling", budget=1.0)
        result = await run_peer_review_pipeline(pad, client)

        # Should not crash; we get at least plan + code + synthesis
        types = [e.entry_type for e in result.entries]
        assert EntryType.PLAN in types
        assert EntryType.SYNTHESIS in types

    @pytest.mark.asyncio
    async def test_model_assignments(self):
        """Per-role model overrides reach the LLM client."""
        client = MockLLMClient(responses={
            REASONER_KEY: "plan",
            CODER_KEY: "code",
            CRITIC_KEY: "critique",
            SYNTHESIZER_KEY: "synth",
        })
        pad = Scratchpad(query="test models", budget=1.0)
        assignments = {
            "reasoner": "model-reason",
            "coder": "model-code",
            "critic": "model-critic",
            "synthesizer": "model-synth",
        }
        await run_peer_review_pipeline(pad, client, model_assignments=assignments)

        models_used = [c["model"] for c in client.calls]
        assert "model-reason" in models_used
        assert "model-code" in models_used
        assert "model-critic" in models_used
        assert "model-synth" in models_used

    @pytest.mark.asyncio
    async def test_returns_same_scratchpad_instance(self):
        client = MockLLMClient(responses={
            REASONER_KEY: "plan",
            CODER_KEY: "code",
            CRITIC_KEY: "critique",
            SYNTHESIZER_KEY: "synth",
        })
        pad = Scratchpad(query="test", budget=1.0)
        result = await run_peer_review_pipeline(pad, client)
        assert result is pad


# ---------------------------------------------------------------------------
# Tests -- custom context per step
# ---------------------------------------------------------------------------

class TestCustomContext:
    """Verify that each step sees the correct subset of previous entries."""

    @pytest.mark.asyncio
    async def test_step3_reasoner_sees_reviews(self):
        """Step 3 Reasoner (rebuttal) should see the PLAN + CODE + CRITIQUE entries."""
        client = MockLLMClient(responses={
            REASONER_KEY: "initial plan for the problem",
            CODER_KEY: "code review: looks good but missing tests",
            CRITIC_KEY: "critique: edge cases not handled",
            SYNTHESIZER_KEY: "final synthesis",
        })
        pad = Scratchpad(query="test context visibility", budget=1.0)
        await run_peer_review_pipeline(pad, client)

        # Find the rebuttal call (Step 3) -- it's the second call with REASONER_KEY
        reasoner_calls = [
            c for c in client.calls if REASONER_KEY in c["system_prompt"]
        ]
        # Should have 2 reasoner calls: Step 1 (produce) and Step 3 (rebuttal)
        assert len(reasoner_calls) == 2

        rebuttal_prompt = reasoner_calls[1]["prompt"]
        # The rebuttal prompt should contain the code review and critique
        assert "code review" in rebuttal_prompt.lower() or "looks good" in rebuttal_prompt
        assert "critique" in rebuttal_prompt.lower() or "edge cases" in rebuttal_prompt

    @pytest.mark.asyncio
    async def test_step4_coder_sees_only_own_review(self):
        """Step 4 Coder reaction should see only CODE entry, not CRITIQUE."""
        client = MockLLMClient(responses={
            REASONER_KEY: "plan output",
            CODER_KEY: "CODER_UNIQUE_MARKER: my code review",
            CRITIC_KEY: "CRITIC_UNIQUE_MARKER: my critique",
            SYNTHESIZER_KEY: "synthesis",
        })
        pad = Scratchpad(query="test coder context", budget=1.0)
        await run_peer_review_pipeline(pad, client)

        # Find the Coder's Step 4 reaction call -- it's the second Coder call
        coder_calls = [
            c for c in client.calls if CODER_KEY in c["system_prompt"]
        ]
        assert len(coder_calls) == 2  # Step 2 review + Step 4 reaction

        reaction_prompt = coder_calls[1]["prompt"]
        # Coder should see its own CODE review
        assert "CODER_UNIQUE_MARKER" in reaction_prompt
        # Coder should NOT see Critic's CRITIQUE
        assert "CRITIC_UNIQUE_MARKER" not in reaction_prompt

    @pytest.mark.asyncio
    async def test_step4_critic_sees_only_own_review(self):
        """Step 4 Critic reaction should see only CRITIQUE entry, not CODE."""
        client = MockLLMClient(responses={
            REASONER_KEY: "plan output",
            CODER_KEY: "CODER_UNIQUE_MARKER: my code review",
            CRITIC_KEY: "CRITIC_UNIQUE_MARKER: my critique",
            SYNTHESIZER_KEY: "synthesis",
        })
        pad = Scratchpad(query="test critic context", budget=1.0)
        await run_peer_review_pipeline(pad, client)

        # Find the Critic's Step 4 reaction call -- it's the second Critic call
        critic_calls = [
            c for c in client.calls if CRITIC_KEY in c["system_prompt"]
        ]
        assert len(critic_calls) == 2  # Step 2 review + Step 4 reaction

        reaction_prompt = critic_calls[1]["prompt"]
        # Critic should see its own CRITIQUE review
        assert "CRITIC_UNIQUE_MARKER" in reaction_prompt
        # Critic should NOT see Coder's CODE review
        assert "CODER_UNIQUE_MARKER" not in reaction_prompt

    @pytest.mark.asyncio
    async def test_step1_sees_query_only(self):
        """Step 1 Reasoner should see only the query, no prior entries."""
        client = MockLLMClient(responses={
            REASONER_KEY: "plan",
            CODER_KEY: "code",
            CRITIC_KEY: "critique",
            SYNTHESIZER_KEY: "synth",
        })
        pad = Scratchpad(query="unique_query_string_42", budget=1.0)
        await run_peer_review_pipeline(pad, client)

        # First call is Step 1 Reasoner
        first_prompt = client.calls[0]["prompt"]
        assert "unique_query_string_42" in first_prompt
        # Should not contain any entry headers (no prior entries)
        assert "---" not in first_prompt

    @pytest.mark.asyncio
    async def test_step5_synthesizer_sees_everything(self):
        """Step 5 Synthesizer should see all entries."""
        client = MockLLMClient(responses={
            REASONER_KEY: "PLAN_MARKER_XYZ",
            CODER_KEY: "CODE_MARKER_XYZ",
            CRITIC_KEY: "CRITIQUE_MARKER_XYZ",
            SYNTHESIZER_KEY: "final output",
        })
        pad = Scratchpad(query="synth visibility test", budget=1.0)
        await run_peer_review_pipeline(pad, client)

        synth_calls = [
            c for c in client.calls if SYNTHESIZER_KEY in c["system_prompt"]
        ]
        assert len(synth_calls) == 1

        synth_prompt = synth_calls[0]["prompt"]
        assert "PLAN_MARKER_XYZ" in synth_prompt
        assert "CODE_MARKER_XYZ" in synth_prompt
        assert "CRITIQUE_MARKER_XYZ" in synth_prompt


# ---------------------------------------------------------------------------
# Tests -- stream_peer_review_pipeline
# ---------------------------------------------------------------------------

class TestStreamPeerReviewPipeline:
    """Tests for the streaming peer review pipeline."""

    @pytest.mark.asyncio
    async def test_streaming_emits_correct_event_types(self):
        """Streaming should emit step_start, agent_start, agent_chunk,
        agent_done, step_complete, and done events."""
        client = MockLLMClient(responses={
            REASONER_KEY: "plan output",
            CODER_KEY: "code output",
            CRITIC_KEY: "critique output",
            SYNTHESIZER_KEY: "synthesis output",
        })
        pad = Scratchpad(query="test streaming events", budget=1.0)

        events = []
        async for event in stream_peer_review_pipeline(pad, client):
            events.append(event)

        event_types = {e["type"] for e in events}
        assert "step_start" in event_types
        assert "agent_start" in event_types
        assert "agent_chunk" in event_types
        assert "agent_done" in event_types
        assert "step_complete" in event_types
        assert "done" in event_types

    @pytest.mark.asyncio
    async def test_streaming_ends_with_done(self):
        client = MockLLMClient(responses={
            REASONER_KEY: "plan",
            CODER_KEY: "code",
            CRITIC_KEY: "critique",
            SYNTHESIZER_KEY: "synth",
        })
        pad = Scratchpad(query="test done event", budget=1.0)

        events = []
        async for event in stream_peer_review_pipeline(pad, client):
            events.append(event)

        assert events[-1]["type"] == "done"
        assert "total_cost" in events[-1]
        assert "scratchpad" in events[-1]

    @pytest.mark.asyncio
    async def test_streaming_produces_5_step_starts(self):
        """There should be 5 step_start events for the 5 steps."""
        client = MockLLMClient(responses={
            REASONER_KEY: "plan",
            CODER_KEY: "code",
            CRITIC_KEY: "critique",
            SYNTHESIZER_KEY: "synth",
        })
        pad = Scratchpad(query="test steps", budget=1.0)

        events = []
        async for event in stream_peer_review_pipeline(pad, client):
            events.append(event)

        step_starts = [e for e in events if e["type"] == "step_start"]
        assert len(step_starts) == 5

    @pytest.mark.asyncio
    async def test_streaming_populates_scratchpad(self):
        """After streaming, the scratchpad should have all 7 entries."""
        client = MockLLMClient(responses={
            REASONER_KEY: "plan",
            CODER_KEY: "code",
            CRITIC_KEY: "critique",
            SYNTHESIZER_KEY: "synth",
        })
        pad = Scratchpad(query="test scratchpad", budget=1.0)

        async for _ in stream_peer_review_pipeline(pad, client):
            pass

        assert len(pad.entries) == 7

    @pytest.mark.asyncio
    async def test_streaming_budget_exceeded(self):
        """Budget exceeded during streaming should still produce a done event."""
        client = MockLLMClient(responses={
            REASONER_KEY: "plan",
            CODER_KEY: "code",
            CRITIC_KEY: "critique",
            SYNTHESIZER_KEY: "synth from partial",
        })
        pad = Scratchpad(query="test budget streaming", budget=0.01)
        # Pre-exhaust budget
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.REASONER, agent_name="prior",
            content="x", entry_type=EntryType.NOTE, cost=0.02,
        ))

        events = []
        async for event in stream_peer_review_pipeline(pad, client):
            events.append(event)

        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_streaming_agent_chunks_have_content(self):
        """All agent_chunk events should have non-empty content."""
        client = MockLLMClient(responses={
            REASONER_KEY: "some plan content",
            CODER_KEY: "some code content",
            CRITIC_KEY: "some critique content",
            SYNTHESIZER_KEY: "final synthesis content",
        })
        pad = Scratchpad(query="test chunks", budget=1.0)

        chunks = []
        async for event in stream_peer_review_pipeline(pad, client):
            if event["type"] == "agent_chunk":
                chunks.append(event)

        assert len(chunks) > 0
        for c in chunks:
            assert "content" in c
            assert len(c["content"]) > 0
            assert "agent" in c

    @pytest.mark.asyncio
    async def test_empty_stream_falls_back_to_non_stream_content(self):
        """An empty stream should retry once in non-stream mode before marking done."""
        client = EmptyStreamFallbackClient(responses={
            REASONER_KEY: "plan",
            CODER_KEY: "code",
            CRITIC_KEY: "critique from fallback",
            SYNTHESIZER_KEY: "synth",
        })
        pad = Scratchpad(query="test empty stream fallback", budget=1.0)

        events = []
        async for event in stream_peer_review_pipeline(pad, client):
            events.append(event)

        critic_chunks = [
            e for e in events
            if e["type"] == "agent_chunk" and e["agent"] == "Critic"
        ]
        assert any("critique from fallback" in e["content"] for e in critic_chunks)

    @pytest.mark.asyncio
    async def test_empty_stream_and_fallback_emit_agent_error(self):
        """Empty output in both modes should surface as an agent error, not blank success."""
        client = EmptyEverywhereClient()
        pad = Scratchpad(query="test empty stream error", budget=1.0)

        events = []
        async for event in stream_peer_review_pipeline(pad, client):
            events.append(event)

        critic_errors = [
            e for e in events
            if e["type"] == "agent_error" and e["agent"] == "Critic"
        ]
        assert critic_errors
