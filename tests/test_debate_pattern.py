"""Tests for the debate pattern (two perspectives + reconciler)."""

import asyncio
from dataclasses import dataclass
from typing import Optional

import pytest

from adam.agents.scratchpad import Scratchpad, ScratchpadEntry, AgentRole, EntryType
from adam.agents.patterns.debate import (
    run_debate_pipeline,
    stream_debate_pipeline,
    _create_debater_a,
    _create_debater_b,
    _create_reconciler,
    DEBATER_A_PROMPT,
    DEBATER_B_PROMPT,
    RECONCILER_PROMPT,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

@dataclass
class MockLLMResponse:
    content: str
    cost: float = 0.01
    total_tokens: int = 100
    model: str = "mock-model"


class MockLLMClient:
    """LLM client that returns canned responses keyed by system prompt."""

    def __init__(self, responses=None, fail_for=None):
        self.responses = responses or {}
        self.fail_for = fail_for or set()
        self.calls = []

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

        # Determine which agent is calling by matching system_prompt
        for key in self.fail_for:
            if key in system_prompt:
                raise RuntimeError(f"Simulated failure for '{key}'")

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


# ---------------------------------------------------------------------------
# Tests — run_debate_pipeline
# ---------------------------------------------------------------------------

class TestRunDebatePipeline:
    """Tests for the non-streaming debate pipeline."""

    @pytest.mark.asyncio
    async def test_both_debaters_run(self):
        client = MockLLMClient(responses={
            DEBATER_A_PROMPT: "Solution A: use quicksort",
            DEBATER_B_PROMPT: "Solution B: use mergesort",
            RECONCILER_PROMPT: "Combined: use timsort",
        })
        pad = Scratchpad(query="Sort a list efficiently")
        result = await run_debate_pipeline(pad, client)

        # Both debater entries should be present
        plan_entries = result.get_entries_by_type(EntryType.PLAN)
        assert len(plan_entries) == 2
        names = {e.agent_name for e in plan_entries}
        assert "Perspective A" in names
        assert "Perspective B" in names

    @pytest.mark.asyncio
    async def test_reconciler_sees_both_outputs(self):
        client = MockLLMClient(responses={
            DEBATER_A_PROMPT: "Solution A: use quicksort",
            DEBATER_B_PROMPT: "Solution B: use mergesort",
            RECONCILER_PROMPT: "Combined: use timsort",
        })
        pad = Scratchpad(query="Sort a list efficiently")
        result = await run_debate_pipeline(pad, client)

        # Reconciler entry should exist
        synth = result.get_entries_by_type(EntryType.SYNTHESIS)
        assert len(synth) == 1
        assert synth[0].agent_name == "Reconciler"

        # The reconciler's call should have seen both perspectives in its prompt
        reconciler_call = [
            c for c in client.calls if RECONCILER_PROMPT in c["system_prompt"]
        ]
        assert len(reconciler_call) == 1
        # Synthesizer sees everything via build_context_for_agent
        prompt_text = reconciler_call[0]["prompt"]
        assert "Solution A" in prompt_text
        assert "Solution B" in prompt_text

    @pytest.mark.asyncio
    async def test_budget_exceeded_before_debaters(self):
        client = MockLLMClient()
        pad = Scratchpad(query="test", budget=0.01)
        # Burn budget
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.REASONER, agent_name="prior",
            content="x", entry_type=EntryType.NOTE, cost=0.02,
        ))
        result = await run_debate_pipeline(pad, client)
        # No new entries beyond the pre-existing one
        assert len(result.entries) == 1
        assert len(client.calls) == 0

    @pytest.mark.asyncio
    async def test_budget_exceeded_after_debaters_skips_reconciler(self):
        client = MockLLMClient(responses={
            DEBATER_A_PROMPT: "A output",
            DEBATER_B_PROMPT: "B output",
        })
        pad = Scratchpad(query="test", budget=0.02)
        result = await run_debate_pipeline(pad, client)

        # Both debaters ran (cost 0.01 each = 0.02 total)
        plan_entries = result.get_entries_by_type(EntryType.PLAN)
        assert len(plan_entries) == 2

        # Reconciler should NOT have run — budget exhausted
        synth = result.get_entries_by_type(EntryType.SYNTHESIS)
        assert len(synth) == 0

    @pytest.mark.asyncio
    async def test_one_debater_fails_still_reconciles(self):
        client = MockLLMClient(
            responses={
                DEBATER_A_PROMPT: "A succeeded",
                RECONCILER_PROMPT: "Reconciled from A only",
            },
            fail_for={DEBATER_B_PROMPT},
        )
        pad = Scratchpad(query="test")
        result = await run_debate_pipeline(pad, client)

        plans = result.get_entries_by_type(EntryType.PLAN)
        assert len(plans) == 1
        assert plans[0].agent_name == "Perspective A"

        synth = result.get_entries_by_type(EntryType.SYNTHESIS)
        assert len(synth) == 1

    @pytest.mark.asyncio
    async def test_both_debaters_fail(self):
        client = MockLLMClient(
            fail_for={DEBATER_A_PROMPT, DEBATER_B_PROMPT},
        )
        pad = Scratchpad(query="test")
        result = await run_debate_pipeline(pad, client)

        assert len(result.entries) == 0

    @pytest.mark.asyncio
    async def test_model_assignments(self):
        client = MockLLMClient(responses={
            DEBATER_A_PROMPT: "A",
            DEBATER_B_PROMPT: "B",
            RECONCILER_PROMPT: "R",
        })
        pad = Scratchpad(query="test")
        assignments = {
            "debater_a": "model-fast",
            "debater_b": "model-creative",
            "reconciler": "model-smart",
        }
        await run_debate_pipeline(pad, client, model_assignments=assignments)

        models_used = [c["model"] for c in client.calls]
        assert "model-fast" in models_used
        assert "model-creative" in models_used
        assert "model-smart" in models_used


# ---------------------------------------------------------------------------
# Tests — agent creation helpers
# ---------------------------------------------------------------------------

class TestDebaterCreation:
    def test_debater_a_is_reasoner(self):
        agent = _create_debater_a(MockLLMClient())
        assert agent.config.role == AgentRole.REASONER
        assert agent.config.name == "Perspective A"
        assert agent.config.entry_type == EntryType.PLAN

    def test_debater_b_is_reasoner(self):
        agent = _create_debater_b(MockLLMClient())
        assert agent.config.role == AgentRole.REASONER
        assert agent.config.name == "Perspective B"
        assert agent.config.entry_type == EntryType.PLAN

    def test_reconciler_is_synthesizer(self):
        agent = _create_reconciler(MockLLMClient())
        assert agent.config.role == AgentRole.SYNTHESIZER
        assert agent.config.name == "Reconciler"
        assert agent.config.entry_type == EntryType.SYNTHESIS

    def test_debaters_have_different_prompts(self):
        a = _create_debater_a(MockLLMClient())
        b = _create_debater_b(MockLLMClient())
        assert a.config.system_prompt != b.config.system_prompt


# ---------------------------------------------------------------------------
# Tests — stream_debate_pipeline
# ---------------------------------------------------------------------------

class TestStreamDebatePipeline:
    """Tests for the streaming debate pipeline."""

    @pytest.mark.asyncio
    async def test_yields_correct_event_structure(self):
        client = MockLLMClient(responses={
            DEBATER_A_PROMPT: "alpha response",
            DEBATER_B_PROMPT: "beta response",
            RECONCILER_PROMPT: "reconciled output",
        })
        pad = Scratchpad(query="test streaming")

        events = []
        async for event in stream_debate_pipeline(pad, client):
            events.append(event)

        types = [e["type"] for e in events]

        # Must start with debate_start
        assert types[0] == "debate_start"
        assert events[0]["debaters"] == ["Perspective A", "Perspective B"]

        # Must contain agent_start for both debaters and reconciler
        agent_starts = [e for e in events if e["type"] == "agent_start"]
        agent_start_names = {e["agent"] for e in agent_starts}
        assert "Perspective A" in agent_start_names
        assert "Perspective B" in agent_start_names
        assert "Reconciler" in agent_start_names

        # Must have agent_chunk events for both debaters
        chunk_agents = {e["agent"] for e in events if e["type"] == "agent_chunk"}
        assert "Perspective A" in chunk_agents
        assert "Perspective B" in chunk_agents
        assert "Reconciler" in chunk_agents

        # Must have reconciliation_start before reconciler chunks
        assert "reconciliation_start" in types
        recon_idx = types.index("reconciliation_start")
        # All reconciler chunks come after reconciliation_start
        for i, e in enumerate(events):
            if e["type"] == "agent_chunk" and e.get("agent") == "Reconciler":
                assert i > recon_idx

        # Must end with done
        assert types[-1] == "done"
        assert events[-1]["reason"] == "complete"

    @pytest.mark.asyncio
    async def test_interleaved_debater_chunks(self):
        client = MockLLMClient(responses={
            DEBATER_A_PROMPT: "word1 word2 word3",
            DEBATER_B_PROMPT: "alpha beta gamma",
            RECONCILER_PROMPT: "merged",
        })
        pad = Scratchpad(query="test interleaving")

        chunks = []
        async for event in stream_debate_pipeline(pad, client):
            if event["type"] == "agent_chunk" and event["agent"] in (
                "Perspective A", "Perspective B"
            ):
                chunks.append(event["agent"])

        # Both perspectives should appear, and they should be interleaved
        assert "Perspective A" in chunks
        assert "Perspective B" in chunks
        # Interleaving means we see alternation (A, B, A, B, ...)
        # Check that the first A and first B are within 2 positions of each other
        first_a = chunks.index("Perspective A")
        first_b = chunks.index("Perspective B")
        assert abs(first_a - first_b) <= 1

    @pytest.mark.asyncio
    async def test_stream_budget_exceeded_early_exit(self):
        client = MockLLMClient()
        pad = Scratchpad(query="test", budget=0.001)
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.REASONER, agent_name="prior",
            content="x", entry_type=EntryType.NOTE, cost=0.01,
        ))

        events = []
        async for event in stream_debate_pipeline(pad, client):
            events.append(event)

        types = [e["type"] for e in events]
        assert "debate_start" in types
        assert types[-1] == "done"
        assert events[-1]["reason"] == "budget_exceeded"
        # No agent_chunk should have been emitted
        assert "agent_chunk" not in types

    @pytest.mark.asyncio
    async def test_stream_scratchpad_updated(self):
        client = MockLLMClient(responses={
            DEBATER_A_PROMPT: "A output",
            DEBATER_B_PROMPT: "B output",
            RECONCILER_PROMPT: "reconciled",
        })
        pad = Scratchpad(query="test pad update")

        async for _ in stream_debate_pipeline(pad, client):
            pass

        plans = pad.get_entries_by_type(EntryType.PLAN)
        assert len(plans) == 2

        synth = pad.get_entries_by_type(EntryType.SYNTHESIS)
        assert len(synth) == 1
        assert synth[0].agent_name == "Reconciler"
