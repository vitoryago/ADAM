"""Debate pattern: two agents solve independently, a reconciler synthesizes.

Two debater agents (both Reasoners with different perspectives) analyze
the same query in parallel. A Synthesizer then reconciles both perspectives
into a single, unified response.
"""

import asyncio
import logging
from typing import Optional, Dict, AsyncGenerator

from ..base import Agent, AgentConfig
from ..scratchpad import Scratchpad, ScratchpadEntry, AgentRole, EntryType

logger = logging.getLogger(__name__)

# --- Debater System Prompts ---

DEBATER_A_PROMPT = "Analyze this problem and propose your best solution. Be thorough and specific."

DEBATER_B_PROMPT = (
    "Analyze this problem independently. "
    "Consider unconventional approaches and edge cases."
)

RECONCILER_PROMPT = (
    "You are a Reconciler. You have read two independent perspectives on the same problem. "
    "Your job is to synthesize the best elements of both into a single, coherent solution. "
    "Where the perspectives agree, reinforce the conclusion. "
    "Where they disagree, weigh the arguments and choose the stronger one, "
    "explaining why. Present a unified answer."
)


def _create_debater_a(llm_client, preferred_model: Optional[str] = None) -> Agent:
    """Create the first debater agent (thorough, conventional)."""
    config = AgentConfig(
        role=AgentRole.REASONER,
        name="Perspective A",
        system_prompt=DEBATER_A_PROMPT,
        entry_type=EntryType.PLAN,
        preferred_model=preferred_model,
        temperature=0.5,
        max_tokens=2000,
    )
    return Agent(config, llm_client)


def _create_debater_b(llm_client, preferred_model: Optional[str] = None) -> Agent:
    """Create the second debater agent (unconventional, edge-case focused)."""
    config = AgentConfig(
        role=AgentRole.REASONER,
        name="Perspective B",
        system_prompt=DEBATER_B_PROMPT,
        entry_type=EntryType.PLAN,
        preferred_model=preferred_model,
        temperature=0.7,
        max_tokens=2000,
    )
    return Agent(config, llm_client)


def _create_reconciler(llm_client, preferred_model: Optional[str] = None) -> Agent:
    """Create the reconciler agent that synthesizes both perspectives."""
    config = AgentConfig(
        role=AgentRole.SYNTHESIZER,
        name="Reconciler",
        system_prompt=RECONCILER_PROMPT,
        entry_type=EntryType.SYNTHESIS,
        preferred_model=preferred_model,
        temperature=0.6,
        max_tokens=4000,
    )
    return Agent(config, llm_client)


async def _run_debater(
    agent: Agent,
    scratchpad: Scratchpad,
) -> Optional[ScratchpadEntry]:
    """Run a single debater, returning None on failure."""
    try:
        entry = await agent.think(scratchpad)
        return entry
    except Exception as exc:
        logger.warning("Debater '%s' failed: %s", agent.config.name, exc)
        return None


async def run_debate_pipeline(
    scratchpad: Scratchpad,
    llm_client,
    model_assignments: Optional[Dict[str, str]] = None,
) -> Scratchpad:
    """Run the debate pattern: two debaters in parallel, then a reconciler.

    Args:
        scratchpad: Shared scratchpad (must already contain the query).
        llm_client: LLM client for all agents.
        model_assignments: Optional mapping of role names to model ids,
            e.g. {"debater_a": "gpt-4", "debater_b": "gpt-4", "reconciler": "gpt-4"}.

    Returns:
        The scratchpad with all debate entries added.
    """
    model_assignments = model_assignments or {}

    # --- Budget check before debaters ---
    if scratchpad.is_over_budget():
        logger.warning("Budget exceeded before debate — skipping.")
        return scratchpad

    # Create debater agents
    debater_a = _create_debater_a(
        llm_client, model_assignments.get("debater_a")
    )
    debater_b = _create_debater_b(
        llm_client, model_assignments.get("debater_b")
    )

    # Run both debaters in parallel
    results = await asyncio.gather(
        _run_debater(debater_a, scratchpad),
        _run_debater(debater_b, scratchpad),
    )

    entry_a, entry_b = results

    # Add whichever debater results succeeded
    if entry_a is not None:
        scratchpad.add_entry(entry_a)
    if entry_b is not None:
        scratchpad.add_entry(entry_b)

    # If neither debater produced output, return early
    if entry_a is None and entry_b is None:
        logger.error("Both debaters failed — no reconciliation possible.")
        return scratchpad

    # --- Budget check before reconciler ---
    if scratchpad.is_over_budget():
        logger.warning("Budget exceeded after debaters — skipping reconciler.")
        return scratchpad

    # Reconcile
    reconciler = _create_reconciler(
        llm_client, model_assignments.get("reconciler")
    )
    reconciler_entry = await reconciler.think(scratchpad)
    scratchpad.add_entry(reconciler_entry)

    return scratchpad


async def stream_debate_pipeline(
    scratchpad: Scratchpad,
    llm_client,
    model_assignments: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[Dict, None]:
    """Stream the debate pattern, yielding progress events.

    Event types:
        debate_start   — signals the beginning, lists debater names
        agent_start    — a debater or reconciler is starting
        agent_chunk    — a token/chunk from a debater or reconciler
        agent_done     — a debater or reconciler has finished
        reconciliation_start — signals the reconciler is beginning
        done           — the entire pipeline has finished

    For the two debaters the chunks are interleaved (A, B, A, B, ...).

    Yields:
        dict events as described above.
    """
    model_assignments = model_assignments or {}

    debater_names = ["Perspective A", "Perspective B"]

    yield {"type": "debate_start", "debaters": debater_names}

    # --- Budget check ---
    if scratchpad.is_over_budget():
        yield {"type": "done", "reason": "budget_exceeded"}
        return

    # Create agents
    debater_a = _create_debater_a(
        llm_client, model_assignments.get("debater_a")
    )
    debater_b = _create_debater_b(
        llm_client, model_assignments.get("debater_b")
    )

    # Signal start for both debaters
    yield {"type": "agent_start", "agent": "Perspective A", "role": "reasoner"}
    yield {"type": "agent_start", "agent": "Perspective B", "role": "reasoner"}

    # Stream both debaters interleaved
    stream_a = debater_a.think_stream(scratchpad)
    stream_b = debater_b.think_stream(scratchpad)

    content_a = []
    content_b = []
    done_a = False
    done_b = False

    async def _next_chunk(stream):
        """Get the next chunk from an async generator, or return sentinel."""
        try:
            return await stream.__anext__()
        except StopAsyncIteration:
            return None

    while not (done_a and done_b):
        if not done_a:
            chunk_a = await _next_chunk(stream_a)
            if chunk_a is None:
                done_a = True
            else:
                content_a.append(chunk_a)
                yield {"type": "agent_chunk", "agent": "Perspective A", "content": chunk_a}

        if not done_b:
            chunk_b = await _next_chunk(stream_b)
            if chunk_b is None:
                done_b = True
            else:
                content_b.append(chunk_b)
                yield {"type": "agent_chunk", "agent": "Perspective B", "content": chunk_b}

    # Build entries for both debaters
    full_a = "".join(content_a)
    full_b = "".join(content_b)

    if full_a:
        scratchpad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.REASONER,
            agent_name="Perspective A",
            content=full_a,
            entry_type=EntryType.PLAN,
        ))
        yield {"type": "agent_done", "agent": "Perspective A"}

    if full_b:
        scratchpad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.REASONER,
            agent_name="Perspective B",
            content=full_b,
            entry_type=EntryType.PLAN,
        ))
        yield {"type": "agent_done", "agent": "Perspective B"}

    if not full_a and not full_b:
        yield {"type": "done", "reason": "all_debaters_failed"}
        return

    # --- Budget check before reconciler ---
    if scratchpad.is_over_budget():
        yield {"type": "done", "reason": "budget_exceeded"}
        return

    # Reconciliation
    yield {"type": "reconciliation_start"}
    yield {"type": "agent_start", "agent": "Reconciler", "role": "synthesizer"}

    reconciler = _create_reconciler(
        llm_client, model_assignments.get("reconciler")
    )

    reconciler_content = []
    async for chunk in reconciler.think_stream(scratchpad):
        reconciler_content.append(chunk)
        yield {"type": "agent_chunk", "agent": "Reconciler", "content": chunk}

    full_reconciler = "".join(reconciler_content)
    if full_reconciler:
        scratchpad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.SYNTHESIZER,
            agent_name="Reconciler",
            content=full_reconciler,
            entry_type=EntryType.SYNTHESIS,
        ))

    yield {"type": "agent_done", "agent": "Reconciler"}
    yield {"type": "done", "reason": "complete"}
