"""Peer Review pattern: produce, review, rebut, react, synthesize.

A five-step pipeline where a Reasoner produces an initial plan, two
reviewers (Coder and Critic) evaluate it in parallel, the Reasoner
rebuts both reviews, the reviewers react to the rebuttal in parallel,
and a Synthesizer brings everything together.

This pattern bypasses ``build_context_for_agent()`` and builds custom
prompts so that each step sees only the entries it should.
"""

import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, Optional, List
from datetime import datetime

from ..base import Agent, AgentConfig
from ..scratchpad import Scratchpad, ScratchpadEntry, AgentRole, EntryType
from ..roles import (
    create_reasoner,
    create_coder,
    create_critic,
    create_synthesizer,
    REASONER_PROMPT,
    CODER_PROMPT,
    CRITIC_PROMPT,
    SYNTHESIZER_PROMPT,
)

logger = logging.getLogger(__name__)

# Step labels for logging and events.
_STEP_NAMES = {
    1: "PRODUCE",
    2: "REVIEW",
    3: "REBUTTAL",
    4: "REACT",
    5: "SYNTHESIZE",
}


# ---------------------------------------------------------------------------
# Context builders (bypass build_context_for_agent)
# ---------------------------------------------------------------------------

def _build_context(
    scratchpad: Scratchpad,
    entry_types: Optional[List[EntryType]] = None,
    agent_name_filter: Optional[str] = None,
) -> str:
    """Build a custom context string from the scratchpad.

    Args:
        scratchpad: The shared scratchpad.
        entry_types: Only include entries matching these types.
            If None, include all entries.
        agent_name_filter: If set, only include entries from this agent name
            (used in Step 4 so each reviewer sees only their own review).

    Returns:
        A prompt string starting with the query, followed by matching entries.
    """
    parts = [f"USER QUERY:\n{scratchpad.query}\n"]

    for entry in scratchpad.entries:
        if entry_types is not None and entry.entry_type not in entry_types:
            continue
        if agent_name_filter is not None and entry.agent_name != agent_name_filter:
            continue

        header = f"\n--- {entry.agent_name} ({entry.entry_type.value}) ---"
        parts.append(f"{header}\n{entry.content}\n")

    return "\n".join(parts)


def _context_step1(scratchpad: Scratchpad) -> str:
    """Step 1: Reasoner sees only the query."""
    return f"USER QUERY:\n{scratchpad.query}\n"


def _context_step2(scratchpad: Scratchpad) -> str:
    """Step 2: Reviewers see query + PLAN entries."""
    return _build_context(scratchpad, entry_types=[EntryType.PLAN])


def _context_step3(scratchpad: Scratchpad) -> str:
    """Step 3: Reasoner sees query + PLAN + CODE + CRITIQUE entries."""
    return _build_context(
        scratchpad,
        entry_types=[EntryType.PLAN, EntryType.CODE, EntryType.CRITIQUE],
    )


def _context_step4_coder(scratchpad: Scratchpad) -> str:
    """Step 4: Coder sees query + PLAN + own CODE review + REBUTTAL entries."""
    parts = [f"USER QUERY:\n{scratchpad.query}\n"]

    for entry in scratchpad.entries:
        if entry.entry_type == EntryType.PLAN:
            header = f"\n--- {entry.agent_name} ({entry.entry_type.value}) ---"
            parts.append(f"{header}\n{entry.content}\n")
        elif entry.entry_type == EntryType.CODE:
            header = f"\n--- {entry.agent_name} ({entry.entry_type.value}) ---"
            parts.append(f"{header}\n{entry.content}\n")
        elif entry.entry_type == EntryType.REBUTTAL:
            header = f"\n--- {entry.agent_name} ({entry.entry_type.value}) ---"
            parts.append(f"{header}\n{entry.content}\n")

    return "\n".join(parts)


def _context_step4_critic(scratchpad: Scratchpad) -> str:
    """Step 4: Critic sees query + PLAN + own CRITIQUE review + REBUTTAL entries."""
    parts = [f"USER QUERY:\n{scratchpad.query}\n"]

    for entry in scratchpad.entries:
        if entry.entry_type == EntryType.PLAN:
            header = f"\n--- {entry.agent_name} ({entry.entry_type.value}) ---"
            parts.append(f"{header}\n{entry.content}\n")
        elif entry.entry_type == EntryType.CRITIQUE:
            header = f"\n--- {entry.agent_name} ({entry.entry_type.value}) ---"
            parts.append(f"{header}\n{entry.content}\n")
        elif entry.entry_type == EntryType.REBUTTAL:
            header = f"\n--- {entry.agent_name} ({entry.entry_type.value}) ---"
            parts.append(f"{header}\n{entry.content}\n")

    return "\n".join(parts)


def _context_step5(scratchpad: Scratchpad) -> str:
    """Step 5: Synthesizer sees everything."""
    return _build_context(scratchpad, entry_types=None)


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def _create_agents(
    llm_client,
    model_assignments: Optional[Dict[str, str]] = None,
) -> Dict[str, Agent]:
    """Create the four pipeline agents with optional per-role model overrides."""
    models = model_assignments or {}
    return {
        "reasoner": create_reasoner(llm_client, models.get("reasoner")),
        "coder": create_coder(llm_client, models.get("coder")),
        "critic": create_critic(llm_client, models.get("critic")),
        "synthesizer": create_synthesizer(llm_client, models.get("synthesizer")),
    }


async def _run_agent_with_context(
    agent: Agent,
    context: str,
    entry_type: EntryType,
) -> Optional[ScratchpadEntry]:
    """Run a single agent with a custom context prompt (bypassing build_context_for_agent).

    Returns None on failure instead of raising.
    """
    try:
        response = await agent.llm_client.complete(
            prompt=context + f"\nAs the {agent.config.name}, provide your analysis:",
            model=agent.config.preferred_model,
            system_prompt=agent.config.system_prompt,
            temperature=agent.config.temperature,
            max_tokens=agent.config.max_tokens,
        )

        cost = getattr(response, "cost", 0.0)
        tokens = getattr(response, "total_tokens", 0)
        agent.total_cost += cost
        agent.total_tokens += tokens

        return ScratchpadEntry(
            agent_role=agent.config.role,
            agent_name=agent.config.name,
            content=response.content,
            entry_type=entry_type,
            model_used=getattr(response, "model", ""),
            cost=cost,
            tokens=tokens,
            timestamp=datetime.now(),
        )
    except Exception as exc:
        logger.warning("Agent '%s' failed: %s", agent.config.name, exc)
        return None


# ---------------------------------------------------------------------------
# Non-streaming pipeline
# ---------------------------------------------------------------------------

async def run_peer_review_pipeline(
    scratchpad: Scratchpad,
    llm_client,
    model_assignments: Optional[Dict[str, str]] = None,
) -> Scratchpad:
    """Run the peer review pipeline: produce, review, rebut, react, synthesize.

    Args:
        scratchpad: Scratchpad with the user query already set.
        llm_client: LLM client with an async ``complete()`` method.
        model_assignments: Optional mapping of role names to model ids,
            e.g. {"reasoner": "gpt-4", "coder": "gpt-4", "critic": "gpt-4",
                   "synthesizer": "gpt-4"}.

    Returns:
        The same scratchpad instance, now populated with all agent
        contributions that were completed before the budget ran out.
    """
    agents = _create_agents(llm_client, model_assignments)
    steps_completed = 0

    # --- Step 1: PRODUCE ---
    if scratchpad.is_over_budget():
        logger.warning("Budget exceeded before Step 1 (PRODUCE) -- skipping pipeline")
        return scratchpad

    logger.info("Step 1: PRODUCE -- Reasoner analyzes the problem")
    context = _context_step1(scratchpad)
    entry = await _run_agent_with_context(agents["reasoner"], context, EntryType.PLAN)
    if entry is not None:
        scratchpad.add_entry(entry)
        logger.info(
            "Reasoner contributed (%d tokens, $%.4f)",
            entry.tokens, entry.cost,
        )
    steps_completed += 1

    # --- Step 2: REVIEW (parallel) ---
    if scratchpad.is_over_budget():
        logger.warning("Budget exceeded before Step 2 (REVIEW) -- skipping to synthesis")
        return await _emergency_synthesize(scratchpad, agents["synthesizer"])

    logger.info("Step 2: REVIEW -- Coder and Critic review in parallel")
    review_context = _context_step2(scratchpad)

    coder_entry, critic_entry = await asyncio.gather(
        _run_agent_with_context(agents["coder"], review_context, EntryType.CODE),
        _run_agent_with_context(agents["critic"], review_context, EntryType.CRITIQUE),
    )

    if coder_entry is not None:
        scratchpad.add_entry(coder_entry)
        logger.info("Coder contributed (%d tokens, $%.4f)", coder_entry.tokens, coder_entry.cost)
    if critic_entry is not None:
        scratchpad.add_entry(critic_entry)
        logger.info("Critic contributed (%d tokens, $%.4f)", critic_entry.tokens, critic_entry.cost)
    steps_completed += 1

    # --- Step 3: REBUTTAL ---
    if scratchpad.is_over_budget():
        logger.warning("Budget exceeded before Step 3 (REBUTTAL) -- skipping to synthesis")
        return await _emergency_synthesize(scratchpad, agents["synthesizer"])

    logger.info("Step 3: REBUTTAL -- Reasoner responds to reviews")
    rebuttal_context = _context_step3(scratchpad)
    rebuttal_entry = await _run_agent_with_context(
        agents["reasoner"], rebuttal_context, EntryType.REBUTTAL,
    )
    if rebuttal_entry is not None:
        scratchpad.add_entry(rebuttal_entry)
        logger.info(
            "Reasoner rebuttal (%d tokens, $%.4f)",
            rebuttal_entry.tokens, rebuttal_entry.cost,
        )
    steps_completed += 1

    # --- Step 4: REACT (parallel) ---
    if scratchpad.is_over_budget():
        logger.warning("Budget exceeded before Step 4 (REACT) -- skipping to synthesis")
        return await _emergency_synthesize(scratchpad, agents["synthesizer"])

    logger.info("Step 4: REACT -- Coder and Critic react to rebuttal in parallel")
    coder_react_context = _context_step4_coder(scratchpad)
    critic_react_context = _context_step4_critic(scratchpad)

    coder_reaction, critic_reaction = await asyncio.gather(
        _run_agent_with_context(agents["coder"], coder_react_context, EntryType.REACTION),
        _run_agent_with_context(agents["critic"], critic_react_context, EntryType.REACTION),
    )

    if coder_reaction is not None:
        scratchpad.add_entry(coder_reaction)
        logger.info("Coder reaction (%d tokens, $%.4f)", coder_reaction.tokens, coder_reaction.cost)
    if critic_reaction is not None:
        scratchpad.add_entry(critic_reaction)
        logger.info("Critic reaction (%d tokens, $%.4f)", critic_reaction.tokens, critic_reaction.cost)
    steps_completed += 1

    # --- Step 5: SYNTHESIZE ---
    if scratchpad.is_over_budget():
        logger.warning("Budget exceeded before Step 5 (SYNTHESIZE) -- skipping synthesis")
        return scratchpad

    logger.info("Step 5: SYNTHESIZE -- Synthesizer combines all contributions")
    synth_context = _context_step5(scratchpad)
    synth_entry = await _run_agent_with_context(
        agents["synthesizer"], synth_context, EntryType.SYNTHESIS,
    )
    if synth_entry is not None:
        scratchpad.add_entry(synth_entry)
        logger.info(
            "Synthesizer contributed (%d tokens, $%.4f)",
            synth_entry.tokens, synth_entry.cost,
        )
    steps_completed += 1

    logger.info(
        "Peer review pipeline finished: %d steps completed, total cost $%.4f",
        steps_completed, scratchpad.total_cost,
    )
    return scratchpad


async def _emergency_synthesize(
    scratchpad: Scratchpad,
    synthesizer: Agent,
) -> Scratchpad:
    """Run the synthesizer on whatever partial results exist.

    Called when budget is exceeded mid-pipeline so we still produce
    a useful synthesis from available entries.
    """
    logger.info("Emergency synthesis on partial results (%d entries)", len(scratchpad.entries))
    context = _context_step5(scratchpad)
    entry = await _run_agent_with_context(synthesizer, context, EntryType.SYNTHESIS)
    if entry is not None:
        scratchpad.add_entry(entry)
    return scratchpad


# ---------------------------------------------------------------------------
# Streaming pipeline
# ---------------------------------------------------------------------------

async def stream_peer_review_pipeline(
    scratchpad: Scratchpad,
    llm_client,
    model_assignments: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream the peer review pipeline, yielding progress events.

    Event types emitted:
        step_start     -- a pipeline step is beginning
        agent_start    -- an agent within the step is starting
        agent_chunk    -- a token/chunk from an agent
        agent_done     -- an agent has finished
        step_complete  -- a pipeline step has finished
        done           -- the entire pipeline has finished

    For parallel steps (2, 4): agents are streamed sequentially
    to simplify SSE event ordering.

    Yields:
        dict events as described above.
    """
    agents = _create_agents(llm_client, model_assignments)

    # Define the 5-step plan.
    steps = [
        (1, "PRODUCE", [("reasoner", EntryType.PLAN, lambda: _context_step1(scratchpad))]),
        (2, "REVIEW", [
            ("coder", EntryType.CODE, lambda: _context_step2(scratchpad)),
            ("critic", EntryType.CRITIQUE, lambda: _context_step2(scratchpad)),
        ]),
        (3, "REBUTTAL", [("reasoner", EntryType.REBUTTAL, lambda: _context_step3(scratchpad))]),
        (4, "REACT", [
            ("coder", EntryType.REACTION, lambda: _context_step4_coder(scratchpad)),
            ("critic", EntryType.REACTION, lambda: _context_step4_critic(scratchpad)),
        ]),
        (5, "SYNTHESIZE", [("synthesizer", EntryType.SYNTHESIS, lambda: _context_step5(scratchpad))]),
    ]

    for step_num, step_name, agent_tasks in steps:
        # Budget gate
        if scratchpad.is_over_budget():
            logger.warning(
                "Budget exceeded before Step %d (%s) -- skipping",
                step_num, step_name,
            )
            # If we haven't synthesized yet, do an emergency synthesis
            if step_num <= 4:
                yield {"type": "step_start", "step": 5, "name": "SYNTHESIZE"}
                async for event in _stream_single_agent(
                    agents["synthesizer"],
                    _context_step5(scratchpad),
                    EntryType.SYNTHESIS,
                    scratchpad,
                ):
                    yield event
                yield {"type": "step_complete", "step": 5, "name": "SYNTHESIZE"}
            break

        yield {"type": "step_start", "step": step_num, "name": step_name}

        for role_key, entry_type, context_fn in agent_tasks:
            agent = agents[role_key]
            context = context_fn()

            async for event in _stream_single_agent(
                agent, context, entry_type, scratchpad,
            ):
                yield event

        yield {"type": "step_complete", "step": step_num, "name": step_name}

    # Terminal event
    yield {
        "type": "done",
        "total_cost": scratchpad.total_cost,
        "entries_count": len(scratchpad.entries),
        "scratchpad": scratchpad.to_dict(),
    }


async def _stream_single_agent(
    agent: Agent,
    context: str,
    entry_type: EntryType,
    scratchpad: Scratchpad,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream a single agent's output, yielding agent_start/chunk/done events.

    After streaming, estimates tokens from accumulated text and builds
    a ScratchpadEntry that is added to the scratchpad.
    """
    name = agent.config.name
    role = agent.config.role.value

    yield {"type": "agent_start", "agent": name, "role": role}

    accumulated = ""
    prompt = context + f"\nAs the {name}, provide your analysis:"

    async def _fallback_complete() -> str:
        response = await agent.llm_client.complete(
            prompt=prompt,
            model=agent.config.preferred_model,
            system_prompt=agent.config.system_prompt,
            temperature=agent.config.temperature,
            max_tokens=agent.config.max_tokens,
            stream=False,
        )
        return response.content or ""

    try:
        result = await agent.llm_client.complete(
            prompt=prompt,
            model=agent.config.preferred_model,
            system_prompt=agent.config.system_prompt,
            temperature=agent.config.temperature,
            max_tokens=agent.config.max_tokens,
            stream=True,
        )

        import inspect
        if inspect.isasyncgen(result):
            async for chunk in result:
                accumulated += chunk
                yield {"type": "agent_chunk", "agent": name, "content": chunk}
        else:
            accumulated = result.content
            if accumulated.strip():
                yield {"type": "agent_chunk", "agent": name, "content": accumulated}

    except Exception as exc:
        logger.warning("Streaming failed for %s: %s -- trying fallback", name, exc)
        try:
            accumulated = await _fallback_complete()
            if not accumulated.strip():
                raise ValueError(f"{name} returned empty content during fallback")
            yield {"type": "agent_chunk", "agent": name, "content": accumulated}
        except Exception:
            logger.exception("Fallback also failed for %s -- skipping", name)
            yield {"type": "agent_error", "agent": name, "error": f"{name} returned no content."}
            return

    if not accumulated.strip():
        logger.warning("%s returned no streamed content -- trying fallback", name)
        try:
            accumulated = await _fallback_complete()
            if not accumulated.strip():
                raise ValueError(f"{name} returned empty content during fallback")
            yield {"type": "agent_chunk", "agent": name, "content": accumulated}
        except Exception:
            logger.exception("Empty-output fallback also failed for %s -- skipping", name)
            yield {"type": "agent_error", "agent": name, "error": f"{name} returned no content."}
            return

    # Estimate tokens from text length: words * 1.3
    word_count = len(accumulated.split())
    estimated_tokens = int(word_count * 1.3)
    estimated_cost = 0.0  # Cost tracking from streaming is approximate

    # Build entry and add to scratchpad
    if accumulated:
        entry = ScratchpadEntry(
            agent_role=agent.config.role,
            agent_name=name,
            content=accumulated,
            entry_type=entry_type,
            model_used=agent.config.preferred_model or "",
            cost=estimated_cost,
            tokens=estimated_tokens,
            timestamp=datetime.now(),
        )
        scratchpad.add_entry(entry)

    yield {
        "type": "agent_done",
        "agent": name,
        "cost": estimated_cost,
        "tokens": estimated_tokens,
    }
