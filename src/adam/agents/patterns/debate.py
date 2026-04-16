"""Debate pattern: point, counterpoint, rebuttal, reconcile.

Perspective A presents an opening position. Perspective B directly responds
to that position with a competing case. Perspective A then returns with a
rebuttal. Finally, a reconciler synthesizes the strongest arguments into a
single answer.
"""

import inspect
import logging
from datetime import datetime
from typing import AsyncGenerator, Dict, Optional

from ..base import Agent, AgentConfig
from ..scratchpad import AgentRole, EntryType, Scratchpad, ScratchpadEntry

logger = logging.getLogger(__name__)

# --- Debate System Prompts ---

DEBATER_A_PROMPT = (
    "You are Perspective A in a structured debate. "
    "Take a clear position, argue for it directly, and make your reasoning explicit. "
    "Be specific about trade-offs, constraints, and why your position is strong."
)

DEBATER_B_PROMPT = (
    "You are Perspective B in a structured debate. "
    "Respond directly to Perspective A. Challenge weak assumptions, offer a competing approach, "
    "and explain why your position is stronger."
)

REBUTTAL_PROMPT = (
    "You are Perspective A returning for a rebuttal. "
    "Respond directly to Perspective B's strongest points. Defend what still holds, concede what is valid, "
    "and sharpen the case for your position."
)

RECONCILER_PROMPT = (
    "You are a Reconciler. You have read the debate on the same problem. "
    "Your job is to synthesize the best elements into a single, coherent solution. "
    "Where the perspectives agree, reinforce the conclusion. "
    "Where they disagree, weigh the arguments and choose the stronger one, "
    "explaining why. Present a unified answer."
)


def _create_debater_a(llm_client, preferred_model: Optional[str] = None) -> Agent:
    """Create the opening debater."""
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
    """Create the counterpoint debater."""
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


def _create_rebuttal_agent(llm_client, preferred_model: Optional[str] = None) -> Agent:
    """Create the rebuttal turn for Perspective A."""
    config = AgentConfig(
        role=AgentRole.REASONER,
        name="Perspective A Rebuttal",
        system_prompt=REBUTTAL_PROMPT,
        entry_type=EntryType.REBUTTAL,
        preferred_model=preferred_model,
        temperature=0.5,
        max_tokens=1800,
    )
    return Agent(config, llm_client)


def _create_reconciler(llm_client, preferred_model: Optional[str] = None) -> Agent:
    """Create the reconciler agent that synthesizes the debate."""
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


def _build_context(
    scratchpad: Scratchpad,
    entry_types: Optional[list[EntryType]] = None,
) -> str:
    """Build a custom debate context from selected scratchpad entries."""
    parts = [f"USER QUERY:\n{scratchpad.query}\n"]

    for entry in scratchpad.entries:
        if entry_types is not None and entry.entry_type not in entry_types:
            continue
        header = f"\n--- {entry.agent_name} ({entry.entry_type.value}) ---"
        parts.append(f"{header}\n{entry.content}\n")

    return "\n".join(parts)


def _context_opening(scratchpad: Scratchpad) -> str:
    """Opening turn sees only the query."""
    return f"USER QUERY:\n{scratchpad.query}\n"


def _context_counterpoint(scratchpad: Scratchpad) -> str:
    """Counterpoint sees the opening position."""
    return _build_context(scratchpad, entry_types=[EntryType.PLAN])


def _context_rebuttal(scratchpad: Scratchpad) -> str:
    """Rebuttal sees both opening positions."""
    return _build_context(scratchpad, entry_types=[EntryType.PLAN])


def _context_reconcile(scratchpad: Scratchpad) -> str:
    """Reconciler sees both positions plus any rebuttal."""
    return _build_context(scratchpad, entry_types=[EntryType.PLAN, EntryType.REBUTTAL])


async def _run_agent_with_context(
    agent: Agent,
    context: str,
) -> Optional[ScratchpadEntry]:
    """Run a single debate turn with a custom context string."""
    try:
        response = await agent.llm_client.complete(
            prompt=context + f"\nAs {agent.config.name}, provide your argument:",
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
            entry_type=agent.config.entry_type,
            model_used=getattr(response, "model", ""),
            cost=cost,
            tokens=tokens,
            timestamp=datetime.now(),
        )
    except Exception as exc:
        logger.warning("Debate turn '%s' failed: %s", agent.config.name, exc)
        return None


async def _stream_agent_turn(
    agent: Agent,
    context: str,
    prompt_suffix: str,
) -> tuple[str, bool]:
    """Stream a debate turn and return the full content plus success flag."""
    try:
        result = await agent.llm_client.complete(
            prompt=context + prompt_suffix,
            model=agent.config.preferred_model,
            system_prompt=agent.config.system_prompt,
            temperature=agent.config.temperature,
            max_tokens=agent.config.max_tokens,
            stream=True,
        )

        chunks: list[str] = []
        if inspect.isasyncgen(result):
            async for chunk in result:
                chunks.append(chunk)
        else:
            chunks.append(result.content)

        return "".join(chunks), True
    except Exception as exc:
        logger.warning("Streaming debate turn '%s' failed: %s", agent.config.name, exc)
        return "", False


async def run_debate_pipeline(
    scratchpad: Scratchpad,
    llm_client,
    model_assignments: Optional[Dict[str, str]] = None,
) -> Scratchpad:
    """Run the debate pattern: opening, counterpoint, rebuttal, reconcile."""
    model_assignments = model_assignments or {}

    if scratchpad.is_over_budget():
        logger.warning("Budget exceeded before debate — skipping.")
        return scratchpad

    debater_a = _create_debater_a(llm_client, model_assignments.get("debater_a"))
    debater_b = _create_debater_b(llm_client, model_assignments.get("debater_b"))
    rebuttal = _create_rebuttal_agent(llm_client, model_assignments.get("debater_a"))

    entry_a = await _run_agent_with_context(debater_a, _context_opening(scratchpad))
    if entry_a is not None:
        scratchpad.add_entry(entry_a)

    if scratchpad.is_over_budget():
        logger.warning("Budget exceeded after Perspective A — stopping debate.")
        return scratchpad

    entry_b = await _run_agent_with_context(debater_b, _context_counterpoint(scratchpad))
    if entry_b is not None:
        scratchpad.add_entry(entry_b)

    if entry_a is None and entry_b is None:
        logger.error("Both debate positions failed — no reconciliation possible.")
        return scratchpad

    if scratchpad.is_over_budget():
        logger.warning("Budget exceeded after Perspective B — skipping rebuttal and reconciler.")
        return scratchpad

    if entry_a is not None and entry_b is not None:
        rebuttal_entry = await _run_agent_with_context(rebuttal, _context_rebuttal(scratchpad))
        if rebuttal_entry is not None:
            scratchpad.add_entry(rebuttal_entry)

    if scratchpad.is_over_budget():
        logger.warning("Budget exceeded after rebuttal — skipping reconciler.")
        return scratchpad

    reconciler = _create_reconciler(llm_client, model_assignments.get("reconciler"))
    reconciler_entry = await _run_agent_with_context(reconciler, _context_reconcile(scratchpad))
    if reconciler_entry is not None:
        scratchpad.add_entry(reconciler_entry)

    return scratchpad


async def stream_debate_pipeline(
    scratchpad: Scratchpad,
    llm_client,
    model_assignments: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[Dict, None]:
    """Stream the debate pattern, yielding progress events."""
    model_assignments = model_assignments or {}

    yield {
        "type": "debate_start",
        "debaters": ["Perspective A", "Perspective B", "Perspective A Rebuttal"],
    }

    if scratchpad.is_over_budget():
        yield {"type": "done", "reason": "budget_exceeded"}
        return

    debater_a = _create_debater_a(llm_client, model_assignments.get("debater_a"))
    debater_b = _create_debater_b(llm_client, model_assignments.get("debater_b"))
    rebuttal = _create_rebuttal_agent(llm_client, model_assignments.get("debater_a"))
    reconciler = _create_reconciler(llm_client, model_assignments.get("reconciler"))

    yield {"type": "agent_start", "agent": "Perspective A", "role": "reasoner"}
    full_a, success_a = await _stream_agent_turn(
        debater_a,
        _context_opening(scratchpad),
        "\nAs Perspective A, provide your argument:",
    )
    if full_a:
        for chunk in full_a.splitlines(keepends=True) or [full_a]:
            yield {"type": "agent_chunk", "agent": "Perspective A", "content": chunk}
        scratchpad.add_entry(
            ScratchpadEntry(
                agent_role=AgentRole.REASONER,
                agent_name="Perspective A",
                content=full_a,
                entry_type=EntryType.PLAN,
            )
        )
        yield {"type": "agent_done", "agent": "Perspective A"}
    elif not success_a:
        yield {"type": "agent_error", "agent": "Perspective A", "error": "Perspective A failed."}

    if scratchpad.is_over_budget():
        yield {"type": "done", "reason": "budget_exceeded"}
        return

    yield {"type": "agent_start", "agent": "Perspective B", "role": "reasoner"}
    full_b, success_b = await _stream_agent_turn(
        debater_b,
        _context_counterpoint(scratchpad),
        "\nAs Perspective B, provide your argument:",
    )
    if full_b:
        for chunk in full_b.splitlines(keepends=True) or [full_b]:
            yield {"type": "agent_chunk", "agent": "Perspective B", "content": chunk}
        scratchpad.add_entry(
            ScratchpadEntry(
                agent_role=AgentRole.REASONER,
                agent_name="Perspective B",
                content=full_b,
                entry_type=EntryType.PLAN,
            )
        )
        yield {"type": "agent_done", "agent": "Perspective B"}
    elif not success_b:
        yield {"type": "agent_error", "agent": "Perspective B", "error": "Perspective B failed."}

    if not full_a and not full_b:
        yield {"type": "done", "reason": "all_debaters_failed"}
        return

    if scratchpad.is_over_budget():
        yield {"type": "done", "reason": "budget_exceeded"}
        return

    if full_a and full_b:
        yield {
            "type": "agent_start",
            "agent": "Perspective A Rebuttal",
            "role": "reasoner",
            "badge": "REBUTTAL",
        }
        rebuttal_content, rebuttal_success = await _stream_agent_turn(
            rebuttal,
            _context_rebuttal(scratchpad),
            "\nAs Perspective A Rebuttal, provide your rebuttal:",
        )
        if rebuttal_content:
            for chunk in rebuttal_content.splitlines(keepends=True) or [rebuttal_content]:
                yield {
                    "type": "agent_chunk",
                    "agent": "Perspective A Rebuttal",
                    "content": chunk,
                }
            scratchpad.add_entry(
                ScratchpadEntry(
                    agent_role=AgentRole.REASONER,
                    agent_name="Perspective A Rebuttal",
                    content=rebuttal_content,
                    entry_type=EntryType.REBUTTAL,
                )
            )
            yield {"type": "agent_done", "agent": "Perspective A Rebuttal"}
        elif not rebuttal_success:
            yield {
                "type": "agent_error",
                "agent": "Perspective A Rebuttal",
                "error": "Rebuttal failed.",
            }

    if scratchpad.is_over_budget():
        yield {"type": "done", "reason": "budget_exceeded"}
        return

    yield {"type": "reconciliation_start"}
    yield {"type": "agent_start", "agent": "Reconciler", "role": "synthesizer"}
    full_reconciler, success_reconciler = await _stream_agent_turn(
        reconciler,
        _context_reconcile(scratchpad),
        "\nAs Reconciler, produce the final synthesis:",
    )
    if full_reconciler:
        for chunk in full_reconciler.splitlines(keepends=True) or [full_reconciler]:
            yield {"type": "agent_chunk", "agent": "Reconciler", "content": chunk}
        scratchpad.add_entry(
            ScratchpadEntry(
                agent_role=AgentRole.SYNTHESIZER,
                agent_name="Reconciler",
                content=full_reconciler,
                entry_type=EntryType.SYNTHESIS,
            )
        )
        yield {"type": "agent_done", "agent": "Reconciler"}
        yield {"type": "done", "reason": "complete"}
        return

    if not success_reconciler:
        yield {"type": "agent_error", "agent": "Reconciler", "error": "Reconciler failed."}
    yield {"type": "done", "reason": "reconciler_failed"}
