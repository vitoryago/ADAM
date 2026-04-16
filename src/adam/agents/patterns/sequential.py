"""Sequential pipeline: Reasoner -> Coder -> Critic -> Synthesizer.

This pattern runs four specialized agents in sequence. Each agent reads
from the shared scratchpad (respecting visibility rules) and writes its
contribution back. Budget is checked between agents to avoid overspending.
"""

import logging
from typing import AsyncGenerator, Dict, Any, Optional
from datetime import datetime

from ..base import Agent
from ..scratchpad import Scratchpad, ScratchpadEntry, AgentRole
from ..roles import create_reasoner, create_coder, create_critic, create_synthesizer

logger = logging.getLogger(__name__)

# The fixed agent order for the sequential pipeline.
_PIPELINE_ORDER = [
    (AgentRole.REASONER, create_reasoner),
    (AgentRole.CODER, create_coder),
    (AgentRole.CRITIC, create_critic),
    (AgentRole.SYNTHESIZER, create_synthesizer),
]


def _build_agents(
    llm_client,
    model_assignments: Optional[Dict[AgentRole, str]] = None,
) -> list:
    """Create the four pipeline agents with optional per-role model overrides."""
    models = model_assignments or {}
    return [
        factory(llm_client, models.get(role))
        for role, factory in _PIPELINE_ORDER
    ]


async def run_sequential_pipeline(
    scratchpad: Scratchpad,
    llm_client,
    model_assignments: Optional[Dict[AgentRole, str]] = None,
) -> Scratchpad:
    """Run the sequential pipeline: Reasoner -> Coder -> Critic -> Synthesizer.

    Args:
        scratchpad: Scratchpad with the user query already set.
        llm_client: LLM client with an async ``complete()`` method.
        model_assignments: Optional mapping of AgentRole -> model name
            so that each agent can use a different provider/model.

    Returns:
        The same scratchpad instance, now populated with all agent
        contributions that were completed before the budget ran out.
    """
    agents = _build_agents(llm_client, model_assignments)
    agents_run = 0

    for agent in agents:
        # Budget gate -- check *before* calling the agent.
        if scratchpad.is_over_budget():
            logger.warning(
                "Budget exceeded ($%.4f / $%.4f) before %s -- stopping pipeline",
                scratchpad.total_cost,
                scratchpad.budget,
                agent.config.name,
            )
            break

        try:
            entry = await agent.think(scratchpad)
            scratchpad.add_entry(entry)
            agents_run += 1
            logger.info(
                "%s contributed (%d tokens, $%.4f)",
                agent.config.name,
                entry.tokens,
                entry.cost,
            )
        except Exception:
            logger.exception("Agent %s failed", agent.config.name)
            # Do not crash the pipeline -- continue to next agent.

    logger.info(
        "Sequential pipeline finished: %d agents ran, total cost $%.4f",
        agents_run,
        scratchpad.total_cost,
    )
    return scratchpad


async def stream_sequential_pipeline(
    scratchpad: Scratchpad,
    llm_client,
    model_assignments: Optional[Dict[AgentRole, str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream the sequential pipeline, yielding progress events.

    Event types emitted (in order, per agent):
        {"type": "agent_start",  "agent": name, "role": role}
        {"type": "agent_chunk",  "agent": name, "content": chunk}
        {"type": "agent_done",   "agent": name, "cost": float, "tokens": int}

    When the budget is exceeded before an agent starts:
        {"type": "budget_exceeded", "agent": name}

    Terminal event (always the last yield):
        {"type": "done", "total_cost": float, "agents_used": int,
         "scratchpad": dict}

    If ``think_stream()`` raises, the function transparently falls back to
    ``think()`` so the pipeline keeps moving.
    """
    agents = _build_agents(llm_client, model_assignments)
    agents_used = 0

    for agent in agents:
        name = agent.config.name
        role = agent.config.role.value

        # Budget gate
        if scratchpad.is_over_budget():
            logger.warning(
                "Budget exceeded before %s -- skipping", name,
            )
            yield {"type": "budget_exceeded", "agent": name}
            break

        yield {"type": "agent_start", "agent": name, "role": role}

        # Try streaming first; fall back to non-streaming on any error.
        accumulated = ""
        used_fallback = False

        try:
            async for chunk in agent.think_stream(scratchpad):
                accumulated += chunk
                yield {
                    "type": "agent_chunk",
                    "agent": name,
                    "content": chunk,
                }
        except Exception as exc:
            logger.error(
                "Streaming failed for %s (%s), falling back to think()",
                name,
                exc,
            )
            used_fallback = True
            try:
                fallback_entry = await agent.think(scratchpad)
                accumulated = fallback_entry.content
                yield {
                    "type": "agent_chunk",
                    "agent": name,
                    "content": accumulated,
                }
            except Exception:
                logger.exception(
                    "Fallback think() also failed for %s -- skipping", name,
                )
                yield {
                    "type": "agent_error",
                    "agent": name,
                    "error": f"{name} failed during streaming and fallback execution.",
                }
                continue

        # Build a scratchpad entry from the accumulated content.
        entry = ScratchpadEntry(
            agent_role=agent.config.role,
            agent_name=name,
            content=accumulated,
            entry_type=agent.config.entry_type,
            model_used=agent.config.preferred_model or "",
            cost=agent.total_cost,
            tokens=agent.total_tokens,
            timestamp=datetime.now(),
        )
        scratchpad.add_entry(entry)
        agents_used += 1

        yield {
            "type": "agent_done",
            "agent": name,
            "cost": agent.total_cost,
            "tokens": agent.total_tokens,
        }

    # Terminal event
    yield {
        "type": "done",
        "total_cost": scratchpad.total_cost,
        "agents_used": agents_used,
        "scratchpad": scratchpad.to_dict(),
    }
