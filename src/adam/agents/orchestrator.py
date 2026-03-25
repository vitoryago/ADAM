"""Multi-agent orchestrator with pattern selection.

The Orchestrator is the main entry point for multi-agent reasoning. It decides
whether a query warrants multi-agent processing, selects the best orchestration
pattern (sequential or debate), and manages the full execution flow.
"""

import logging
import re
from enum import Enum
from typing import Optional, Dict, Any, AsyncGenerator

from .scratchpad import Scratchpad, EntryType
from .cost_guard import CostGuard
from .synthesis import FallbackSynthesizer
from .patterns.sequential import run_sequential_pipeline, stream_sequential_pipeline
from .patterns.debate import run_debate_pipeline, stream_debate_pipeline

logger = logging.getLogger(__name__)


class OrchestrationPattern(Enum):
    SEQUENTIAL = "sequential"  # Reasoner -> Coder -> Critic -> Synthesizer
    DEBATE = "debate"          # Two perspectives -> Reconciler
    SINGLE = "single"          # No multi-agent, just one model
    PEER_REVIEW = "peer_review"  # Produce -> Review -> Rebut -> React -> Synthesize


# Keywords that indicate a query is complex enough for multi-agent reasoning.
_MULTI_AGENT_TRIGGERS = re.compile(
    r"\b("
    r"design|architect|implement|compare|debug|refactor|optimize|"
    r"build|create\s+a\s+(?:system|service|api|app|application|framework)|"
    r"pros\s+and\s+cons|trade[\s-]?offs?|versus|vs\.?|"
    r"analyze|review|migrate|scale|secure|"
    r"microservices?|distributed|full[\s-]?stack"
    r")\b",
    re.IGNORECASE,
)

# Keywords that specifically signal a comparison / debate scenario.
_DEBATE_TRIGGERS = re.compile(
    r"\b("
    r"compare|vs\.?|versus|pros\s+and\s+cons|trade[\s-]?offs?|"
    r"which\s+is\s+better|should\s+i\s+(?:use|choose|pick)|"
    r"differences?\s+between|advantages?\s+and\s+disadvantages?"
    r")\b",
    re.IGNORECASE,
)


class Orchestrator:
    """Main entry point for multi-agent reasoning.

    The orchestrator decides when to engage the multi-agent system, selects the
    appropriate pattern, and manages the full execution flow including budget
    enforcement and fallback synthesis.
    """

    def __init__(self, llm_client, memory_service=None):
        self.llm_client = llm_client
        self.memory_service = memory_service

    def should_use_multi_agent(self, query: str, complexity: str = "simple") -> bool:
        """Decide if multi-agent reasoning is warranted for this query.

        Returns True when the query contains trigger keywords that indicate
        architectural, implementation, comparison, or debugging work. Simple
        factual questions are handled by a single model call.

        Args:
            query: The user's question or request.
            complexity: Hint from the caller -- "complex" forces True.

        Returns:
            True if the query should go through multi-agent reasoning.
        """
        if complexity == "complex":
            return True
        if complexity == "simple" and not _MULTI_AGENT_TRIGGERS.search(query):
            return False
        return bool(_MULTI_AGENT_TRIGGERS.search(query))

    def select_pattern(self, query: str) -> OrchestrationPattern:
        """Select the best orchestration pattern for this query.

        - DEBATE is chosen for comparison / trade-off queries where two
          independent perspectives add value.
        - SEQUENTIAL is the default for everything else that qualifies for
          multi-agent processing.

        Args:
            query: The user's question or request.

        Returns:
            The chosen OrchestrationPattern.
        """
        if _DEBATE_TRIGGERS.search(query):
            return OrchestrationPattern.DEBATE
        return OrchestrationPattern.SEQUENTIAL

    async def orchestrate(
        self,
        query: str,
        budget: float = 0.50,
        model_assignments: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Run multi-agent reasoning on a query.

        Creates a scratchpad, selects the pattern, runs the appropriate
        pipeline, and extracts the final response. If synthesis is missing
        the FallbackSynthesizer kicks in.

        Args:
            query: The user's question or request.
            budget: Maximum spend for this orchestration run.
            model_assignments: Optional per-agent model overrides.

        Returns:
            Dict with keys: response, pattern, total_cost, agents_used, scratchpad.
        """
        pattern = self.select_pattern(query)
        scratchpad = Scratchpad(query=query, budget=budget)
        cost_guard = CostGuard(budget=budget)

        logger.info(
            "Orchestrating with pattern=%s, budget=$%.2f",
            pattern.value,
            budget,
        )

        if pattern == OrchestrationPattern.DEBATE:
            scratchpad = await run_debate_pipeline(
                scratchpad, self.llm_client, model_assignments=model_assignments,
            )
        else:
            scratchpad = await run_sequential_pipeline(
                scratchpad, self.llm_client, model_assignments=model_assignments,
            )

        # Sync cost guard with scratchpad spend
        for entry in scratchpad.entries:
            cost_guard.record_spend(entry.agent_name, entry.cost)

        # Extract final response
        response = self._extract_response(scratchpad)

        agents_used = [e.agent_name for e in scratchpad.entries]

        return {
            "response": response,
            "pattern": pattern.value,
            "total_cost": scratchpad.total_cost,
            "agents_used": agents_used,
            "scratchpad": scratchpad.to_dict(),
        }

    async def orchestrate_stream(
        self,
        query: str,
        budget: float = 0.50,
        model_assignments: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream the orchestration with progress events.

        Yields events as they happen so callers can display real-time progress.
        The first event is always ``orchestration_start`` with the chosen
        pattern. Subsequent events are forwarded from the underlying pattern
        stream. The last event is always ``done``.

        Args:
            query: The user's question or request.
            budget: Maximum spend for this orchestration run.
            model_assignments: Optional per-agent model overrides.

        Yields:
            Dict events -- see pattern streams for the full event vocabulary.
        """
        pattern = self.select_pattern(query)
        scratchpad = Scratchpad(query=query, budget=budget)

        yield {
            "type": "orchestration_start",
            "pattern": pattern.value,
        }

        if pattern == OrchestrationPattern.DEBATE:
            stream = stream_debate_pipeline(
                scratchpad, self.llm_client, model_assignments=model_assignments,
            )
        else:
            stream = stream_sequential_pipeline(
                scratchpad, self.llm_client, model_assignments=model_assignments,
            )

        async for event in stream:
            yield event

    def _extract_response(self, scratchpad: Scratchpad) -> str:
        """Extract the final response from a completed scratchpad.

        Prefers the synthesis entry. Falls back to FallbackSynthesizer when
        no synthesis entry is present.
        """
        synthesis_entries = scratchpad.get_entries_by_type(EntryType.SYNTHESIS)
        if synthesis_entries:
            return synthesis_entries[-1].content

        logger.warning("No synthesis entry found -- using FallbackSynthesizer")
        return FallbackSynthesizer.synthesize(scratchpad)
