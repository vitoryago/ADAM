"""Deep Discussion session lifecycle management.

A ``DeepDiscussionSession`` wraps a :class:`~.config.SessionConfig` and
orchestrates the chosen multi-agent pattern from start to finish, tracking
status, cost, and results throughout.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional

from .config import SessionConfig

logger = logging.getLogger(__name__)


class DeepDiscussionSession:
    """Lifecycle manager for a single Deep Discussion session.

    Attributes:
        id: Unique session identifier (UUID4 string).
        config: The session's :class:`~.config.SessionConfig`.
        status: One of ``"configuring"``, ``"running"``, ``"completed"``,
            ``"failed"``.
        result: Final synthesised response (set on completion).
        scratchpad_data: Serialised scratchpad dict (set on completion).
        total_cost: Accumulated cost in USD.
        created_at: Timestamp of session creation.
        completed_at: Timestamp of session completion (or ``None``).
    """

    def __init__(self, config: SessionConfig) -> None:
        self.id: str = str(uuid.uuid4())
        self.config: SessionConfig = config
        self.status: str = "configuring"
        self.result: Optional[str] = None
        self.scratchpad_data: Optional[Dict[str, Any]] = None
        self.total_cost: float = 0.0
        self.created_at: datetime = datetime.now()
        self.completed_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def update_config(
        self,
        model_assignments: Optional[Dict[str, str]] = None,
        pattern: Optional[str] = None,
        budget: Optional[float] = None,
    ) -> None:
        """Update mutable fields on the session config in-place.

        Any argument that is ``None`` is left unchanged.

        Args:
            model_assignments: New role-to-model mapping.
            pattern: New orchestration pattern.
            budget: New budget ceiling in USD.
        """
        if model_assignments is not None:
            self.config.model_assignments = model_assignments
        if pattern is not None:
            self.config.pattern = pattern
        if budget is not None:
            self.config.budget = budget

    def get_pattern_model_assignments(self) -> Dict[str, str]:
        """Map canonical role keys to the slot names expected by each pattern.

        - **sequential** — pass-through: ``{reasoner, coder, critic, synthesizer}``
        - **debate** — remapped: ``{debater_a, debater_b, reconciler}`` where
          ``debater_a`` uses the *reasoner* model, ``debater_b`` uses the
          *coder* model, and ``reconciler`` uses the *critic* model.
        - **peer_review** — pass-through: ``{reasoner, coder, critic, synthesizer}``

        Returns:
            A dict ready to pass directly as ``model_assignments`` to the
            appropriate pattern pipeline.
        """
        assignments = self.config.model_assignments

        if self.config.pattern == "debate":
            return {
                "debater_a": assignments.get("reasoner", ""),
                "debater_b": assignments.get("coder", ""),
                "reconciler": assignments.get("critic", ""),
            }

        # sequential and peer_review both use the canonical four roles.
        return {
            "reasoner": assignments.get("reasoner", ""),
            "coder": assignments.get("coder", ""),
            "critic": assignments.get("critic", ""),
            "synthesizer": assignments.get("synthesizer", ""),
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run(self, llm_client: Any) -> str:
        """Run the session synchronously (non-streaming).

        Delegates to the correct pattern pipeline and stores the result.

        Args:
            llm_client: An LLM client with an async ``complete()`` method.

        Returns:
            The final synthesised response string.

        Raises:
            Exception: Re-raises any exception from the pattern pipeline
                after setting ``status`` to ``"failed"``.
        """
        from adam.agents.scratchpad import Scratchpad
        from adam.agents.patterns.sequential import run_sequential_pipeline
        from adam.agents.patterns.debate import run_debate_pipeline
        from adam.agents.patterns.peer_review import run_peer_review_pipeline
        from adam.agents.synthesis import FallbackSynthesizer
        from adam.agents.scratchpad import EntryType

        self.status = "running"
        pattern_assignments = self.get_pattern_model_assignments()
        scratchpad = Scratchpad(
            query=self.config.question,
            budget=self.config.budget,
        )

        try:
            if self.config.pattern == "sequential":
                scratchpad = await run_sequential_pipeline(
                    scratchpad, llm_client, model_assignments=pattern_assignments
                )
            elif self.config.pattern == "debate":
                scratchpad = await run_debate_pipeline(
                    scratchpad, llm_client, model_assignments=pattern_assignments
                )
            elif self.config.pattern == "peer_review":
                scratchpad = await run_peer_review_pipeline(
                    scratchpad, llm_client, model_assignments=pattern_assignments
                )
            else:
                raise ValueError(f"Unknown pattern: {self.config.pattern!r}")

            # Extract final answer
            synthesis_entries = scratchpad.get_entries_by_type(EntryType.SYNTHESIS)
            if synthesis_entries:
                response = synthesis_entries[-1].content
            else:
                logger.warning("No synthesis entry — using FallbackSynthesizer")
                response = FallbackSynthesizer.synthesize(scratchpad)

            self.result = response
            self.scratchpad_data = scratchpad.to_dict()
            self.total_cost = scratchpad.total_cost
            self.status = "completed"
            self.completed_at = datetime.now()
            return response

        except Exception:
            self.status = "failed"
            self.completed_at = datetime.now()
            logger.exception("Session %s failed", self.id)
            raise

    async def run_stream(
        self, llm_client: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream the session, yielding progress events.

        Wraps the underlying pattern stream with session-level envelope events:

        - ``session_start`` — emitted first with ``session_id`` and ``pattern``.
        - All pattern events are forwarded as-is.
        - ``session_complete`` — emitted last with ``session_id``, ``total_cost``,
          and ``result`` (the synthesised text).
        - ``session_error`` — emitted instead of ``session_complete`` on failure,
          with an ``error`` key.

        Args:
            llm_client: An LLM client with an async ``complete()`` method.

        Yields:
            Dict events describing session and pattern progress.
        """
        from adam.agents.scratchpad import Scratchpad, EntryType
        from adam.agents.patterns.sequential import stream_sequential_pipeline
        from adam.agents.patterns.debate import stream_debate_pipeline
        from adam.agents.patterns.peer_review import stream_peer_review_pipeline
        from adam.agents.synthesis import FallbackSynthesizer

        self.status = "running"
        pattern_assignments = self.get_pattern_model_assignments()
        scratchpad = Scratchpad(
            query=self.config.question,
            budget=self.config.budget,
        )

        yield {
            "type": "session_start",
            "session_id": self.id,
            "pattern": self.config.pattern,
        }

        try:
            if self.config.pattern == "sequential":
                stream = stream_sequential_pipeline(
                    scratchpad, llm_client, model_assignments=pattern_assignments
                )
            elif self.config.pattern == "debate":
                stream = stream_debate_pipeline(
                    scratchpad, llm_client, model_assignments=pattern_assignments
                )
            elif self.config.pattern == "peer_review":
                stream = stream_peer_review_pipeline(
                    scratchpad, llm_client, model_assignments=pattern_assignments
                )
            else:
                raise ValueError(f"Unknown pattern: {self.config.pattern!r}")

            async for event in stream:
                yield event

            # Capture final state
            synthesis_entries = scratchpad.get_entries_by_type(EntryType.SYNTHESIS)
            if synthesis_entries:
                response = synthesis_entries[-1].content
            else:
                logger.warning("No synthesis entry — using FallbackSynthesizer")
                response = FallbackSynthesizer.synthesize(scratchpad)

            self.result = response
            self.scratchpad_data = scratchpad.to_dict()
            self.total_cost = scratchpad.total_cost
            self.status = "completed"
            self.completed_at = datetime.now()

            yield {
                "type": "session_complete",
                "session_id": self.id,
                "total_cost": self.total_cost,
                "result": self.result,
            }

        except Exception as exc:
            self.status = "failed"
            self.completed_at = datetime.now()
            logger.exception("Session %s stream failed", self.id)
            yield {
                "type": "session_error",
                "session_id": self.id,
                "error": str(exc),
            }
