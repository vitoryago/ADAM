"""Session configuration for Deep Discussion Mode.

Provides ``SessionConfig`` (a dataclass with smart defaults) and helper
constants for the available models a user can choose from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Smart defaults
# ---------------------------------------------------------------------------


def get_smart_defaults() -> Dict[str, str]:
    """Return the default model-per-role mapping.

    The four canonical roles map to best-in-class models that cover
    reasoning, coding, critique, and synthesis respectively.

    Returns:
        A dict with keys ``reasoner``, ``coder``, ``critic``, ``synthesizer``.
    """
    return {
        "reasoner": "grok-4.20-multi-agent-0309",
        "coder": "claude-opus-4-6",
        "critic": "gpt-5.4-2026-03-05",
        "synthesizer": "claude-sonnet-4-6",
    }


# ---------------------------------------------------------------------------
# Available models catalogue
# ---------------------------------------------------------------------------

#: All models the user may assign to any role, keyed by human-readable label.
#: Grouped loosely by provider for display purposes.
AVAILABLE_MODELS: Dict[str, str] = {
    # Anthropic
    "Claude Opus 4.6": "claude-opus-4-6",
    "Claude Sonnet 4.6": "claude-sonnet-4-6",
    "Claude Haiku 3.5": "claude-haiku-3-5",
    # OpenAI
    "GPT-5.4 (2026-03-05)": "gpt-5.4-2026-03-05",
    "GPT-4o": "gpt-4o",
    "o3": "o3",
    # xAI
    "Grok 4.20 Multi-Agent": "grok-4.20-multi-agent-0309",
    "Grok 3": "grok-3",
    # Google
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 2.0 Flash": "gemini-2.0-flash",
}


# ---------------------------------------------------------------------------
# SessionConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class SessionConfig:
    """Configuration for a single Deep Discussion session.

    Attributes:
        question: The user's question or topic to discuss.
        pattern: Orchestration pattern — one of ``"sequential"``,
            ``"debate"``, or ``"peer_review"``.
        model_assignments: Maps canonical role names to model IDs.
            Defaults to :func:`get_smart_defaults` when not provided.
        budget: Maximum spend in USD for the session. Defaults to ``2.0``.
        conversation_id: Optional ID of an existing ADAM conversation to
            associate this session with.
        conversation_context: Optional prior conversation text injected as
            extra context for all agents.
    """

    question: str
    pattern: str  # "sequential" | "debate" | "peer_review"
    model_assignments: Dict[str, str] = field(default_factory=dict)
    budget: float = 2.0
    conversation_id: Optional[str] = None
    conversation_context: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.model_assignments:
            self.model_assignments = get_smart_defaults()
