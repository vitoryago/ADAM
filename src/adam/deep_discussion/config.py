"""Session configuration for Deep Discussion Mode.

Provides ``SessionConfig`` (a dataclass with smart defaults) and helper
constants for the available models a user can choose from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from adam.llm.config import LLMConfig

logger = logging.getLogger(__name__)


def _get_local_models() -> list:
    """Return available local models, or empty list if provider not ready."""
    try:
        from adam.services.llm_service import LLMService
        service = LLMService()
        if service.llm_client and service.llm_client._local_provider:
            return service.llm_client._local_provider.get_available_models()
    except Exception:
        pass
    return []


def get_smart_defaults(prefer_local: bool = False) -> Dict[str, str]:
    """Return the default model-per-role mapping.

    Args:
        prefer_local: When True, assign the most capable local model to
            all roles. Falls back to cloud defaults if no local models
            are available.
    """
    if prefer_local:
        local_models = _get_local_models()
        if local_models:
            best_local = max(local_models, key=lambda m: m.parameter_count)
            return {
                "reasoner": best_local.model_id,
                "coder": best_local.model_id,
                "critic": best_local.model_id,
                "synthesizer": best_local.model_id,
            }

    llm_config = LLMConfig()
    available_models = set(llm_config.get_available_models())

    def pick(*candidates: str) -> Optional[str]:
        for candidate in candidates:
            if candidate in available_models:
                return candidate
        return None

    defaults = {
        "reasoner": pick(
            "grok-4.20-0309-reasoning",
            "gpt-5.4-2026-03-05",
            "grok-4.20-0309-non-reasoning",
            "gpt-5.4-mini-2026-03-17",
            "grok-3-mini-high",
            "gpt-4.1-mini-2025-04-14",
        ),
        "coder": pick(
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "gpt-5.4-2026-03-05",
            "gpt-5.4-mini-2026-03-17",
            "grok-4.20-0309-reasoning",
            "grok-4.20-0309-non-reasoning",
        ),
        "critic": pick(
            "gpt-5.4-2026-03-05",
            "gpt-5.4-mini-2026-03-17",
            "claude-sonnet-4-6",
            "grok-4.20-0309-reasoning",
            "grok-4.20-0309-non-reasoning",
        ),
        "synthesizer": pick(
            "claude-sonnet-4-6",
            "gpt-5.4-2026-03-05",
            "gpt-5.4-mini-2026-03-17",
            "grok-4.20-0309-non-reasoning",
            "grok-4.20-0309-reasoning",
        ),
    }

    fallback_defaults = {
        "reasoner": "grok-4.20-multi-agent-0309",
        "coder": "claude-opus-4-6",
        "critic": "gpt-5.4-2026-03-05",
        "synthesizer": "claude-sonnet-4-6",
    }

    return {
        role: model or fallback_defaults[role]
        for role, model in defaults.items()
    }


AVAILABLE_MODELS: Dict[str, str] = {
    "Claude Opus 4.6": "claude-opus-4-6",
    "Claude Sonnet 4.6": "claude-sonnet-4-6",
    "Claude Haiku 3.5": "claude-haiku-3-5",
    "GPT-5.4 (2026-03-05)": "gpt-5.4-2026-03-05",
    "GPT-4o": "gpt-4o",
    "o3": "o3",
    "Grok 4.20 Multi-Agent": "grok-4.20-multi-agent-0309",
    "Grok 3": "grok-3",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 2.0 Flash": "gemini-2.0-flash",
}


@dataclass
class SessionConfig:
    question: str
    pattern: str
    model_assignments: Dict[str, str] = field(default_factory=dict)
    budget: float = 2.0
    prefer_local: bool = False
    conversation_id: Optional[str] = None
    conversation_context: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.model_assignments:
            self.model_assignments = get_smart_defaults(prefer_local=self.prefer_local)
