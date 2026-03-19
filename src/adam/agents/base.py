"""Agent base class for multi-agent reasoning."""

import logging
from dataclasses import dataclass
from typing import Optional, AsyncGenerator
from datetime import datetime

from .scratchpad import Scratchpad, ScratchpadEntry, AgentRole, EntryType

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent instance."""
    role: AgentRole
    name: str
    system_prompt: str
    entry_type: EntryType = EntryType.NOTE
    preferred_model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4000


class Agent:
    """A specialized AI agent that contributes to multi-agent reasoning.

    Each agent has a role, a system prompt, and can read from / write to
    the shared scratchpad. Agents are lightweight -- they're structured
    LLM calls, not separate processes.
    """

    def __init__(self, config: AgentConfig, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.total_cost = 0.0
        self.total_tokens = 0

    async def think(self, scratchpad: Scratchpad, focus: Optional[str] = None) -> ScratchpadEntry:
        """Process the scratchpad and contribute a response.

        Args:
            scratchpad: Shared context with all agent contributions so far
            focus: Optional additional instruction for this thinking step

        Returns:
            A new ScratchpadEntry to add to the scratchpad
        """
        prompt = self._build_prompt(scratchpad, focus)

        response = await self.llm_client.complete(
            prompt=prompt,
            model=self.config.preferred_model,
            system_prompt=self.config.system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        self.total_cost += getattr(response, 'cost', 0.0)
        self.total_tokens += getattr(response, 'total_tokens', 0)

        return ScratchpadEntry(
            agent_role=self.config.role,
            agent_name=self.config.name,
            content=response.content,
            entry_type=self.config.entry_type,
            model_used=getattr(response, 'model', ''),
            cost=getattr(response, 'cost', 0.0),
            tokens=getattr(response, 'total_tokens', 0),
            timestamp=datetime.now(),
        )

    async def think_stream(self, scratchpad: Scratchpad, focus: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Stream the agent's thinking process token by token."""
        import inspect

        prompt = self._build_prompt(scratchpad, focus)
        result = await self.llm_client.complete(
            prompt=prompt,
            model=self.config.preferred_model,
            system_prompt=self.config.system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )

        if inspect.isasyncgen(result):
            async for chunk in result:
                yield chunk
        else:
            yield result.content

    def _build_prompt(self, scratchpad: Scratchpad, focus: Optional[str] = None) -> str:
        """Build the prompt for this agent from the scratchpad context."""
        context = scratchpad.build_context_for_agent(self.config.role)

        prompt_parts = [context]

        if focus:
            prompt_parts.append(f"\nFOCUS: {focus}")

        prompt_parts.append(f"\nAs the {self.config.name}, provide your analysis:")

        return "\n".join(prompt_parts)
