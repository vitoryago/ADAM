"""ADAM Multi-Agent Reasoning System."""

from .base import Agent, AgentConfig
from .scratchpad import Scratchpad, ScratchpadEntry, AgentRole, EntryType
from .roles import (
    create_reasoner, create_coder, create_researcher,
    create_critic, create_synthesizer,
)

__all__ = [
    'Agent', 'AgentConfig',
    'Scratchpad', 'ScratchpadEntry', 'AgentRole', 'EntryType',
    'create_reasoner', 'create_coder', 'create_researcher',
    'create_critic', 'create_synthesizer',
]
