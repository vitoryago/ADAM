"""ADAM Multi-Agent Reasoning System."""

from .base import Agent, AgentConfig
from .scratchpad import Scratchpad, ScratchpadEntry, AgentRole, EntryType
from .roles import (
    create_reasoner, create_coder, create_researcher,
    create_critic, create_synthesizer,
)
from .orchestrator import Orchestrator, OrchestrationPattern
from .cost_guard import CostGuard
from .synthesis import FallbackSynthesizer
from .patterns.sequential import run_sequential_pipeline, stream_sequential_pipeline
from .patterns.debate import run_debate_pipeline, stream_debate_pipeline

__all__ = [
    'Agent', 'AgentConfig',
    'Scratchpad', 'ScratchpadEntry', 'AgentRole', 'EntryType',
    'create_reasoner', 'create_coder', 'create_researcher',
    'create_critic', 'create_synthesizer',
    'Orchestrator', 'OrchestrationPattern',
    'CostGuard',
    'FallbackSynthesizer',
    'run_sequential_pipeline', 'stream_sequential_pipeline',
    'run_debate_pipeline', 'stream_debate_pipeline',
]
