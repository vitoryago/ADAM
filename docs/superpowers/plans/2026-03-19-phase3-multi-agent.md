# Phase 3: Multi-Agent Reasoning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-agent reasoning system where specialized AI agents from different providers collaborate to solve complex problems. When ADAM faces a hard question, a Reasoner plans, a Coder implements, a Critic reviews, and a Synthesizer combines everything into a thorough answer.

**Architecture:** New `src/adam/agents/` package with an Agent base class, role-specific system prompts, a shared Scratchpad for inter-agent communication, and an Orchestrator that selects patterns (sequential pipeline, debate/consensus) and manages the flow. Integrated into the chat pipeline via SSE streaming with agent progress indicators.

**Tech Stack:** Python 3.9+, asyncio, dataclasses, UnifiedLLMClient (multi-provider), FastAPI SSE

**Spec:** `docs/superpowers/specs/2026-03-18-adam-roadmap-design.md` (Phase 3 section)

---

## File Structure

```
src/adam/agents/
├── __init__.py           # Public exports
├── base.py               # Agent base class with think() and think_stream()
├── roles.py              # Built-in roles: Reasoner, Coder, Researcher, Critic, Synthesizer
├── scratchpad.py          # Shared scratchpad data structure
├── orchestrator.py        # Main orchestration engine
├── cost_guard.py          # Budget tracking and enforcement
├── synthesis.py           # Multi-agent result synthesis
└── patterns/
    ├── __init__.py
    ├── sequential.py      # Sequential pipeline: Reasoner → Coder → Critic → Synthesize
    └── debate.py          # Debate: Agent A + Agent B → Reconciler
```

## Dependency Graph

```
Task 1 (Agent Foundation)
   ├──→ Task 2 (Sequential Pipeline) ──┐
   ├──→ Task 3 (Debate Pattern)  ──────┤
   └──→ Task 4 (Cost Guard + Synthesis)┘
                                        ├──→ Task 5 (Orchestrator) ──→ Task 6 (Integration) ──→ Task 7 (Tests)
```

**Parallel: Tasks 2 + 3 + 4 can run simultaneously after Task 1.**

---

### Task 1: Agent Foundation

**Goal:** Build the core agent abstraction, scratchpad, and role definitions. Everything else builds on this.

**Files:**
- Create: `src/adam/agents/__init__.py`
- Create: `src/adam/agents/base.py`
- Create: `src/adam/agents/scratchpad.py`
- Create: `src/adam/agents/roles.py`
- Create: `src/adam/agents/patterns/__init__.py`
- Create: `tests/test_agents_base.py`

- [ ] **Step 1: Create src/adam/agents/scratchpad.py**

The scratchpad is the shared memory between agents during a multi-agent session.

```python
"""Shared scratchpad for multi-agent communication."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class AgentRole(Enum):
    REASONER = "reasoner"
    CODER = "coder"
    RESEARCHER = "researcher"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"


class EntryType(Enum):
    QUERY = "query"           # Original user question
    PLAN = "plan"             # Reasoner's breakdown
    CODE = "code"             # Coder's implementation
    RESEARCH = "research"     # Researcher's findings
    CRITIQUE = "critique"     # Critic's review
    SYNTHESIS = "synthesis"   # Final combined answer
    NOTE = "note"             # General contribution


@dataclass
class ScratchpadEntry:
    """A single contribution to the shared scratchpad."""
    agent_role: AgentRole
    agent_name: str
    content: str
    entry_type: EntryType
    model_used: str = ""
    cost: float = 0.0
    tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scratchpad:
    """Shared context for multi-agent collaboration."""
    query: str
    entries: List[ScratchpadEntry] = field(default_factory=list)
    total_cost: float = 0.0
    budget: float = 0.50
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_entry(self, entry: ScratchpadEntry):
        self.entries.append(entry)
        self.total_cost += entry.cost

    def get_entries_by_role(self, role: AgentRole) -> List[ScratchpadEntry]:
        return [e for e in self.entries if e.agent_role == role]

    def get_entries_by_type(self, entry_type: EntryType) -> List[ScratchpadEntry]:
        return [e for e in self.entries if e.entry_type == entry_type]

    def get_latest_entry(self) -> Optional[ScratchpadEntry]:
        return self.entries[-1] if self.entries else None

    def budget_remaining(self) -> float:
        return max(0.0, self.budget - self.total_cost)

    def is_over_budget(self) -> bool:
        return self.total_cost >= self.budget

    def build_context_for_agent(self, role: AgentRole) -> str:
        """Build context string showing what this agent should see.

        Visibility rules:
        - REASONER: sees only the original query
        - CODER: sees query + reasoner's plan
        - RESEARCHER: sees query + plan
        - CRITIC: sees query + plan + code + research
        - SYNTHESIZER: sees everything
        """
        parts = [f"USER QUERY:\n{self.query}\n"]

        visible_entries = self._get_visible_entries(role)
        for entry in visible_entries:
            header = f"\n--- {entry.agent_name} ({entry.entry_type.value}) ---"
            parts.append(f"{header}\n{entry.content}\n")

        return "\n".join(parts)

    def _get_visible_entries(self, role: AgentRole) -> List[ScratchpadEntry]:
        """Determine which entries are visible to a given role."""
        if role == AgentRole.REASONER:
            return []  # Reasoner works from query only
        elif role == AgentRole.CODER:
            return [e for e in self.entries if e.entry_type in (EntryType.PLAN, EntryType.RESEARCH)]
        elif role == AgentRole.RESEARCHER:
            return [e for e in self.entries if e.entry_type == EntryType.PLAN]
        elif role == AgentRole.CRITIC:
            return [e for e in self.entries if e.entry_type in (EntryType.PLAN, EntryType.CODE, EntryType.RESEARCH)]
        elif role == AgentRole.SYNTHESIZER:
            return self.entries  # Sees everything
        return self.entries

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses."""
        return {
            "query": self.query,
            "total_cost": self.total_cost,
            "budget": self.budget,
            "budget_remaining": self.budget_remaining(),
            "entries": [
                {
                    "agent_role": e.agent_role.value,
                    "agent_name": e.agent_name,
                    "content": e.content,
                    "entry_type": e.entry_type.value,
                    "model_used": e.model_used,
                    "cost": e.cost,
                    "tokens": e.tokens,
                }
                for e in self.entries
            ],
        }
```

- [ ] **Step 2: Create src/adam/agents/base.py**

```python
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
    the shared scratchpad. Agents are lightweight — they're structured
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
```

- [ ] **Step 3: Create src/adam/agents/roles.py**

```python
"""Built-in agent roles with specialized system prompts."""

from .base import AgentConfig, Agent
from .scratchpad import AgentRole, EntryType


# --- System Prompts ---

REASONER_PROMPT = """You are a Reasoning Specialist in a team of AI agents solving a complex problem.

Your job is to ANALYZE the problem and create a STRUCTURED PLAN. You do NOT implement solutions.

When given a problem:
1. Identify the core challenge and any sub-problems
2. Consider constraints, edge cases, and requirements
3. List the steps needed to solve this, in order
4. Identify what expertise is needed (coding, research, domain knowledge)

Output a clear, numbered plan that other specialists will follow.
Be specific about what each step should achieve."""

CODER_PROMPT = """You are a Code Specialist in a team of AI agents.

Your job is to IMPLEMENT solutions based on the plan created by the Reasoning Specialist.

When given a plan:
1. Write clean, production-quality code
2. Follow best practices for the language and framework
3. Include error handling for edge cases
4. Add brief comments for complex logic only

Focus on implementation. The plan has already been analyzed — follow it.
If the plan is unclear on a point, note what you assumed."""

RESEARCHER_PROMPT = """You are a Research Specialist in a team of AI agents.

Your job is to find relevant information, context, and best practices.

When given a topic:
1. Identify what information the team needs
2. Provide relevant patterns, documentation references, and best practices
3. Note similar solved problems and their approaches
4. Highlight potential pitfalls the team should watch for

Focus on gathering and organizing knowledge — leave implementation to the Code Specialist."""

CRITIC_PROMPT = """You are a Code Review Specialist in a team of AI agents.

Your job is to find problems, gaps, and potential improvements in the team's work.

When reviewing:
1. CORRECTNESS — Does it solve the stated problem?
2. EDGE CASES — What could go wrong?
3. BEST PRACTICES — Is this the right approach?
4. COMPLETENESS — Is anything missing?

Be constructive. For each issue, provide a specific suggestion.

End with a confidence rating:
- HIGH: Ready to ship
- MEDIUM: Minor issues, mostly good
- LOW: Significant issues need fixing before this is ready"""

SYNTHESIZER_PROMPT = """You are a Synthesis Specialist. Your job is to combine the work of multiple AI specialists into a single, coherent response for the user.

You have access to contributions from:
- Reasoning Specialist (problem analysis and planning)
- Code Specialist (implementation)
- Research Specialist (context and references)
- Code Review Specialist (quality assessment)

Create a unified response that:
1. Starts with a brief summary of the approach taken
2. Presents the solution clearly (code, explanation, or both)
3. Highlights key decisions and trade-offs noted by the specialists
4. Notes any caveats or limitations from the reviewer
5. Reads naturally — the user should feel like talking to one smart assistant, not reading a committee report

Do NOT mention the other agents or the multi-agent process. Present this as a well-thought-out answer."""


# --- Agent Factory ---

def create_reasoner(llm_client, preferred_model: str = None) -> Agent:
    config = AgentConfig(
        role=AgentRole.REASONER,
        name="Reasoner",
        system_prompt=REASONER_PROMPT,
        entry_type=EntryType.PLAN,
        preferred_model=preferred_model,
        temperature=0.5,
        max_tokens=2000,
    )
    return Agent(config, llm_client)

def create_coder(llm_client, preferred_model: str = None) -> Agent:
    config = AgentConfig(
        role=AgentRole.CODER,
        name="Coder",
        system_prompt=CODER_PROMPT,
        entry_type=EntryType.CODE,
        preferred_model=preferred_model,
        temperature=0.3,
        max_tokens=4000,
    )
    return Agent(config, llm_client)

def create_researcher(llm_client, preferred_model: str = None) -> Agent:
    config = AgentConfig(
        role=AgentRole.RESEARCHER,
        name="Researcher",
        system_prompt=RESEARCHER_PROMPT,
        entry_type=EntryType.RESEARCH,
        preferred_model=preferred_model,
        temperature=0.5,
        max_tokens=2000,
    )
    return Agent(config, llm_client)

def create_critic(llm_client, preferred_model: str = None) -> Agent:
    config = AgentConfig(
        role=AgentRole.CRITIC,
        name="Critic",
        system_prompt=CRITIC_PROMPT,
        entry_type=EntryType.CRITIQUE,
        preferred_model=preferred_model,
        temperature=0.4,
        max_tokens=2000,
    )
    return Agent(config, llm_client)

def create_synthesizer(llm_client, preferred_model: str = None) -> Agent:
    config = AgentConfig(
        role=AgentRole.SYNTHESIZER,
        name="Synthesizer",
        system_prompt=SYNTHESIZER_PROMPT,
        entry_type=EntryType.SYNTHESIS,
        preferred_model=preferred_model,
        temperature=0.6,
        max_tokens=4000,
    )
    return Agent(config, llm_client)
```

- [ ] **Step 4: Create src/adam/agents/__init__.py and patterns/__init__.py**

```python
# src/adam/agents/__init__.py
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
```

```python
# src/adam/agents/patterns/__init__.py
"""Orchestration patterns for multi-agent reasoning."""
```

- [ ] **Step 5: Write tests/test_agents_base.py**

```python
"""Tests for the agent foundation: base class, scratchpad, and roles."""
import pytest
from adam.agents.scratchpad import Scratchpad, ScratchpadEntry, AgentRole, EntryType
from adam.agents.base import Agent, AgentConfig
from adam.agents.roles import (
    create_reasoner, create_coder, create_researcher,
    create_critic, create_synthesizer,
    REASONER_PROMPT, CODER_PROMPT, CRITIC_PROMPT,
)


class TestScratchpad:
    def test_create_empty(self):
        pad = Scratchpad(query="How do I sort a list?")
        assert pad.query == "How do I sort a list?"
        assert len(pad.entries) == 0
        assert pad.total_cost == 0.0

    def test_add_entry(self):
        pad = Scratchpad(query="test")
        entry = ScratchpadEntry(
            agent_role=AgentRole.REASONER,
            agent_name="Reasoner",
            content="Step 1: Analyze. Step 2: Implement.",
            entry_type=EntryType.PLAN,
            cost=0.01,
            tokens=50,
        )
        pad.add_entry(entry)
        assert len(pad.entries) == 1
        assert pad.total_cost == 0.01

    def test_budget_tracking(self):
        pad = Scratchpad(query="test", budget=0.05)
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.REASONER, agent_name="R",
            content="plan", entry_type=EntryType.PLAN, cost=0.03
        ))
        assert pad.budget_remaining() == pytest.approx(0.02)
        assert not pad.is_over_budget()
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.CODER, agent_name="C",
            content="code", entry_type=EntryType.CODE, cost=0.03
        ))
        assert pad.is_over_budget()

    def test_visibility_reasoner_sees_nothing(self):
        pad = Scratchpad(query="test")
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.CODER, agent_name="C",
            content="old code", entry_type=EntryType.CODE
        ))
        visible = pad._get_visible_entries(AgentRole.REASONER)
        assert len(visible) == 0

    def test_visibility_coder_sees_plan(self):
        pad = Scratchpad(query="test")
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.REASONER, agent_name="R",
            content="the plan", entry_type=EntryType.PLAN
        ))
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.CRITIC, agent_name="Cr",
            content="critique", entry_type=EntryType.CRITIQUE
        ))
        visible = pad._get_visible_entries(AgentRole.CODER)
        assert len(visible) == 1
        assert visible[0].entry_type == EntryType.PLAN

    def test_visibility_synthesizer_sees_all(self):
        pad = Scratchpad(query="test")
        for role, etype in [(AgentRole.REASONER, EntryType.PLAN),
                             (AgentRole.CODER, EntryType.CODE),
                             (AgentRole.CRITIC, EntryType.CRITIQUE)]:
            pad.add_entry(ScratchpadEntry(
                agent_role=role, agent_name=role.value,
                content=f"{role.value} output", entry_type=etype
            ))
        visible = pad._get_visible_entries(AgentRole.SYNTHESIZER)
        assert len(visible) == 3

    def test_context_building(self):
        pad = Scratchpad(query="Design an API")
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.REASONER, agent_name="Reasoner",
            content="1. Define endpoints\n2. Add auth",
            entry_type=EntryType.PLAN
        ))
        context = pad.build_context_for_agent(AgentRole.CODER)
        assert "Design an API" in context
        assert "Define endpoints" in context

    def test_to_dict(self):
        pad = Scratchpad(query="test", budget=1.0)
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.REASONER, agent_name="R",
            content="plan", entry_type=EntryType.PLAN, cost=0.01
        ))
        d = pad.to_dict()
        assert d["query"] == "test"
        assert len(d["entries"]) == 1
        assert d["entries"][0]["agent_role"] == "reasoner"


class TestAgentConfig:
    def test_create_config(self):
        config = AgentConfig(
            role=AgentRole.REASONER,
            name="Reasoner",
            system_prompt="You are a reasoner.",
        )
        assert config.role == AgentRole.REASONER
        assert config.temperature == 0.7
        assert config.max_tokens == 4000


class TestRoles:
    def test_all_roles_have_prompts(self):
        assert len(REASONER_PROMPT) > 100
        assert len(CODER_PROMPT) > 100
        assert len(CRITIC_PROMPT) > 100

    def test_create_all_roles(self):
        class MockClient:
            pass
        client = MockClient()

        agents = [
            create_reasoner(client),
            create_coder(client),
            create_researcher(client),
            create_critic(client),
            create_synthesizer(client),
        ]

        roles = [a.config.role for a in agents]
        assert AgentRole.REASONER in roles
        assert AgentRole.CODER in roles
        assert AgentRole.CRITIC in roles
        assert AgentRole.SYNTHESIZER in roles

    def test_agent_temperatures(self):
        class MockClient:
            pass
        reasoner = create_reasoner(MockClient())
        coder = create_coder(MockClient())
        assert coder.config.temperature < reasoner.config.temperature  # Coder more deterministic


class TestAgent:
    def test_build_prompt(self):
        config = AgentConfig(
            role=AgentRole.CODER, name="Coder",
            system_prompt="You write code.",
        )

        class MockClient:
            pass

        agent = Agent(config, MockClient())
        pad = Scratchpad(query="Write a sort function")
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.REASONER, agent_name="Reasoner",
            content="Use quicksort algorithm",
            entry_type=EntryType.PLAN,
        ))

        prompt = agent._build_prompt(pad)
        assert "Write a sort function" in prompt
        assert "quicksort" in prompt
        assert "Coder" in prompt

    def test_build_prompt_with_focus(self):
        config = AgentConfig(
            role=AgentRole.CRITIC, name="Critic",
            system_prompt="You review code.",
        )
        agent = Agent(config, None)
        pad = Scratchpad(query="test")
        prompt = agent._build_prompt(pad, focus="Focus on error handling")
        assert "Focus on error handling" in prompt
```

- [ ] **Step 6: Run tests**
```bash
python -m pytest tests/test_agents_base.py -v
```
Expected: ALL PASS

- [ ] **Step 7: Commit**
```bash
git add -A && git commit -m "feat: add multi-agent foundation (Agent, Scratchpad, Roles)"
```

---

### Task 2: Sequential Pipeline Pattern

**Goal:** Implement the Reasoner → Coder → Critic → Synthesizer pipeline.

**Files:**
- Create: `src/adam/agents/patterns/sequential.py`
- Create: `tests/test_sequential_pattern.py`

- [ ] **Step 1: Create src/adam/agents/patterns/sequential.py**

```python
"""Sequential pipeline: Reasoner → Coder → Critic → Synthesizer."""

import logging
from typing import AsyncGenerator, Dict, Any, Optional
from ..base import Agent
from ..scratchpad import Scratchpad, AgentRole
from ..roles import create_reasoner, create_coder, create_critic, create_synthesizer

logger = logging.getLogger(__name__)


async def run_sequential_pipeline(
    scratchpad: Scratchpad,
    llm_client,
    model_assignments: Optional[Dict[AgentRole, str]] = None,
) -> Scratchpad:
    """Run the sequential pipeline: Reasoner → Coder → Critic → Synthesizer.

    Args:
        scratchpad: Scratchpad with the user query
        llm_client: UnifiedLLMClient instance
        model_assignments: Optional mapping of role → model name

    Returns:
        The scratchpad with all agent contributions
    """
    models = model_assignments or {}

    agents = [
        create_reasoner(llm_client, models.get(AgentRole.REASONER)),
        create_coder(llm_client, models.get(AgentRole.CODER)),
        create_critic(llm_client, models.get(AgentRole.CRITIC)),
        create_synthesizer(llm_client, models.get(AgentRole.SYNTHESIZER)),
    ]

    for agent in agents:
        if scratchpad.is_over_budget():
            logger.warning(f"Budget exceeded before {agent.config.name}, skipping to synthesis")
            break

        entry = await agent.think(scratchpad)
        scratchpad.add_entry(entry)
        logger.info(f"{agent.config.name} contributed ({entry.tokens} tokens, ${entry.cost:.4f})")

    return scratchpad


async def stream_sequential_pipeline(
    scratchpad: Scratchpad,
    llm_client,
    model_assignments: Optional[Dict[AgentRole, str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream the sequential pipeline with progress updates.

    Yields dicts with:
    - {"type": "agent_start", "agent": "Reasoner", "role": "reasoner"}
    - {"type": "agent_chunk", "agent": "Reasoner", "content": "..."}
    - {"type": "agent_done", "agent": "Reasoner", "cost": 0.01, "tokens": 150}
    - {"type": "synthesis", "content": "Final answer..."}
    - {"type": "done", "total_cost": 0.05, "agents_used": 4}
    """
    models = model_assignments or {}

    pipeline = [
        create_reasoner(llm_client, models.get(AgentRole.REASONER)),
        create_coder(llm_client, models.get(AgentRole.CODER)),
        create_critic(llm_client, models.get(AgentRole.CRITIC)),
        create_synthesizer(llm_client, models.get(AgentRole.SYNTHESIZER)),
    ]

    for agent in pipeline:
        if scratchpad.is_over_budget():
            yield {"type": "budget_exceeded", "agent": agent.config.name}
            break

        yield {
            "type": "agent_start",
            "agent": agent.config.name,
            "role": agent.config.role.value,
        }

        # Stream agent's thinking
        accumulated = ""
        try:
            async for chunk in agent.think_stream(scratchpad):
                accumulated += chunk
                yield {
                    "type": "agent_chunk",
                    "agent": agent.config.name,
                    "content": chunk,
                }
        except Exception as e:
            logger.error(f"Streaming failed for {agent.config.name}: {e}")
            # Fallback to non-streaming
            entry = await agent.think(scratchpad)
            accumulated = entry.content
            yield {
                "type": "agent_chunk",
                "agent": agent.config.name,
                "content": accumulated,
            }

        # Create scratchpad entry from accumulated content
        from ..scratchpad import ScratchpadEntry
        from datetime import datetime
        entry = ScratchpadEntry(
            agent_role=agent.config.role,
            agent_name=agent.config.name,
            content=accumulated,
            entry_type=agent.config.entry_type,
            model_used=agent.config.preferred_model or "",
            cost=agent.total_cost,
            tokens=agent.total_tokens,
            timestamp=datetime.now(),
        )
        scratchpad.add_entry(entry)

        yield {
            "type": "agent_done",
            "agent": agent.config.name,
            "cost": agent.total_cost,
            "tokens": agent.total_tokens,
        }

    # Final summary
    yield {
        "type": "done",
        "total_cost": scratchpad.total_cost,
        "agents_used": len(scratchpad.entries),
        "scratchpad": scratchpad.to_dict(),
    }
```

- [ ] **Step 2: Write tests/test_sequential_pattern.py**

Use a mock LLM client that returns predefined responses. Test that:
- All 4 agents run in sequence
- Scratchpad accumulates entries in order
- Budget exceeded skips remaining agents
- Streaming yields correct event types

- [ ] **Step 3: Run tests**
```bash
python -m pytest tests/test_sequential_pattern.py -v
```

- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "feat: add sequential pipeline pattern (Reasoner→Coder→Critic→Synthesizer)"
```

---

### Task 3: Debate Pattern

**Goal:** Two agents solve independently, a reconciler compares and picks the best approach.

**Files:**
- Create: `src/adam/agents/patterns/debate.py`
- Create: `tests/test_debate_pattern.py`

- [ ] **Step 1: Create src/adam/agents/patterns/debate.py**

Implement a debate pattern where:
- Two agents (e.g., two different providers or two different system prompts) independently analyze the query
- A reconciler agent reads both perspectives and synthesizes the best answer
- Both debaters write to separate scratchpad entries
- The reconciler sees both and produces a synthesis

Key design: use asyncio.gather() to run both debaters in parallel.

- [ ] **Step 2: Write tests** verifying both agents run and reconciler sees both outputs

- [ ] **Step 3: Run tests and commit**

---

### Task 4: Cost Guard + Synthesis

**Goal:** Budget enforcement and multi-agent result synthesis.

**Files:**
- Create: `src/adam/agents/cost_guard.py`
- Create: `src/adam/agents/synthesis.py`
- Create: `tests/test_cost_guard.py`

- [ ] **Step 1: Create cost_guard.py**

```python
"""Cost tracking and budget enforcement for multi-agent sessions."""

from dataclasses import dataclass
from typing import Optional
from .scratchpad import Scratchpad, AgentRole

@dataclass
class CostEstimate:
    """Estimated cost for an agent call."""
    agent_role: AgentRole
    estimated_tokens: int
    estimated_cost: float
    model: str

class CostGuard:
    """Enforces budget limits during multi-agent orchestration."""

    def __init__(self, budget: float = 0.50):
        self.budget = budget
        self.spent = 0.0

    def can_afford(self, estimated_cost: float) -> bool:
        return (self.spent + estimated_cost) <= self.budget

    def record_spend(self, cost: float):
        self.spent += cost

    def remaining(self) -> float:
        return max(0.0, self.budget - self.spent)

    def suggest_fallback(self) -> str:
        """When budget is tight, suggest using a cheaper model or skipping agents."""
        if self.remaining() < 0.01:
            return "skip"  # Skip remaining agents
        elif self.remaining() < 0.05:
            return "cheap_model"  # Use cheapest available model
        return "continue"  # Normal operation
```

- [ ] **Step 2: Create synthesis.py**

Module that takes a completed scratchpad and produces the final user-facing response. The Synthesizer agent handles this, but this module provides fallback synthesis if the Synthesizer agent fails or is skipped.

- [ ] **Step 3: Write tests and commit**

---

### Task 5: Orchestrator

**Goal:** The main entry point that decides when to use multi-agent, selects patterns, and manages the flow.

**Files:**
- Create: `src/adam/agents/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Create orchestrator.py**

```python
"""Multi-agent orchestration engine."""

import logging
from typing import Optional, Dict, Any, AsyncGenerator
from enum import Enum

from .scratchpad import Scratchpad, AgentRole
from .cost_guard import CostGuard
from .patterns.sequential import run_sequential_pipeline, stream_sequential_pipeline
from .patterns.debate import run_debate_pipeline, stream_debate_pipeline

logger = logging.getLogger(__name__)


class OrchestrationPattern(Enum):
    SEQUENTIAL = "sequential"
    DEBATE = "debate"
    SINGLE = "single"  # No multi-agent, just one model


class Orchestrator:
    """Decides when and how to use multi-agent reasoning."""

    def __init__(self, llm_client, memory_service=None):
        self.llm_client = llm_client
        self.memory_service = memory_service

    def should_use_multi_agent(self, query: str, complexity: str = "simple") -> bool:
        if complexity not in ("complex",):
            return False
        triggers = [
            "design", "architect", "implement a", "build a", "create a system",
            "optimize", "debug complex", "compare", "pros and cons",
            "refactor", "migrate", "full solution",
        ]
        query_lower = query.lower()
        return any(t in query_lower for t in triggers)

    def select_pattern(self, query: str) -> OrchestrationPattern:
        query_lower = query.lower()
        if any(w in query_lower for w in ["compare", "tradeoff", "pros cons", "vs", "which is better"]):
            return OrchestrationPattern.DEBATE
        return OrchestrationPattern.SEQUENTIAL

    async def orchestrate(
        self,
        query: str,
        budget: float = 0.50,
        model_assignments: Optional[Dict[AgentRole, str]] = None,
    ) -> Dict[str, Any]:
        scratchpad = Scratchpad(query=query, budget=budget)
        pattern = self.select_pattern(query)

        if pattern == OrchestrationPattern.DEBATE:
            scratchpad = await run_debate_pipeline(scratchpad, self.llm_client, model_assignments)
        else:
            scratchpad = await run_sequential_pipeline(scratchpad, self.llm_client, model_assignments)

        # Extract final response
        synthesis = scratchpad.get_entries_by_type(
            __import__('adam.agents.scratchpad', fromlist=['EntryType']).EntryType.SYNTHESIS
        )
        final_response = synthesis[-1].content if synthesis else scratchpad.get_latest_entry().content

        return {
            "response": final_response,
            "pattern": pattern.value,
            "total_cost": scratchpad.total_cost,
            "agents_used": len(scratchpad.entries),
            "scratchpad": scratchpad.to_dict(),
        }

    async def orchestrate_stream(
        self,
        query: str,
        budget: float = 0.50,
        model_assignments: Optional[Dict[AgentRole, str]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        scratchpad = Scratchpad(query=query, budget=budget)
        pattern = self.select_pattern(query)

        yield {"type": "orchestration_start", "pattern": pattern.value, "budget": budget}

        if pattern == OrchestrationPattern.DEBATE:
            stream = stream_debate_pipeline(scratchpad, self.llm_client, model_assignments)
        else:
            stream = stream_sequential_pipeline(scratchpad, self.llm_client, model_assignments)

        async for event in stream:
            yield event
```

- [ ] **Step 2: Write tests** — mock LLM client, verify pattern selection, verify orchestration produces results

- [ ] **Step 3: Run tests and commit**

---

### Task 6: Chat Pipeline Integration

**Goal:** Wire the orchestrator into the message flow. When a complex question arrives, use multi-agent instead of single LLM.

**Files:**
- Modify: `src/adam/api/routers/messages.py`
- Modify: `src/adam/services/llm_service.py`
- Create: `tests/test_multi_agent_integration.py`

- [ ] **Step 1: Add multi-agent mode to LLMService**

Add a method to LLMService that decides whether to use multi-agent and dispatches accordingly:

```python
async def generate_with_agents(self, message, history, memory_context, budget=0.50):
    """Use multi-agent reasoning for complex queries."""
    from adam.agents.orchestrator import Orchestrator
    orchestrator = Orchestrator(self.llm_client, self.memory_service)
    result = await orchestrator.orchestrate(message, budget=budget)
    return LLMResponse(
        content=result["response"],
        model_used=f"multi-agent ({result['pattern']})",
        tokens_used=sum(e["tokens"] for e in result["scratchpad"]["entries"]),
        cost=result["total_cost"],
        metadata={"multi_agent": True, "scratchpad": result["scratchpad"]},
    )
```

- [ ] **Step 2: Add multi-agent streaming to messages.py**

In the streaming endpoint, detect if multi-agent mode is needed and stream agent progress:

```python
# In send_message_stream:
if should_use_multi_agent(message.content):
    orchestrator = Orchestrator(llm_client)
    async for event in orchestrator.orchestrate_stream(message.content):
        if event["type"] == "agent_start":
            yield f"data: {json.dumps({'type': 'agent_status', 'agent': event['agent'], 'status': 'thinking'})}\n\n"
        elif event["type"] == "agent_chunk":
            yield f"data: {json.dumps({'type': 'agent_chunk', 'agent': event['agent'], 'content': event['content']})}\n\n"
        elif event["type"] == "done":
            yield f"data: {json.dumps({'type': 'done', 'multi_agent': True, 'total_cost': event['total_cost']})}\n\n"
```

- [ ] **Step 3: Add memory storage for multi-agent results**

After multi-agent completion, store the best artifacts in memory (the synthesis and any high-quality code produced).

- [ ] **Step 4: Write integration test and commit**

---

### Task 7: Comprehensive Testing & Review

**Goal:** Full test suite covering all multi-agent functionality.

**Files:**
- Update: `tests/test_integration.py`
- Run: full test suite

- [ ] **Step 1: Add Phase 3 integration tests**

Test the full flow: query → multi-agent trigger → orchestration → streaming → response.

- [ ] **Step 2: Run ALL tests**
```bash
python -m pytest tests/ -v --timeout=60
```

- [ ] **Step 3: Final commit**
```bash
git commit -m "feat: complete Phase 3 multi-agent reasoning system"
```
