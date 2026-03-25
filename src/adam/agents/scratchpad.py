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
    REBUTTAL = "rebuttal"     # Producer's response to reviewer feedback
    REACTION = "reaction"     # Reviewer's follow-up to the rebuttal


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
