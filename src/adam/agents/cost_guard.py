"""Cost guard for multi-agent orchestration budget enforcement."""


class CostGuard:
    """Enforces budget limits during multi-agent orchestration."""

    def __init__(self, budget: float = 0.50):
        self.budget = budget
        self.spent = 0.0
        self.history = []  # List of (agent_name, cost) tuples

    def can_afford(self, estimated_cost: float) -> bool:
        """Check if there's enough budget for this operation."""
        return (self.spent + estimated_cost) <= self.budget

    def record_spend(self, agent_name: str, cost: float):
        """Record an agent's spend."""
        self.spent += cost
        self.history.append((agent_name, cost))

    def remaining(self) -> float:
        """Get remaining budget."""
        return max(0.0, self.budget - self.spent)

    def suggest_action(self) -> str:
        """Suggest what to do based on remaining budget."""
        remaining = self.remaining()
        if remaining < 0.01:
            return "skip"  # No budget left, skip remaining agents
        elif remaining < 0.05:
            return "cheap_model"  # Use cheapest model available
        return "continue"  # Normal operation

    def estimate_agent_cost(self, role, model=None) -> float:
        """Estimate cost for an agent call based on role and model."""
        # Rough estimates per agent role
        estimates = {
            "reasoner": 0.02,
            "coder": 0.05,
            "researcher": 0.03,
            "critic": 0.02,
            "synthesizer": 0.03,
        }
        return estimates.get(role, 0.03)

    def get_report(self) -> dict:
        """Get a cost breakdown report."""
        return {
            "budget": self.budget,
            "spent": self.spent,
            "remaining": self.remaining(),
            "history": self.history,
            "suggestion": self.suggest_action(),
        }
