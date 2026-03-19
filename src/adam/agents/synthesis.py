"""Fallback synthesis for multi-agent orchestration."""

from adam.agents.scratchpad import EntryType


class FallbackSynthesizer:
    """Fallback synthesis when the Synthesizer agent is unavailable or budget is exceeded."""

    @staticmethod
    def synthesize(scratchpad) -> str:
        """Create a response from scratchpad entries without using an LLM."""
        parts = []

        # Get plan if available
        plans = scratchpad.get_entries_by_type(EntryType.PLAN)
        if plans:
            parts.append("## Approach\n" + plans[-1].content)

        # Get code if available
        codes = scratchpad.get_entries_by_type(EntryType.CODE)
        if codes:
            parts.append("## Implementation\n" + codes[-1].content)

        # Get research if available
        research = scratchpad.get_entries_by_type(EntryType.RESEARCH)
        if research:
            parts.append("## Research & Context\n" + research[-1].content)

        # Get critique if available
        critiques = scratchpad.get_entries_by_type(EntryType.CRITIQUE)
        if critiques:
            parts.append("## Review Notes\n" + critiques[-1].content)

        if not parts:
            return "I wasn't able to complete the multi-agent analysis. Please try rephrasing your question."

        return "\n\n".join(parts)

    @staticmethod
    def extract_best_response(scratchpad) -> str:
        """Extract the single best response from the scratchpad."""
        # Prefer synthesis > code > plan > any
        for entry_type in [EntryType.SYNTHESIS, EntryType.CODE, EntryType.PLAN]:
            entries = scratchpad.get_entries_by_type(entry_type)
            if entries:
                return entries[-1].content

        # Fallback to last entry
        latest = scratchpad.get_latest_entry()
        return latest.content if latest else "No response generated."
