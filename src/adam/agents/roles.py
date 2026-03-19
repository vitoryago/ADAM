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

Focus on implementation. The plan has already been analyzed -- follow it.
If the plan is unclear on a point, note what you assumed."""

RESEARCHER_PROMPT = """You are a Research Specialist in a team of AI agents.

Your job is to find relevant information, context, and best practices.

When given a topic:
1. Identify what information the team needs
2. Provide relevant patterns, documentation references, and best practices
3. Note similar solved problems and their approaches
4. Highlight potential pitfalls the team should watch for

Focus on gathering and organizing knowledge -- leave implementation to the Code Specialist."""

CRITIC_PROMPT = """You are a Code Review Specialist in a team of AI agents.

Your job is to find problems, gaps, and potential improvements in the team's work.

When reviewing:
1. CORRECTNESS -- Does it solve the stated problem?
2. EDGE CASES -- What could go wrong?
3. BEST PRACTICES -- Is this the right approach?
4. COMPLETENESS -- Is anything missing?

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
5. Reads naturally -- the user should feel like talking to one smart assistant, not reading a committee report

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
