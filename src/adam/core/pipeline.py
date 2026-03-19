#!/usr/bin/env python3
"""
LangGraph-based Conversation State Machine for ADAM

This module implements a sophisticated state machine that:
1. Analyzes query complexity (simple/moderate/complex)
2. Checks memory with confidence scoring (0.0-1.0)
3. Verifies memory freshness based on age and complexity
4. Routes to appropriate LLM based on complexity and cost
5. Handles retries and fallbacks gracefully
6. Stores results intelligently based on value

The state machine flow:
START -> Analyze Query -> Check Memory -> [Verify Freshness] -> Route LLM
-> Generate Response -> [Handle Error] -> Store Result -> END

Key decisions:
- Memory confidence > 0.7 triggers freshness verification
- Complex queries with coding keywords go to Claude Opus 4
- High memory confidence (>0.9) allows using cheaper models
- Errors trigger fallback to simpler models (max 3 retries)
"""

from typing import TypedDict, Optional, List, Literal, Annotated
from datetime import datetime, timedelta
import numpy as np
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import logging
import asyncio

logger = logging.getLogger(__name__)


class ConversationState(TypedDict):
    """
    State definition for the conversation flow.
    This is the core data structure that flows through the entire state machine.
    Each node reads from and writes to this state.
    """
    # Input
    query: str  # The user's original query

    # Analysis results
    complexity: Literal["simple", "moderate", "complex"]  # Determined by QueryComplexityAnalyzer
    complexity_score: float  # 0.0 to 1.0, confidence in complexity assessment

    # Memory search results
    memory_found: bool  # Whether any relevant memories were found
    memory_confidence: float  # 0.0 to 1.0, how confident we are in the memory match
    memory_ids: List[str]  # IDs of relevant memories found
    memory_content: Optional[str]  # The actual content of the best matching memory
    memory_age_days: Optional[int]  # How old the memory is in days

    # Decision flags
    should_verify: bool  # Whether we need to verify memory freshness (triggered by high confidence)
    should_use_memory: bool  # Final decision on whether to use the found memory

    # LLM routing
    selected_model: Literal["grok-3-mini-high", "grok-4-reasoning"]  # Model selected based on complexity

    # Response and metadata
    response: Optional[str]  # The final response from the LLM
    total_cost: float  # Accumulated cost for this query (in USD)
    retry_count: int  # Number of retry attempts (max 3)
    error_message: Optional[str]  # Error message if something went wrong

    # Tracking
    conversation_id: str  # Session ID for conversation tracking
    timestamp: datetime  # When this query was processed


class QueryComplexityAnalyzer:
    """
    Analyzes query complexity to determine optimal LLM routing.

    The analyzer uses keyword matching and heuristics to classify queries into:
    - Simple: Basic questions, definitions, explanations (use Grok-3)
    - Moderate: How-to guides, standard problems (use Grok-3)
    - Complex: Debugging, optimization, architecture (use O3 or Claude)
    """

    def __init__(self):
        # Keywords that indicate different complexity levels
        self.complexity_indicators = {
            # Simple queries usually start with these words or contain basic concepts
            "simple": [
                "what is", "how to", "define", "explain", "list",
                "when", "where", "who", "simple", "basic"
            ],
            # Complex queries involve advanced technical challenges
            "complex": [
                "debug", "optimize", "implement", "architect", "design",
                "integrate", "scale", "performance", "production",
                "multi", "distributed", "concurrent", "async"
            ],
            # Very technical queries that require deep understanding
            "technical_depth": [
                "algorithm", "complexity", "theorem", "proof",
                "mathematical", "formal", "rigorous"
            ]
        }

    def analyze(self, query: str) -> tuple[str, float]:
        """
        Analyze query complexity using keyword matching and heuristics.

        Args:
            query: The user's query text

        Returns:
            tuple: (complexity_level, confidence_score)
                - complexity_level: "simple", "moderate", or "complex"
                - confidence_score: 0.0 to 1.0 confidence in the assessment
        """
        query_lower = query.lower()

        # Count how many indicators of each type are present
        simple_count = sum(1 for word in self.complexity_indicators["simple"]
                         if word in query_lower)
        complex_count = sum(1 for word in self.complexity_indicators["complex"]
                          if word in query_lower)
        technical_count = sum(1 for word in self.complexity_indicators["technical_depth"]
                            if word in query_lower)

        # Consider query length as a factor
        word_count = len(query.split())

        # Apply heuristics to determine complexity
        # Short queries with simple keywords are simple
        if word_count < 10 and simple_count > 0 and complex_count == 0:
            return "simple", 0.8
        # Multiple complex keywords or technical depth indicates complex
        elif complex_count >= 2 or technical_count >= 1 or word_count > 50:
            return "complex", 0.85
        # Multiple questions in one query usually means complex
        elif "?" in query and query.count("?") > 1:
            return "complex", 0.7
        # Default to moderate for everything else
        else:
            return "moderate", 0.6


class MemoryConfidenceScorer:
    """
    Calculates confidence in memory matches using multiple factors.

    This scorer goes beyond simple similarity matching to evaluate how
    trustworthy and relevant a memory is for the current query.
    """

    def calculate_confidence(self,
                           query: str,
                           memory_query: str,
                           memory_response: str,
                           similarity_score: float,
                           age_days: int) -> float:
        """
        Calculate confidence score for a memory match.

        Args:
            query: The current user query
            memory_query: The original query that created this memory
            memory_response: The stored response in the memory
            similarity_score: Base semantic similarity (0.0 to 1.0)
            age_days: How many days old the memory is

        Returns:
            float: Confidence score from 0.0 to 1.0
        """
        # Start with the semantic similarity as our base confidence
        confidence = similarity_score

        # Calculate word overlap to catch exact phrase matches
        query_words = set(query.lower().split())
        memory_words = set(memory_query.lower().split())
        overlap = len(query_words & memory_words) / len(query_words)

        # Blend semantic similarity (70%) with word overlap (30%)
        confidence = confidence * 0.7 + overlap * 0.3

        # Apply age decay - newer memories are more trustworthy
        if age_days < 7:
            age_factor = 1.0      # Last week: full confidence
        elif age_days < 30:
            age_factor = 0.9      # Last month: slight decay
        elif age_days < 90:
            age_factor = 0.7      # Last quarter: moderate decay
        else:
            age_factor = 0.5      # Older: significant decay

        confidence *= age_factor

        # Evaluate response quality based on length
        response_length = len(memory_response)
        if response_length > 500:
            quality_factor = 1.0   # Detailed response
        elif response_length > 200:
            quality_factor = 0.9   # Good response
        else:
            quality_factor = 0.8   # Brief response

        confidence *= quality_factor

        # Ensure confidence stays within valid range [0.0, 1.0]
        return min(confidence, 1.0)


def analyze_query_node(state: ConversationState) -> ConversationState:
    """
    Node: Analyze query complexity.
    First node in the state machine.
    """
    analyzer = QueryComplexityAnalyzer()
    complexity, score = analyzer.analyze(state["query"])

    state["complexity"] = complexity
    state["complexity_score"] = score

    logger.info(f"Query complexity: {complexity} (confidence: {score})")
    return state


def check_memory_node(state: ConversationState) -> ConversationState:
    """
    Node: Check memory with confidence scoring.

    # TODO(phase2): connect to real memory system (adam.memory)
    Currently uses mock implementation.
    """
    query = state["query"]

    # TODO(phase2): Replace with actual memory network integration
    # from adam.memory import ADAMMemoryAdvanced
    # memory = ADAMMemoryAdvanced.get_instance()
    # memories = memory.search(query, n_results=5)

    # MOCK IMPLEMENTATION - Simulating memory search results
    if "optimization" in query.lower() or "error" in query.lower():
        state["memory_found"] = True
        state["memory_confidence"] = 0.75
        state["memory_ids"] = ["mem_001", "mem_002"]
        state["memory_content"] = "Previous solution: Check indexes and query plan"
        state["memory_age_days"] = 15
    else:
        state["memory_found"] = False
        state["memory_confidence"] = 0.0
        state["memory_ids"] = []
        state["memory_content"] = None
        state["memory_age_days"] = None

    if state["memory_found"] and state["memory_confidence"] > 0.7:
        state["should_verify"] = True
    else:
        state["should_verify"] = False

    logger.info(f"Memory found: {state['memory_found']}, "
                f"confidence: {state['memory_confidence']}")
    return state


def verify_memory_freshness_node(state: ConversationState) -> ConversationState:
    """
    Node: Verify if memory is still valid/fresh.

    Freshness rules:
    - Simple queries: Can use memories up to 90 days old
    - Moderate queries: Need memories less than 30 days old
    - Complex queries: Require very fresh memories (< 7 days)
    """
    if not state["should_verify"]:
        state["should_use_memory"] = state["memory_found"]
        return state

    age_days = state["memory_age_days"] or 0

    if state["complexity"] == "simple" and age_days < 90:
        state["should_use_memory"] = True
    elif state["complexity"] == "moderate" and age_days < 30:
        state["should_use_memory"] = True
    elif state["complexity"] == "complex" and age_days < 7:
        state["should_use_memory"] = True
    else:
        state["should_use_memory"] = False
        logger.info(f"Memory too old ({age_days} days) for {state['complexity']} query")

    return state


async def route_to_llm_node_async(state: ConversationState) -> ConversationState:
    """
    Node: Route to appropriate LLM based on complexity and memory with dynamic pricing.

    # TODO(phase2): connect to real pricing manager
    """
    from adam.pricing_manager import get_pricing_manager, CostOptimizer

    pricing_manager = get_pricing_manager()
    optimizer = CostOptimizer(pricing_manager)

    query_lower = state["query"].lower()
    is_coding = any(word in query_lower for word in ["implement", "code", "write", "create function", "build"])

    remaining_budget = 1.0  # Default $1 daily budget

    state["selected_model"] = await optimizer.select_optimal_model(
        complexity=state["complexity"],
        is_coding=is_coding,
        memory_confidence=state["memory_confidence"],
        remaining_daily_budget=remaining_budget
    )

    expected_tokens = {
        "simple": 200,
        "moderate": 500,
        "complex": 1500
    }.get(state["complexity"], 500)

    estimated_cost = pricing_manager.estimate_query_cost(state["selected_model"], expected_tokens)
    state["total_cost"] += estimated_cost

    logger.info(f"Selected model: {state['selected_model']} (estimated cost: ${estimated_cost:.4f})")

    should_alert, alert_msg = optimizer.should_alert_cost(estimated_cost, state["total_cost"])
    if should_alert:
        logger.warning(alert_msg)

    return state

def route_to_llm_node(state: ConversationState) -> ConversationState:
    """
    Synchronous wrapper for the async routing function.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(route_to_llm_node_async(state))
    finally:
        loop.close()


def generate_response_node(state: ConversationState) -> ConversationState:
    """
    Node: Generate response using selected LLM.

    # TODO(phase2): connect to real LLM client (adam.llm.client)
    Currently uses mock implementation.
    """
    context = ""
    if state["should_use_memory"] and state["memory_content"]:
        context = f"\nRelevant previous knowledge:\n{state['memory_content']}\n"

    if state["selected_model"] == "grok-3-mini-high":
        prompt = f"{context}\nQuestion: {state['query']}\nProvide a clear, accurate answer with reasoning."
    else:
        prompt = f"{context}\nQuestion: {state['query']}\nProvide deep analysis and comprehensive reasoning with expert-level implementation."

    # TODO(phase2): Replace with actual LLM integration
    # from adam.llm.client import UnifiedLLMClient
    # response = llm_client.generate(prompt, model=state["selected_model"])

    state["response"] = f"Based on {state['selected_model']}: Here's the answer to your query about {state['query'][:30]}..."

    return state


def handle_error_node(state: ConversationState) -> ConversationState:
    """
    Node: Handle errors and retries with graceful degradation.
    """
    state["retry_count"] += 1

    if state["retry_count"] >= 3:
        state["response"] = "I apologize, but I'm having trouble generating a response. Please try again later."
        return state

    if state["selected_model"] == "grok-4-reasoning":
        state["selected_model"] = "grok-3-mini-high"
    else:
        pass

    logger.warning(f"Retrying with {state['selected_model']} (attempt {state['retry_count']})")
    return state


def store_result_node(state: ConversationState) -> ConversationState:
    """
    Node: Store results intelligently.

    # TODO(phase2): connect to real memory network (adam.memory)
    """
    should_store = False

    if state["complexity"] == "complex":
        should_store = True
    elif not state["memory_found"] and state["total_cost"] > 0.005:
        should_store = True
    elif state["response"] and len(state["response"]) > 500:
        should_store = True

    if should_store:
        # TODO(phase2): Integrate with memory network
        # from adam.memory import ADAMMemoryAdvanced
        # memory.add_memory_with_references(
        #     query=state["query"],
        #     response=state["response"],
        #     metadata={"model": state["selected_model"], "cost": state["total_cost"]}
        # )
        logger.info(f"Storing interaction in memory (cost: ${state['total_cost']:.4f})")

    return state


def should_verify_memory(state: ConversationState) -> bool:
    """Edge condition: Determine if memory verification is needed."""
    return state["should_verify"]


def should_retry(state: ConversationState) -> bool:
    """Edge condition: Determine if retry is needed."""
    return state.get("error_message") is not None and state["retry_count"] < 3


def build_conversation_graph() -> StateGraph:
    """
    Build the LangGraph state machine.
    """
    workflow = StateGraph(ConversationState)

    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("check_memory", check_memory_node)
    workflow.add_node("verify_freshness", verify_memory_freshness_node)
    workflow.add_node("route_llm", route_to_llm_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("handle_error", handle_error_node)
    workflow.add_node("store_result", store_result_node)

    workflow.add_edge("analyze_query", "check_memory")

    workflow.add_conditional_edges(
        "check_memory",
        lambda x: "verify_freshness" if x["should_verify"] else "route_llm",
        {
            "verify_freshness": "verify_freshness",
            "route_llm": "route_llm"
        }
    )

    workflow.add_edge("verify_freshness", "route_llm")
    workflow.add_edge("route_llm", "generate_response")

    workflow.add_conditional_edges(
        "generate_response",
        lambda x: "handle_error" if x.get("error_message") else "store_result",
        {
            "handle_error": "handle_error",
            "store_result": "store_result"
        }
    )

    workflow.add_edge("handle_error", "generate_response")
    workflow.add_edge("store_result", END)

    workflow.set_entry_point("analyze_query")

    return workflow.compile()


class LangGraphConversationSystem:
    """
    Main interface for the LangGraph-based conversation system.
    """

    def __init__(self, memory_network=None, conversation_system=None):
        """
        Initialize the LangGraph system.

        Args:
            memory_network: ADAM's memory network for knowledge retrieval
            conversation_system: ADAM's conversation system for history
        """
        self.graph = build_conversation_graph()
        self.memory_network = memory_network
        self.conversation_system = conversation_system

    async def process_query(self, query: str, conversation_id: str) -> dict:
        """
        Process a query through the state machine.

        Args:
            query: The user's question or request
            conversation_id: Session ID for conversation tracking

        Returns:
            dict: Results including response, model used, cost, etc.
        """
        initial_state = ConversationState(
            query=query,
            complexity="simple",
            complexity_score=0.0,
            memory_found=False,
            memory_confidence=0.0,
            memory_ids=[],
            memory_content=None,
            memory_age_days=None,
            should_verify=False,
            should_use_memory=False,
            selected_model="grok-3-mini-high",
            response=None,
            total_cost=0.0,
            retry_count=0,
            error_message=None,
            conversation_id=conversation_id,
            timestamp=datetime.now()
        )

        result = await self.graph.ainvoke(initial_state)

        return {
            "response": result["response"],
            "model_used": result["selected_model"],
            "total_cost": result["total_cost"],
            "memory_used": result["should_use_memory"],
            "complexity": result["complexity"],
            "memory_confidence": result["memory_confidence"]
        }
