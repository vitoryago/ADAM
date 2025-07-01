#!/usr/bin/env python3
"""
LangGraph-based Conversation State Machine for ADAM

This module implements a sophisticated state machine that:
1. Analyzes query complexity
2. Checks memory with confidence scoring
3. Verifies memory freshness
4. Routes to appropriate LLM based on complexity
5. Handles retries and fallbacks
6. Stores results intelligently
"""

from typing import TypedDict, Optional, List, Literal, Annotated
from datetime import datetime, timedelta
import numpy as np
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import logging
import asyncio

logger = logging.getLogger(__name__)


class ConversationState(TypedDict):
    """State definition for the conversation flow"""
    # Input
    query: str
    
    # Analysis results
    complexity: Literal["simple", "moderate", "complex"]
    complexity_score: float  # 0.0 to 1.0
    
    # Memory search results
    memory_found: bool
    memory_confidence: float  # 0.0 to 1.0
    memory_ids: List[str]
    memory_content: Optional[str]
    memory_age_days: Optional[int]
    
    # Decision flags
    should_verify: bool
    should_use_memory: bool
    
    # LLM routing
    selected_model: Literal["grok-3-mini-reasoning-high", "o3", "claude-opus-4"]
    
    # Response and metadata
    response: Optional[str]
    total_cost: float
    retry_count: int
    error_message: Optional[str]
    
    # Tracking
    conversation_id: str
    timestamp: datetime


class QueryComplexityAnalyzer:
    """Analyzes query complexity to determine routing"""
    
    def __init__(self):
        self.complexity_indicators = {
            "simple": [
                "what is", "how to", "define", "explain", "list",
                "when", "where", "who", "simple", "basic"
            ],
            "complex": [
                "debug", "optimize", "implement", "architect", "design",
                "integrate", "scale", "performance", "production",
                "multi", "distributed", "concurrent", "async"
            ],
            "technical_depth": [
                "algorithm", "complexity", "theorem", "proof",
                "mathematical", "formal", "rigorous"
            ]
        }
    
    def analyze(self, query: str) -> tuple[str, float]:
        """
        Analyze query complexity
        Returns: (complexity_level, confidence_score)
        """
        query_lower = query.lower()
        
        # Count indicators
        simple_count = sum(1 for word in self.complexity_indicators["simple"] 
                         if word in query_lower)
        complex_count = sum(1 for word in self.complexity_indicators["complex"] 
                          if word in query_lower)
        technical_count = sum(1 for word in self.complexity_indicators["technical_depth"] 
                            if word in query_lower)
        
        # Calculate scores
        word_count = len(query.split())
        
        # Heuristics for complexity
        if word_count < 10 and simple_count > 0 and complex_count == 0:
            return "simple", 0.8
        elif complex_count >= 2 or technical_count >= 1 or word_count > 50:
            return "complex", 0.85
        elif "?" in query and query.count("?") > 1:
            # Multiple questions usually indicate complexity
            return "complex", 0.7
        else:
            return "moderate", 0.6


class MemoryConfidenceScorer:
    """Calculates confidence in memory matches"""
    
    def calculate_confidence(self, 
                           query: str,
                           memory_query: str,
                           memory_response: str,
                           similarity_score: float,
                           age_days: int) -> float:
        """
        Calculate confidence score for a memory match
        
        Factors:
        - Semantic similarity
        - Query overlap
        - Age decay
        - Response length/quality
        """
        # Base confidence from similarity
        confidence = similarity_score
        
        # Adjust for query overlap
        query_words = set(query.lower().split())
        memory_words = set(memory_query.lower().split())
        overlap = len(query_words & memory_words) / len(query_words)
        confidence = confidence * 0.7 + overlap * 0.3
        
        # Age decay factor
        if age_days < 7:
            age_factor = 1.0
        elif age_days < 30:
            age_factor = 0.9
        elif age_days < 90:
            age_factor = 0.7
        else:
            age_factor = 0.5
        
        confidence *= age_factor
        
        # Response quality factor (longer, detailed responses score higher)
        response_length = len(memory_response)
        if response_length > 500:
            quality_factor = 1.0
        elif response_length > 200:
            quality_factor = 0.9
        else:
            quality_factor = 0.8
        
        confidence *= quality_factor
        
        return min(confidence, 1.0)


def analyze_query_node(state: ConversationState) -> ConversationState:
    """Node: Analyze query complexity"""
    analyzer = QueryComplexityAnalyzer()
    complexity, score = analyzer.analyze(state["query"])
    
    state["complexity"] = complexity
    state["complexity_score"] = score
    
    logger.info(f"Query complexity: {complexity} (confidence: {score})")
    return state


def check_memory_node(state: ConversationState) -> ConversationState:
    """Node: Check memory with confidence scoring"""
    # This would integrate with your existing memory system
    # For now, we'll simulate the integration
    
    from src.adam import MemoryNetworkSystem
    
    # Simulate memory search
    # In real implementation, this would call memory_network.search()
    query = state["query"]
    
    # Mock implementation - replace with actual memory search
    # memories = memory_network.search(query, n_results=5)
    
    # For demonstration, let's simulate finding a memory
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
    
    # Determine if we should verify the memory
    if state["memory_found"] and state["memory_confidence"] > 0.7:
        state["should_verify"] = True
    else:
        state["should_verify"] = False
    
    logger.info(f"Memory found: {state['memory_found']}, "
                f"confidence: {state['memory_confidence']}")
    return state


def verify_memory_freshness_node(state: ConversationState) -> ConversationState:
    """Node: Verify if memory is still valid/fresh"""
    if not state["should_verify"]:
        state["should_use_memory"] = state["memory_found"]
        return state
    
    # Check memory age and context
    age_days = state["memory_age_days"] or 0
    
    # Freshness rules
    if state["complexity"] == "simple" and age_days < 90:
        # Simple queries can use older memories
        state["should_use_memory"] = True
    elif state["complexity"] == "moderate" and age_days < 30:
        # Moderate queries need fresher memories
        state["should_use_memory"] = True
    elif state["complexity"] == "complex" and age_days < 7:
        # Complex queries need very fresh memories
        state["should_use_memory"] = True
    else:
        # Memory too old for the query complexity
        state["should_use_memory"] = False
        logger.info(f"Memory too old ({age_days} days) for {state['complexity']} query")
    
    return state


async def route_to_llm_node_async(state: ConversationState) -> ConversationState:
    """Node: Route to appropriate LLM based on complexity and memory with dynamic pricing"""
    
    # Import pricing manager
    from .pricing_manager import get_pricing_manager, CostOptimizer
    
    pricing_manager = get_pricing_manager()
    optimizer = CostOptimizer(pricing_manager)
    
    # Check if it's a coding task
    query_lower = state["query"].lower()
    is_coding = any(word in query_lower for word in ["implement", "code", "write", "create function", "build"])
    
    # Get remaining daily budget (would come from tracking system in production)
    remaining_budget = 1.0  # Default $1 daily budget
    
    # Select optimal model based on real-time pricing
    state["selected_model"] = await optimizer.select_optimal_model(
        complexity=state["complexity"],
        is_coding=is_coding,
        memory_confidence=state["memory_confidence"],
        remaining_daily_budget=remaining_budget
    )
    
    # Estimate cost based on expected tokens
    expected_tokens = {
        "simple": 200,
        "moderate": 500,
        "complex": 1500
    }.get(state["complexity"], 500)
    
    # Get real-time cost estimate
    estimated_cost = pricing_manager.estimate_query_cost(state["selected_model"], expected_tokens)
    state["total_cost"] += estimated_cost
    
    logger.info(f"Selected model: {state['selected_model']} (estimated cost: ${estimated_cost:.4f})")
    
    # Check for cost alerts
    should_alert, alert_msg = optimizer.should_alert_cost(estimated_cost, state["total_cost"])
    if should_alert:
        logger.warning(alert_msg)
    
    return state

# Wrapper for backward compatibility
def route_to_llm_node(state: ConversationState) -> ConversationState:
    """Synchronous wrapper for the async routing function"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(route_to_llm_node_async(state))
    finally:
        loop.close()


def generate_response_node(state: ConversationState) -> ConversationState:
    """Node: Generate response using selected LLM"""
    
    # Build context from memory if available
    context = ""
    if state["should_use_memory"] and state["memory_content"]:
        context = f"\nRelevant previous knowledge:\n{state['memory_content']}\n"
    
    # Prepare prompt based on model
    if state["selected_model"] == "grok-3-mini-reasoning-high":
        prompt = f"{context}\nQuestion: {state['query']}\nProvide a clear, accurate answer with reasoning."
    elif state["selected_model"] == "o3":
        prompt = f"{context}\nQuestion: {state['query']}\nProvide deep analysis and comprehensive reasoning."
    else:  # claude-opus-4
        prompt = f"{context}\nQuestion: {state['query']}\nProvide expert-level code implementation with best practices."
    
    # Simulate LLM call - in production, this would call actual models
    # response = llm.generate(prompt, model=state["selected_model"])
    
    # Mock response for demonstration
    state["response"] = f"Based on {state['selected_model']}: Here's the answer to your query about {state['query'][:30]}..."
    
    return state


def handle_error_node(state: ConversationState) -> ConversationState:
    """Node: Handle errors and retries"""
    state["retry_count"] += 1
    
    if state["retry_count"] >= 3:
        state["response"] = "I apologize, but I'm having trouble generating a response. Please try again later."
        return state
    
    # Fallback to simpler model
    if state["selected_model"] == "claude-opus-4":
        state["selected_model"] = "o3"
    elif state["selected_model"] == "o3":
        state["selected_model"] = "grok-3-mini-reasoning-high"
    else:
        # Already at simplest model, try one more time
        pass
    
    logger.warning(f"Retrying with {state['selected_model']} (attempt {state['retry_count']})")
    return state


def store_result_node(state: ConversationState) -> ConversationState:
    """Node: Store results intelligently"""
    
    # Determine if this interaction should be stored
    should_store = False
    
    # Store if:
    # 1. Complex query with substantial response
    # 2. No existing memory was found
    # 3. The response cost was high (indicating valuable generation)
    
    if state["complexity"] == "complex":
        should_store = True
    elif not state["memory_found"] and state["total_cost"] > 0.005:
        should_store = True
    elif state["response"] and len(state["response"]) > 500:
        should_store = True
    
    if should_store:
        # In production, this would call memory_network.add_memory_with_references()
        logger.info(f"Storing interaction in memory (cost: ${state['total_cost']:.4f})")
    
    return state


def should_verify_memory(state: ConversationState) -> bool:
    """Edge: Determine if memory verification is needed"""
    return state["should_verify"]


def should_retry(state: ConversationState) -> bool:
    """Edge: Determine if retry is needed"""
    return state.get("error_message") is not None and state["retry_count"] < 3


def build_conversation_graph() -> StateGraph:
    """Build the LangGraph state machine"""
    
    # Create the graph
    workflow = StateGraph(ConversationState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("check_memory", check_memory_node)
    workflow.add_node("verify_freshness", verify_memory_freshness_node)
    workflow.add_node("route_llm", route_to_llm_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("handle_error", handle_error_node)
    workflow.add_node("store_result", store_result_node)
    
    # Add edges
    workflow.add_edge("analyze_query", "check_memory")
    
    # Conditional edge for memory verification
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
    
    # Conditional edge for error handling
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
    
    # Set entry point
    workflow.set_entry_point("analyze_query")
    
    return workflow.compile()


class LangGraphConversationSystem:
    """Main interface for the LangGraph-based conversation system"""
    
    def __init__(self, memory_network=None, conversation_system=None):
        self.graph = build_conversation_graph()
        self.memory_network = memory_network
        self.conversation_system = conversation_system
    
    async def process_query(self, query: str, conversation_id: str) -> dict:
        """Process a query through the state machine"""
        
        # Initialize state
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
            selected_model="mistral",
            response=None,
            total_cost=0.0,
            retry_count=0,
            error_message=None,
            conversation_id=conversation_id,
            timestamp=datetime.now()
        )
        
        # Run the graph
        result = await self.graph.ainvoke(initial_state)
        
        # Extract key results
        return {
            "response": result["response"],
            "model_used": result["selected_model"],
            "total_cost": result["total_cost"],
            "memory_used": result["should_use_memory"],
            "complexity": result["complexity"],
            "memory_confidence": result["memory_confidence"]
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def demo():
        # Create the system
        lg_system = LangGraphConversationSystem()
        
        # Test queries
        test_queries = [
            "What is a Python decorator?",
            "Debug this complex async race condition in my distributed system",
            "How do I optimize a slow SQL query?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            result = await lg_system.process_query(query, "demo_session")
            print(f"Complexity: {result['complexity']}")
            print(f"Model used: {result['model_used']}")
            print(f"Cost: ${result['total_cost']:.4f}")
            print(f"Memory used: {result['memory_used']}")
            print(f"Response: {result['response'][:100]}...")
    
    asyncio.run(demo())