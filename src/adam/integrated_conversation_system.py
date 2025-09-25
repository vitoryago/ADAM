#!/usr/bin/env python3
"""
Integrated Conversation System - Bridges LangGraph with existing ADAM components

This module is the glue that connects the new LangGraph state machine with
ADAM's existing infrastructure. It provides:

1. Real memory integration: Connects to ADAM's ChromaDB-based memory system
2. Session management: Uses ADAM's conversation tracking
3. LLM integration: Manages connections to Grok-3, O3, and Claude Opus 4
4. Cost tracking: Monitors usage across all models
5. Voice output: Optional TTS for responses

The integration allows the sophisticated LangGraph flow to work seamlessly
with all existing ADAM features while adding new capabilities like:
- Dynamic model selection based on query complexity
- Memory confidence scoring and freshness checks
- Intelligent result storage
- Real-time cost optimization
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np

from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from .langgraph_conversation import (
    LangGraphConversationSystem,
    ConversationState,
    QueryComplexityAnalyzer,
    MemoryConfidenceScorer
)
from .memory.network import MemoryNetworkSystem
from .conversation_system import ConversationSystem
from .memory.conversation import ConversationAwareMemorySystem

import logging

logger = logging.getLogger(__name__)


class IntegratedADAMSystem:
    """
    Fully integrated ADAM system with LangGraph flow control.
    
    This class brings together all ADAM components under the sophisticated
    LangGraph state machine. It replaces the simple linear flow with a
    smart decision-making system that optimizes for both quality and cost.
    
    Key improvements over the original flow:
    - Analyzes query complexity before processing
    - Searches and evaluates memory with confidence scoring
    - Verifies memory freshness based on complexity
    - Routes to optimal LLM considering cost and requirements
    - Handles errors gracefully with model fallbacks
    - Stores only valuable interactions
    """
    
    def __init__(self, 
                 base_memory_system,
                 conversation_dir: str = "./conversations",
                 enable_voice: bool = True):
        """
        Initialize the integrated system.
        
        Args:
            base_memory_system: The base ADAMMemoryAdvanced system with ChromaDB
            conversation_dir: Directory for conversation storage
            enable_voice: Whether to enable voice output (TTS)
        """
        # Initialize core ADAM components
        # These are the existing systems we're enhancing with LangGraph
        self.base_memory = base_memory_system
        self.conversation_system = ConversationSystem(conversation_dir)
        
        # Memory network adds graph-based relationships between memories
        self.memory_network = MemoryNetworkSystem(
            self.base_memory, 
            self.conversation_system
        )
        
        # Conversation-aware memory adds session context
        self.cam_system = ConversationAwareMemorySystem(self.base_memory)
        self.cam_system.memory_network = self.memory_network
        
        # Initialize LLM connections
        self._init_llms()
        
        # Initialize analysis components from LangGraph
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.confidence_scorer = MemoryConfidenceScorer()
        
        # Voice settings for optional TTS output
        self.enable_voice = enable_voice
        
        # Cost tracking (will be overridden by dynamic pricing)
        self.total_cost = 0.0
        # Costs per token (divide per-million prices by 1M)
        self.cost_per_model = {
            "grok-4-fast-reasoning": 0.00001,         # $10/M tokens
            "grok-4-fast-non-reasoning": 0.0000065,   # $6.5/M tokens
            "grok-3-mini-reasoning-high": 0.0000004,  # $0.4/M tokens (kept for compatibility)
            "o3": 0.000005,                           # $5/M tokens
            "claude-opus-4": 0.000045                 # $45/M tokens
        }
    
    def _init_llms(self):
        """
        Initialize LLM instances.
        
        In production, this would create actual API clients for each provider:
        - X.AI client for Grok-3-mini-reasoning-high
        - OpenAI client for O3 (next-gen reasoning model)
        - Anthropic client for Claude Opus 4 (best for coding)
        
        Temperature is set to 0.7 for balanced creativity/accuracy.
        """
        self.llms = {
            "grok-4-fast-reasoning": None,        # TODO: Initialize X.AI client for fast reasoning
            "grok-4-fast-non-reasoning": None,    # TODO: Initialize X.AI client for fast non-reasoning
            "grok-3-mini-reasoning-high": None,   # TODO: Initialize X.AI client when available
            "o3": ChatOpenAI(model="o3", temperature=0.7),  # O3 reasoning model
            "claude-opus-4": ChatAnthropic(model="claude-3-opus-20240229", temperature=0.7)
        }
    
    async def process_query_with_langgraph(self, 
                                          query: str, 
                                          session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process query through the complete LangGraph pipeline.
        
        This method manually executes each node of the LangGraph state machine,
        giving us full control and visibility into the process. In production,
        you could also use the LangGraphConversationSystem class directly.
        
        Pipeline stages:
        1. Analyze query complexity
        2. Search memory with confidence scoring
        3. Verify memory freshness (if confidence > 0.7)
        4. Route to optimal LLM
        5. Generate response
        6. Store valuable results
        7. Track conversation and costs
        
        Args:
            query: User's question or request
            session_id: Optional session ID to continue existing conversation
            
        Returns:
            Dict containing response, model used, cost, and metadata
        """
        # Start or continue conversation session
        if not session_id:
            session_id = self.conversation_system.start_session(f"Query: {query[:50]}")
        
        # Initialize state with query
        state = await self._create_initial_state(query, session_id)
        
        # Execute LangGraph pipeline stages
        state = await self._analyze_query(state)
        state = await self._check_memory(state)
        
        # Conditional: Verify memory freshness if high confidence
        if state["should_verify"]:
            state = await self._verify_memory_freshness(state)
        
        state = await self._route_to_llm(state)
        state = await self._generate_response(state)
        
        # Store results if valuable (complex, novel, or expensive)
        await self._store_result(state)
        
        # Record in conversation system for history tracking
        exchange_id = self.conversation_system.record_exchange(
            query, 
            state["response"], 
            self._extract_topics(query)
        )
        
        # Optional voice output
        if self.enable_voice and state["response"]:
            await self._speak_response(state["response"])
        
        # Return clean results to caller
        return {
            "response": state["response"],
            "model_used": state["selected_model"],
            "cost": state["total_cost"],
            "memory_confidence": state["memory_confidence"],
            "complexity": state["complexity"],
            "exchange_id": exchange_id,
            "session_id": session_id
        }
    
    async def _create_initial_state(self, query: str, session_id: str) -> ConversationState:
        """
        Create initial conversation state.
        
        This initializes all state fields with default values. The state
        will be transformed as it flows through the pipeline nodes.
        """
        return ConversationState(
            query=query,
            complexity="simple",              # Will be updated by analyzer
            complexity_score=0.0,             # Confidence in complexity assessment
            memory_found=False,               # Will be updated by memory search
            memory_confidence=0.0,            # How confident in memory match
            memory_ids=[],                    # IDs of relevant memories
            memory_content=None,              # Best matching memory content
            memory_age_days=None,             # Age of memory in days
            should_verify=False,              # Whether to check freshness
            should_use_memory=False,          # Final decision on memory usage
            selected_model="grok-4-fast-non-reasoning",  # Default to fast non-reasoning
            response=None,                    # Generated response
            total_cost=0.0,                   # Accumulated cost
            retry_count=0,                    # Error retry counter
            error_message=None,               # Error details if any
            conversation_id=session_id,       # Link to conversation
            timestamp=datetime.now()          # Processing timestamp
        )
    
    async def _analyze_query(self, state: ConversationState) -> ConversationState:
        """
        Analyze query complexity.
        
        Uses the QueryComplexityAnalyzer to determine if this is a:
        - Simple query: Basic questions, definitions
        - Moderate query: How-to guides, standard problems  
        - Complex query: Debugging, optimization, architecture
        
        This classification drives all downstream decisions.
        """
        complexity, score = self.complexity_analyzer.analyze(state["query"])
        state["complexity"] = complexity
        state["complexity_score"] = score
        logger.info(f"Query complexity: {complexity} (score: {score})")
        return state
    
    async def _check_memory(self, state: ConversationState) -> ConversationState:
        """
        Check memory with real memory system.
        
        This integrates with ADAM's ChromaDB-based memory to find relevant
        past conversations. Unlike simple similarity search, we calculate
        a sophisticated confidence score that considers:
        - Semantic similarity
        - Query word overlap
        - Memory age
        - Response quality
        
        High confidence memories (>0.7) trigger freshness verification.
        """
        # Search ChromaDB for relevant memories
        search_results = self.base_memory.search(state["query"], n_results=5)
        
        if search_results:
            # Process the best match
            best_match = search_results[0]
            
            # Extract memory components
            memory_query = best_match.get('query', '')
            memory_response = best_match.get('response', '')
            distance = best_match.get('distance', 1.0)
            similarity = 1.0 - distance  # Convert distance to similarity
            
            # Calculate memory age from the memory network graph
            memory_id = best_match.get('id')
            age_days = 30  # Default if we can't determine age
            
            # Look up memory in the network graph for metadata
            if memory_id and memory_id in self.memory_network.memory_graph.nodes:
                node_data = self.memory_network.memory_graph.nodes[memory_id]['data']
                age_days = (datetime.now() - node_data.timestamp).days
            
            # Calculate comprehensive confidence score
            # This goes beyond simple similarity to evaluate trust
            confidence = self.confidence_scorer.calculate_confidence(
                state["query"],
                memory_query,
                memory_response,
                similarity,
                age_days
            )
            
            # Update state with memory findings
            state["memory_found"] = True
            state["memory_confidence"] = confidence
            state["memory_ids"] = [r.get('id', '') for r in search_results[:3]]  # Top 3 for references
            state["memory_content"] = memory_response
            state["memory_age_days"] = age_days
            
            # High confidence memories need freshness check
            # This prevents using outdated information for important queries
            state["should_verify"] = confidence > 0.7
            
            logger.info(f"Found memory with confidence {confidence:.2f}, age {age_days} days")
        else:
            # No memories found - will need fresh generation
            state["memory_found"] = False
            state["should_verify"] = False
            logger.info("No relevant memories found")
        
        return state
    
    async def _verify_memory_freshness(self, state: ConversationState) -> ConversationState:
        """
        Verify memory freshness based on complexity.
        
        Different query types have different freshness requirements:
        - Simple queries (what is X?): Can use 90-day old memories
        - Moderate queries (how to X?): Need <30-day old memories
        - Complex queries (debug X): Need very fresh <7-day memories
        
        This ensures we don't give outdated advice for rapidly
        changing technical topics.
        """
        age_days = state["memory_age_days"] or 0
        
        # Define freshness requirements by complexity
        # More complex = needs fresher information
        freshness_limits = {
            "simple": 90,     # Facts remain stable
            "moderate": 30,   # Procedures change monthly
            "complex": 7      # Complex issues need current context
        }
        
        limit = freshness_limits.get(state["complexity"], 30)
        state["should_use_memory"] = age_days <= limit
        
        if not state["should_use_memory"]:
            logger.info(f"Memory too old ({age_days} days) for {state['complexity']} query")
        
        return state
    
    async def _route_to_llm(self, state: ConversationState) -> ConversationState:
        """
        Route to appropriate LLM based on multiple factors.
        
        Model selection strategy:
        1. Simple/Moderate queries → Grok-3 (fast, cheap, good reasoning)
        2. Complex non-coding → O3 (deep analysis, complex reasoning)
        3. Complex coding → Claude Opus 4 (best code generation)
        
        Special optimization: If we have high-confidence memory (>0.9),
        we can use a cheaper model even for complex queries since we're
        augmenting with proven past knowledge.
        """
        # Base routing on complexity using fast models
        if state["complexity"] == "simple":
            state["selected_model"] = "grok-4-fast-non-reasoning"
        elif state["complexity"] == "moderate":
            state["selected_model"] = "grok-4-fast-non-reasoning"
        else:  # complex
            # Detect coding tasks for Claude specialization
            query_lower = state["query"].lower()
            coding_keywords = ["implement", "code", "write", "create function", "build"]
            
            if any(word in query_lower for word in coding_keywords):
                state["selected_model"] = "grok-4-fast-reasoning"  # Fast reasoning for code
            else:
                state["selected_model"] = "grok-4-fast-reasoning"  # Fast reasoning for complex
        
        # Cost optimization: High-confidence memory allows model downgrade
        # This saves money without sacrificing quality
        if (state.get("should_use_memory", False) and
            state["memory_confidence"] > 0.9 and
            state["complexity"] == "complex"):
            state["selected_model"] = "grok-4-fast-non-reasoning"
            logger.info("Using simpler model due to high memory confidence")
        
        logger.info(f"Selected model: {state['selected_model']}")
        return state
    
    async def _generate_response(self, state: ConversationState) -> ConversationState:
        """
        Generate response using selected LLM.
        
        This method:
        1. Builds appropriate context including memory if available
        2. Calls the selected LLM with proper formatting
        3. Calculates actual cost based on token usage
        4. Handles errors gracefully
        
        Each model has its own prompting style for optimal results.
        """
        try:
            # Build message context for LLM
            messages = []
            
            # System message with optional memory context
            system_msg = "You are ADAM, an AI assistant specializing in software development."
            
            # Include memory as context if we decided to trust it
            if state.get("should_use_memory") and state["memory_content"]:
                system_msg += f"\n\nRelevant context from memory:\n{state['memory_content']}"
            
            messages.append(SystemMessage(content=system_msg))
            
            # Add user query
            messages.append(HumanMessage(content=state["query"]))
            
            # Get the appropriate LLM client
            llm = self.llms[state["selected_model"]]
            
            # Generate response based on model type
            if state["selected_model"] in ["grok-4-fast-reasoning", "grok-4-fast-non-reasoning", "grok-3-mini-reasoning-high"]:
                # TODO: Implement X.AI Grok integration
                # Grok excels at step-by-step reasoning
                response = f"[Grok-3 would provide reasoning here for: {state['query'][:50]}...]"
            else:
                # OpenAI (O3) and Anthropic (Claude) use similar interfaces
                response = await llm.ainvoke(messages)
                response = response.content
            
            state["response"] = response
            
            # Calculate actual cost based on token usage
            # Rough estimate: 1 word ≈ 1.3 tokens
            estimated_tokens = len(response.split()) * 1.3
            cost_per_token = self.cost_per_model[state["selected_model"]]
            state["total_cost"] = estimated_tokens * cost_per_token
            self.total_cost += state["total_cost"]
            
            logger.info(f"Generated {len(response)} char response, cost: ${state['total_cost']:.4f}")
            
        except Exception as e:
            # Error handling - will trigger retry logic in state machine
            logger.error(f"Error generating response: {e}")
            state["error_message"] = str(e)
            state["response"] = None
        
        return state
    
    async def _store_result(self, state: ConversationState) -> ConversationState:
        """
        Store valuable results in memory.
        
        Not every interaction is worth storing. We save:
        1. All complex query responses (valuable knowledge)
        2. Novel information (no existing memory + substantial response)
        3. Expensive generations (high cost = high value)
        
        This selective approach keeps memory quality high and costs low.
        """
        if not state["response"]:
            return state
        
        # Evaluate storage criteria
        should_store = False
        
        # Always store complex query results
        if state["complexity"] == "complex":
            should_store = True
        # Store novel information that's substantial
        elif not state["memory_found"] and len(state["response"]) > 200:
            should_store = True
        # Store expensive generations (they represent investment)
        elif state["total_cost"] > 0.005:  # More than half a cent
            should_store = True
        
        if should_store:
            # Extract topics
            topics = self._extract_topics(state["query"])
            
            # Determine memory type
            if "error" in state["query"].lower() or "fix" in state["query"].lower():
                memory_type = "error_solution"
            elif "implement" in state["query"].lower():
                memory_type = "code_implementation"
            elif "how" in state["query"].lower():
                memory_type = "how_to_guide"       # Tutorials
            else:
                memory_type = "explanation"        # General knowledge
            
            # Store in memory network with references
            # References link to related memories found during search
            memory_id = self.memory_network.add_memory_with_references(
                query=state["query"],
                response=state["response"],
                memory_type=memory_type,
                topics=topics,
                potential_references=state["memory_ids"][:3] if state["memory_found"] else None
            )
            
            logger.info(f"Stored as memory {memory_id} (cost: ${state['total_cost']:.4f})")
        
        return state
    
    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract topics from text for memory categorization.
        
        This simple implementation uses keyword matching. In production,
        you could use:
        - NLP libraries for entity extraction
        - LLM-based topic extraction
        - TF-IDF for keyword importance
        
        Topics help with memory retrieval and clustering.
        """
        # Common technical topics to detect
        keywords = [
            "python", "sql", "api", "debug", "error", "optimize", 
            "performance", "async", "database", "deploy", "docker",
            "kubernetes", "microservices", "testing", "security"
        ]
        
        text_lower = text.lower()
        topics = [kw for kw in keywords if kw in text_lower]
        
        # Add query complexity as a topic for filtering
        complexity, _ = self.complexity_analyzer.analyze(text)
        topics.append(f"complexity:{complexity}")
        
        return topics[:5]  # Limit to 5 most relevant topics
    
    async def _speak_response(self, response: str):
        """
        Speak response using TTS (Text-to-Speech).
        
        This provides an optional voice interface for ADAM, making it
        feel more like a conversational AI assistant. The response is
        truncated to 500 characters to avoid overly long speech.
        
        Note: Requires pyttsx3 to be installed and system TTS available.
        """
        try:
            import pyttsx3
            engine = pyttsx3.init()
            
            # Truncate for reasonable speech length
            speech_text = response[:500]
            if len(response) > 500:
                speech_text += "... (response truncated for speech)"
            
            engine.say(speech_text)
            engine.runAndWait()
        except Exception as e:
            # Don't fail the whole request if TTS fails
            logger.warning(f"Voice output failed: {e}")
    
    def get_cost_report(self) -> Dict[str, float]:
        """
        Get cost breakdown for monitoring.
        
        Returns:
            Dict with total cost, average per query, and model pricing
            
        This helps track spending and optimize model selection over time.
        """
        stats = self.conversation_system.get_stats()
        total_exchanges = stats.get('total_exchanges', 0)
        
        return {
            "total_cost": self.total_cost,
            "average_per_query": self.total_cost / max(1, total_exchanges),
            "model_costs": self.cost_per_model,
            "queries_processed": total_exchanges
        }


# Example usage
async def main():
    """Demo the integrated system"""
    from adam_memory_advanced import ADAMMemoryAdvanced
    
    # Initialize base memory
    base_memory = ADAMMemoryAdvanced()
    
    # Create integrated system
    adam = IntegratedADAMSystem(base_memory)
    
    # Test queries
    queries = [
        "What is a Python decorator?",
        "How do I optimize a slow SQL query with multiple joins?",
        "Debug this async race condition in my microservices architecture"
    ]
    
    print("=== ADAM Integrated System Demo ===")
    print("Watch how different queries route to different models...\n")
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = await adam.process_query_with_langgraph(query)
        
        print(f"Response preview: {result['response'][:150]}...")
        print(f"Model used: {result['model_used']}")
        print(f"Cost: ${result['cost']:.4f}")
        print(f"Memory confidence: {result['memory_confidence']:.2f}")
        print(f"Complexity: {result['complexity']}")
        print("-" * 70)
    
    # Show cost report
    cost_report = adam.get_cost_report()
    print(f"\n=== Cost Report ===")
    print(f"Total cost: ${cost_report['total_cost']:.4f}")
    print(f"Average per query: ${cost_report['average_per_query']:.4f}")
    print(f"Queries processed: {cost_report['queries_processed']}")


if __name__ == "__main__":
    asyncio.run(main())