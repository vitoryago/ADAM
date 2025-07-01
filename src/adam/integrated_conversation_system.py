#!/usr/bin/env python3
"""
Integrated Conversation System - Bridges LangGraph with existing ADAM components

This module connects:
- LangGraph state machine for flow control
- Existing MemoryNetworkSystem for memory storage/retrieval
- ConversationSystem for session management
- Real LLMs (Ollama, OpenAI, Anthropic)
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
from .memory_network import MemoryNetworkSystem
from .conversation_system import ConversationSystem
from .conversation_aware_memory import ConversationAwareMemorySystem

import logging

logger = logging.getLogger(__name__)


class IntegratedADAMSystem:
    """
    Fully integrated ADAM system with LangGraph flow control
    """
    
    def __init__(self, 
                 base_memory_system,
                 conversation_dir: str = "./conversations",
                 enable_voice: bool = True):
        """
        Initialize the integrated system
        
        Args:
            base_memory_system: The base ADAMMemoryAdvanced system
            conversation_dir: Directory for conversation storage
            enable_voice: Whether to enable voice output
        """
        # Initialize core components
        self.base_memory = base_memory_system
        self.conversation_system = ConversationSystem(conversation_dir)
        self.memory_network = MemoryNetworkSystem(
            self.base_memory, 
            self.conversation_system
        )
        self.cam_system = ConversationAwareMemorySystem(self.base_memory)
        self.cam_system.memory_network = self.memory_network
        
        # Initialize LLMs
        self._init_llms()
        
        # Initialize analyzers
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.confidence_scorer = MemoryConfidenceScorer()
        
        # Voice settings
        self.enable_voice = enable_voice
        
        # Cost tracking
        self.total_cost = 0.0
        # Dynamic pricing will override these
        self.cost_per_model = {
            "grok-3-mini-reasoning-high": 0.0002,  # Per token estimate
            "o3": 0.0375,  # O3 actual pricing
            "claude-opus-4": 0.045  # Claude Opus 4 actual pricing
        }
    
    def _init_llms(self):
        """Initialize LLM instances"""
        # These would be initialized with actual API clients
        # For now, using placeholders - in production, use:
        # - X.AI for Grok models
        # - OpenAI for O1 models  
        # - Anthropic for Claude
        self.llms = {
            "grok-3-mini-reasoning-high": None,  # Will need X.AI client
            "o3": ChatOpenAI(model="o3", temperature=0.7),  # O3 model
            "claude-opus-4": ChatAnthropic(model="claude-3-opus-20240229", temperature=0.7)
        }
    
    async def process_query_with_langgraph(self, 
                                          query: str, 
                                          session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process query through the complete LangGraph pipeline
        """
        # Start or continue session
        if not session_id:
            session_id = self.conversation_system.start_session(f"Query: {query[:50]}")
        
        # Initialize state
        state = await self._create_initial_state(query, session_id)
        
        # Run through state machine nodes
        state = await self._analyze_query(state)
        state = await self._check_memory(state)
        
        if state["should_verify"]:
            state = await self._verify_memory_freshness(state)
        
        state = await self._route_to_llm(state)
        state = await self._generate_response(state)
        
        # Store results if valuable
        await self._store_result(state)
        
        # Record in conversation system
        exchange_id = self.conversation_system.record_exchange(
            query, 
            state["response"], 
            self._extract_topics(query)
        )
        
        # Voice output if enabled
        if self.enable_voice and state["response"]:
            await self._speak_response(state["response"])
        
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
        """Create initial conversation state"""
        return ConversationState(
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
            conversation_id=session_id,
            timestamp=datetime.now()
        )
    
    async def _analyze_query(self, state: ConversationState) -> ConversationState:
        """Analyze query complexity"""
        complexity, score = self.complexity_analyzer.analyze(state["query"])
        state["complexity"] = complexity
        state["complexity_score"] = score
        logger.info(f"Query complexity: {complexity} (score: {score})")
        return state
    
    async def _check_memory(self, state: ConversationState) -> ConversationState:
        """Check memory with real memory system"""
        # Search memories
        search_results = self.base_memory.search(state["query"], n_results=5)
        
        if search_results:
            # Calculate confidence for best match
            best_match = search_results[0]
            
            # Extract memory details
            memory_query = best_match.get('query', '')
            memory_response = best_match.get('response', '')
            distance = best_match.get('distance', 1.0)
            similarity = 1.0 - distance
            
            # Get memory age
            memory_id = best_match.get('id')
            age_days = 30  # Default
            if memory_id and memory_id in self.memory_network.memory_graph.nodes:
                node = self.memory_network.memory_graph.nodes[memory_id]['data']
                age_days = (datetime.now() - node.timestamp).days
            
            # Calculate confidence
            confidence = self.confidence_scorer.calculate_confidence(
                state["query"],
                memory_query,
                memory_response,
                similarity,
                age_days
            )
            
            state["memory_found"] = True
            state["memory_confidence"] = confidence
            state["memory_ids"] = [r.get('id', '') for r in search_results[:3]]
            state["memory_content"] = memory_response
            state["memory_age_days"] = age_days
            
            # Should verify if high confidence
            state["should_verify"] = confidence > 0.7
            
            logger.info(f"Found memory with confidence {confidence:.2f}, age {age_days} days")
        else:
            state["memory_found"] = False
            state["should_verify"] = False
            logger.info("No relevant memories found")
        
        return state
    
    async def _verify_memory_freshness(self, state: ConversationState) -> ConversationState:
        """Verify memory freshness based on complexity"""
        age_days = state["memory_age_days"] or 0
        
        # Freshness thresholds by complexity
        freshness_limits = {
            "simple": 90,
            "moderate": 30,
            "complex": 7
        }
        
        limit = freshness_limits.get(state["complexity"], 30)
        state["should_use_memory"] = age_days <= limit
        
        if not state["should_use_memory"]:
            logger.info(f"Memory too old ({age_days} days) for {state['complexity']} query")
        
        return state
    
    async def _route_to_llm(self, state: ConversationState) -> ConversationState:
        """Route to appropriate LLM"""
        # Model routing based on complexity
        if state["complexity"] == "simple":
            state["selected_model"] = "grok-3-mini-reasoning-high"
        elif state["complexity"] == "moderate":
            state["selected_model"] = "grok-3-mini-reasoning-high"
        else:  # complex
            # Check if it's a coding task
            query_lower = state["query"].lower()
            if any(word in query_lower for word in ["implement", "code", "write", "create function", "build"]):
                state["selected_model"] = "claude-opus-4"
            else:
                state["selected_model"] = "o3"
        
        # Memory bonus - if we have excellent memory, we can use simpler model
        if state.get("should_use_memory", False) and state["memory_confidence"] > 0.9 and state["complexity"] == "complex":
            state["selected_model"] = "grok-3-mini-reasoning-high"
            logger.info("Using simpler model due to high memory confidence")
        
        logger.info(f"Selected model: {state['selected_model']}")
        return state
    
    async def _generate_response(self, state: ConversationState) -> ConversationState:
        """Generate response using selected LLM"""
        try:
            # Build context
            messages = []
            
            # System message
            system_msg = "You are ADAM, an AI assistant specializing in software development."
            if state.get("should_use_memory") and state["memory_content"]:
                system_msg += f"\n\nRelevant context from memory:\n{state['memory_content']}"
            messages.append(SystemMessage(content=system_msg))
            
            # User query
            messages.append(HumanMessage(content=state["query"]))
            
            # Get LLM
            llm = self.llms[state["selected_model"]]
            
            # Generate response
            if state["selected_model"] == "grok-3-mini-reasoning-high":
                # Grok would need custom implementation
                # For now, mock the response
                response = f"[Grok-3 would provide reasoning here for: {state['query'][:50]}...]"
            else:
                # OpenAI/Anthropic style
                response = await llm.ainvoke(messages)
                response = response.content
            
            state["response"] = response
            
            # Calculate cost (rough estimate based on response length)
            tokens = len(response.split()) * 1.3  # Rough token estimate
            state["total_cost"] = tokens * self.cost_per_model[state["selected_model"]]
            self.total_cost += state["total_cost"]
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state["error_message"] = str(e)
            state["response"] = None
        
        return state
    
    async def _store_result(self, state: ConversationState) -> ConversationState:
        """Store valuable results in memory"""
        if not state["response"]:
            return state
        
        # Determine if worth storing
        should_store = False
        
        if state["complexity"] == "complex":
            should_store = True
        elif not state["memory_found"] and len(state["response"]) > 200:
            should_store = True
        elif state["total_cost"] > 0.005:  # Expensive generation
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
                memory_type = "how_to_guide"
            else:
                memory_type = "explanation"
            
            # Add to memory network
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
        """Extract topics from text"""
        # Simple keyword extraction
        keywords = ["python", "sql", "api", "debug", "error", "optimize", 
                   "performance", "async", "database", "deploy"]
        
        text_lower = text.lower()
        topics = [kw for kw in keywords if kw in text_lower]
        
        # Add complexity as topic
        complexity, _ = self.complexity_analyzer.analyze(text)
        topics.append(complexity)
        
        return topics[:5]  # Limit to 5 topics
    
    async def _speak_response(self, response: str):
        """Speak response using TTS"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(response[:500])  # Limit length for voice
            engine.runAndWait()
        except Exception as e:
            logger.warning(f"Voice output failed: {e}")
    
    def get_cost_report(self) -> Dict[str, float]:
        """Get cost breakdown"""
        return {
            "total_cost": self.total_cost,
            "average_per_query": self.total_cost / max(1, self.conversation_system.get_stats()['total_exchanges']),
            "model_costs": self.cost_per_model
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
    
    print("=== ADAM Integrated System Demo ===\n")
    
    for query in queries:
        print(f"Query: {query}")
        result = await adam.process_query_with_langgraph(query)
        
        print(f"Response: {result['response'][:200]}...")
        print(f"Model: {result['model_used']}")
        print(f"Cost: ${result['cost']:.4f}")
        print(f"Memory confidence: {result['memory_confidence']:.2f}")
        print(f"Complexity: {result['complexity']}")
        print("-" * 50 + "\n")
    
    # Show cost report
    cost_report = adam.get_cost_report()
    print(f"Total cost: ${cost_report['total_cost']:.4f}")
    print(f"Average per query: ${cost_report['average_per_query']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())