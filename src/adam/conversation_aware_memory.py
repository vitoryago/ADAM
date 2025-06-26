#!/usr/bin/env python3
"""
Conversation-Aware Memory System - Bridging conversations with memories
This integrates the conversation system with the memory network for seamless operation
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

from .conversation_system import ConversationSystem
from .memory_network import MemoryNetworkSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationAwareMemorySystem:
    """
    Bridges the conversation system with memory network
    
    This class ensures that:
    1. Every memory is linked to its conversation context
    2. Conversations can be continued with full memory context
    3. The memory network can access conversation metadata
    """
    
    def __init__(self, base_memory_system, storage_path: str = "./adam_memory_advanced"):
        """
        Initialize the conversation-aware memory system
        
        Args:
            base_memory_system: The original ADAMMemoryAdvanced system
            storage_path: Base path for all storage
        """
        # Initialize conversation system
        self.conversation_system = ConversationSystem(
            storage_path=f"{storage_path}/conversations"
        )
        
        # Initialize memory network with conversation awareness
        self.memory_network = MemoryNetworkSystem(
            base_memory_system=base_memory_system,
            conversation_system=self.conversation_system
        )
        
        # Track current conversation context
        self.current_context = {}
        
        logger.info("Initialized conversation-aware memory system")
    
    def process_interaction(self, query: str, response: str, 
                          topics: List[str], 
                          generation_cost: float,
                          model_used: str,
                          context: Optional[Dict] = None) -> Tuple[str, str]:
        """
        Process a complete interaction with conversation and memory tracking
        
        This is the main entry point for all user-ADAM interactions
        
        Args:
            query: User's input
            response: ADAM's response
            topics: Topics discussed
            generation_cost: Cost of generating the response
            model_used: Which model was used
            context: Optional context (screen content, etc.)
            
        Returns:
            Tuple of (exchange_id, memory_id or None)
        """
        # Update current context
        if context:
            self.current_context.update(context)
        
        # Determine if this should be stored in memory
        memory_worthy = self._evaluate_memory_worthiness(
            query, response, generation_cost, topics
        )
        
        memory_id = None
        if memory_worthy:
            # Store in memory network with conversation context
            memory_id = self.memory_network.add_memory_with_references(
                query=query,
                response=response,
                memory_type=self._classify_memory_type(query, response),
                topics=topics
            )
        
        # Record in conversation system
        exchange_id = self.conversation_system.record_exchange(
            query=query,
            response=response,
            topics=topics,
            context=self.current_context,
            stored_in_memory=memory_worthy,
            memory_id=memory_id
        )
        
        logger.info(f"Processed interaction - Exchange: {exchange_id}, Memory: {memory_id}")
        
        return exchange_id, memory_id
    
    def continue_conversation(self, topic: str) -> Tuple[str, List[str]]:
        """
        Continue a previous conversation with full context
        
        Args:
            topic: Topic to continue discussing
            
        Returns:
            Tuple of (recap_text, memory_ids_to_load)
        """
        # Get conversation continuity
        parent_session_id, new_session_id = self.conversation_system.continue_conversation(topic)
        
        if not parent_session_id:
            return f"Starting a new conversation about {topic}.", []
        
        # Get memory context from the thread
        recap, memory_ids = self.memory_network.continue_thread(topic)
        
        # Add conversation context to recap
        parent_summary = self.conversation_system.get_session_summary(parent_session_id)
        
        enhanced_recap = f"{recap}\n\n"
        enhanced_recap += f"Our last conversation was {self._format_time_ago(parent_summary['start_time'])}"
        
        if parent_summary['duration_minutes']:
            enhanced_recap += f" and lasted {parent_summary['duration_minutes']:.0f} minutes"
        
        enhanced_recap += f" with {parent_summary['total_exchanges']} exchanges."
        
        return enhanced_recap, memory_ids
    
    def search_conversations_and_memories(self, query: str, 
                                        lookback_days: int = 30) -> Dict[str, any]:
        """
        Search both conversations and memories for relevant content
        
        Args:
            query: Search query
            lookback_days: How far back to search
            
        Returns:
            Combined search results
        """
        results = {
            'query': query,
            'conversations': [],
            'memories': [],
            'threads': []
        }
        
        # Extract potential topics from query
        query_words = query.lower().split()
        
        # Search conversations by topic
        for word in query_words:
            sessions = self.conversation_system.find_recent_sessions_by_topic(
                word, days_back=lookback_days
            )
            for session in sessions:
                if session.session_id not in [s['session_id'] for s in results['conversations']]:
                    results['conversations'].append({
                        'session_id': session.session_id,
                        'title': session.title,
                        'start_time': session.start_time,
                        'topics': list(session.topics),
                        'exchanges': session.total_exchanges
                    })
        
        # Search memories using the memory network
        if hasattr(self.memory_network.base_memory, 'search'):
            memory_results = self.memory_network.base_memory.search(query, n_results=10)
            results['memories'] = memory_results
        
        # Find similar problem patterns
        similar_patterns = self.memory_network.find_similar_patterns(query, {})
        for thread_id, score in similar_patterns:
            thread_summary = self.memory_network.get_thread_summary(
                self.memory_network.threads[thread_id].primary_topic
            )
            if thread_summary:
                results['threads'].append({
                    'thread_id': thread_id,
                    'similarity_score': score,
                    'summary': thread_summary
                })
        
        return results
    
    def get_current_context(self, include_conversation: bool = True,
                          include_memories: bool = True) -> Dict[str, any]:
        """
        Get comprehensive current context
        
        Args:
            include_conversation: Include recent conversation exchanges
            include_memories: Include relevant memories
            
        Returns:
            Current context dictionary
        """
        context = {
            'timestamp': datetime.now(),
            'session_id': self.conversation_system.current_session.session_id if self.conversation_system.current_session else None
        }
        
        if include_conversation and self.conversation_system.current_session:
            context['recent_exchanges'] = self.conversation_system.get_conversation_context(lookback_exchanges=3)
            context['session_topics'] = list(self.conversation_system.current_session.topics)
        
        if include_memories and context.get('session_topics'):
            # Get memories related to current topics
            relevant_memories = []
            for topic in context['session_topics'][:3]:  # Top 3 topics
                memories = list(self.memory_network.topic_to_memories.get(topic, []))[:5]
                relevant_memories.extend(memories)
            
            context['relevant_memory_ids'] = list(set(relevant_memories))
        
        return context
    
    def start_new_session(self, title: Optional[str] = None) -> str:
        """
        Explicitly start a new conversation session
        
        Args:
            title: Optional session title
            
        Returns:
            New session ID
        """
        return self.conversation_system.start_session(title=title)
    
    def end_current_session(self) -> Optional[str]:
        """
        End the current conversation session
        
        Returns:
            Ended session ID
        """
        return self.conversation_system.end_session()
    
    def get_analytics(self, days: int = 30) -> Dict[str, any]:
        """
        Get combined analytics from conversations and memories
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Comprehensive analytics
        """
        # Get conversation analytics
        conv_analytics = self.conversation_system.get_analytics(days=days)
        
        # Get memory network stats
        memory_stats = {
            'total_memories': len(self.memory_network.memory_graph.nodes),
            'total_connections': len(self.memory_network.memory_graph.edges),
            'total_threads': len(self.memory_network.threads),
            'topics_tracked': len(self.memory_network.topic_to_memories)
        }
        
        # Calculate memory health metrics
        if memory_stats['total_memories'] > 0:
            avg_connections = memory_stats['total_connections'] / memory_stats['total_memories']
            
            # Check for decaying memories
            decaying_count = 0
            for node_id in self.memory_network.memory_graph.nodes():
                node_data = self.memory_network.memory_graph.nodes[node_id]['data']
                decay_factor = self.memory_network._calculate_memory_decay(
                    node_data.timestamp,
                    node_data.access_count,
                    node_data.last_accessed
                )
                if decay_factor < 0.5:
                    decaying_count += 1
            
            memory_stats['avg_connections_per_memory'] = avg_connections
            memory_stats['decaying_memories'] = decaying_count
            memory_stats['memory_health'] = f"{((memory_stats['total_memories'] - decaying_count) / memory_stats['total_memories'] * 100):.1f}%"
        
        # Combine analytics
        combined_analytics = {
            'period_days': days,
            'conversations': conv_analytics,
            'memory_network': memory_stats,
            'integration_metrics': {
                'memories_per_conversation': (conv_analytics['total_memories'] / conv_analytics['total_sessions']) if conv_analytics['total_sessions'] > 0 else 0,
                'topics_per_conversation': len(conv_analytics['top_topics']) / conv_analytics['total_sessions'] if conv_analytics['total_sessions'] > 0 else 0
            }
        }
        
        return combined_analytics
    
    def _evaluate_memory_worthiness(self, query: str, response: str,
                                  generation_cost: float, topics: List[str]) -> bool:
        """
        Determine if an exchange should be stored in memory
        
        Args:
            query: User query
            response: ADAM's response
            generation_cost: Cost to generate response
            topics: Topics discussed
            
        Returns:
            True if memory-worthy
        """
        # High-cost responses are always worth storing
        if generation_cost > 0.01:
            return True
        
        # Error solutions are valuable
        if any(word in query.lower() for word in ['error', 'issue', 'problem', 'fail']):
            return True
        
        # Code implementations
        if any(marker in response for marker in ['```', 'def ', 'class ', 'function']):
            return True
        
        # Complex explanations (long responses)
        if len(response.split()) > 100:
            return True
        
        # Multiple topics indicate substantive discussion
        if len(topics) >= 3:
            return True
        
        # Simple exchanges aren't worth storing
        return False
    
    def _classify_memory_type(self, query: str, response: str) -> str:
        """
        Classify the type of memory
        
        Args:
            query: User query
            response: ADAM's response
            
        Returns:
            Memory type classification
        """
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Error solutions
        if any(word in query_lower for word in ['error', 'fail', 'issue', 'bug']):
            return "error_solution"
        
        # Code implementations
        if '```' in response or 'def ' in response:
            return "code_implementation"
        
        # How-to guides
        if query_lower.startswith(('how to', 'how do', 'how can')):
            return "how_to_guide"
        
        # Explanations
        if any(word in query_lower for word in ['what is', 'explain', 'why']):
            return "explanation"
        
        # Analysis
        if any(word in query_lower for word in ['analyze', 'review', 'evaluate']):
            return "analysis"
        
        return "general"
    
    def _format_time_ago(self, timestamp: datetime) -> str:
        """
        Format a timestamp as human-readable time ago
        
        Args:
            timestamp: Time to format
            
        Returns:
            Human-readable string like "2 hours ago" or "yesterday"
        """
        delta = datetime.now() - timestamp
        
        if delta.days > 1:
            return f"{delta.days} days ago"
        elif delta.days == 1:
            return "yesterday"
        elif delta.seconds > 3600:
            hours = delta.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif delta.seconds > 60:
            minutes = delta.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "just now"