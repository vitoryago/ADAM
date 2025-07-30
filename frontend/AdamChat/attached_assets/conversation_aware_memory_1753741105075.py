#!/usr/bin/env python3
"""
Conversation-Aware Memory System - Bridging conversations with memories

This integrates the conversation system with the memory network for seamless operation.

ARCHITECTURE OVERVIEW:
====================
This system acts as the orchestrator between two key components:
1. **ConversationSystem**: Tracks all exchanges and sessions
2. **MemoryNetworkSystem**: Stores important information with connections

KEY RESPONSIBILITIES:
- Decide which exchanges are "memory-worthy"
- Link memories to their conversation context
- Enable conversation continuation with memory context
- Provide unified search across conversations and memories

DESIGN PHILOSOPHY:
Not every exchange needs to be a memory. This system implements
intelligent filtering to store only valuable information while
maintaining complete conversation logs.

INTEGRATION FLOW:
1. User interacts with ADAM
2. This system evaluates the exchange
3. If valuable, stores in memory network
4. Always records in conversation system
5. Links memory and conversation for future reference
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
    
    WHY THIS ARCHITECTURE:
    - Separation of concerns: Conversations track everything, memories store value
    - Bidirectional linking: Can go from memory->conversation or conversation->memory
    - Unified interface: Single point of integration for ADAM components
    
    CORE CONCEPTS:
    - **Memory Worthiness**: Not all exchanges deserve long-term storage
    - **Context Preservation**: Every memory knows its conversational origin
    - **Intelligent Continuation**: Resume discussions with relevant memories loaded
    
    TYPICAL WORKFLOW:
    ```python
    # Initialize system
    cam = ConversationAwareMemorySystem(base_memory)
    
    # Process user interaction
    exchange_id, memory_id = cam.process_interaction(
        query="How do I optimize this SQL query?",
        response="Use indexes on join columns...",
        topics=["SQL", "optimization"],
        generation_cost=0.02,
        model_used="gpt-4"
    )
    
    # Later, continue the conversation
    recap, memories = cam.continue_conversation("SQL")
    ```
    """
    
    def __init__(self, base_memory_system, storage_path: str = "./adam_memory_advanced"):
        """
        Initialize the conversation-aware memory system
        
        Args:
            base_memory_system: The original ADAMMemoryAdvanced system
            storage_path: Base path for all storage
            
        INITIALIZATION:
        1. Create conversation system with dedicated storage
        2. Wrap memory system with conversation awareness
        3. Initialize shared context tracking
        
        STORAGE STRUCTURE:
        ```
        adam_memory_advanced/
        ├── conversations/      # All conversation sessions
        │   ├── session_*.json
        ├── memories/          # Memory network data
        │   ├── memory_*.json
        └── indexes/           # Search indexes
        ```
        
        The separation of storage allows independent backup/recovery
        of conversations vs memories.
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
                          model_used: Optional[str] = None,
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
            
        DECISION FLOW:
        1. Update context with any new information
        2. Evaluate if exchange is memory-worthy based on:
           - Generation cost (expensive = valuable)
           - Content type (errors, code, explanations)
           - Response length and complexity
        3. If worthy, store in memory network with references
        4. Always record in conversation system
        5. Return IDs for reference
        
        MEMORY WORTHINESS CRITERIA:
        - High cost (>$0.01) - expensive to regenerate
        - Error solutions - valuable for debugging
        - Code implementations - reusable artifacts
        - Long explanations - knowledge transfer
        - Multi-topic discussions - complex understanding
        
        EXAMPLE:
        ```python
        # Simple greeting - not memory worthy
        exc_id, mem_id = process_interaction(
            "Hello", "Hi! How can I help?",
            ["greeting"], 0.001, "mistral"
        )
        # mem_id will be None
        
        # Complex solution - memory worthy
        exc_id, mem_id = process_interaction(
            "Fix this SQL error", "[detailed solution]",
            ["SQL", "error", "database"], 0.025, "gpt-4"
        )
        # mem_id will have value
        ```
        """
        # Update current context
        if context:
            self.current_context.update(context)
        
        # Determine if this should be stored in memory
        # This is a critical decision point that balances storage costs
        # with information value. Too liberal = bloated memory, 
        # too conservative = lost knowledge
        memory_worthy = self._evaluate_memory_worthiness(
            query, response, generation_cost, topics
        )
        
        memory_id = None
        if memory_worthy:
            # Store in memory network with conversation context
            # The memory network will:
            # 1. Create memory node
            # 2. Find related memories
            # 3. Build connections
            # 4. Update topic threads
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
            
        CONTINUATION STRATEGY:
        1. Find previous conversations about the topic
        2. Create new session linked to most recent
        3. Get memory thread for the topic
        4. Generate human-friendly recap
        5. Return memory IDs for context loading
        
        RECAP GENERATION:
        The recap combines:
        - Memory network summary (what we discussed)
        - Conversation timing (when we discussed it)
        - Session metrics (how long, how many exchanges)
        
        This gives users a natural continuation experience:
        "We discussed SQL optimization 2 days ago for 45 minutes.
        You were working on query performance issues..."
        
        USAGE:
        ```python
        # User: "Let's continue our Python debugging session"
        recap, memory_ids = continue_conversation("python")
        
        # Load memories into context
        for mem_id in memory_ids:
            memory = load_memory(mem_id)
            context.add(memory)
        
        # Show recap to user
        print(recap)
        ```
        """
        # Get conversation continuity
        parent_session_id, _ = self.conversation_system.continue_conversation(topic)
        
        if not parent_session_id:
            return f"Starting a new conversation about {topic}.", []
        
        # Get memory context from the thread
        recap, memory_ids = self.memory_network.continue_thread(topic)
        
        # Add conversation context to recap
        # This enriches the memory-based recap with conversational metadata
        # making the continuation feel more natural and contextual
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
            
        SEARCH STRATEGY:
        1. **Topic-based conversation search**: Fast, uses indexes
        2. **Memory content search**: Deeper, uses embeddings
        3. **Pattern matching**: Finds similar problem/solution pairs
        
        RESULT STRUCTURE:
        ```python
        {
            'conversations': [  # Recent relevant sessions
                {
                    'session_id': '...',
                    'title': 'Debugging SQL Performance',
                    'topics': ['SQL', 'performance'],
                    'exchanges': 12
                }
            ],
            'memories': [  # Specific valuable exchanges
                {
                    'id': 'mem_123',
                    'query': '...',
                    'response': '...',
                    'score': 0.92
                }
            ],
            'threads': [  # Problem-solution patterns
                {
                    'thread_id': '...',
                    'similarity_score': 0.85,
                    'summary': 'Previous SQL optimization solutions'
                }
            ]
        }
        ```
        
        USE CASES:
        - "What did we discuss about Docker last week?"
        - "Find all conversations about error handling"
        - "Show me similar problems we've solved"
        """
        results = {
            'query': query,
            'conversations': [],
            'memories': [],
            'threads': []
        }
        
        # Extract potential topics from query
        # Simple word-based extraction, could be enhanced with NLP
        query_words = query.lower().split()
        
        # Search conversations by topic
        # We search each word as a potential topic to maximize recall
        # Deduplication ensures each session appears only once
        for word in query_words:
            sessions = self.conversation_system.find_recent_sessions_by_topic(
                word, days_back=lookback_days
            )
            for session in sessions:
                # Avoid duplicate sessions in results
                if session.session_id not in [s['session_id'] for s in results['conversations']]:
                    results['conversations'].append({
                        'session_id': session.session_id,
                        'title': session.title,
                        'start_time': session.start_time,
                        'topics': list(session.topics),
                        'exchanges': session.total_exchanges
                    })
        
        # Search memories using the memory network
        # This uses semantic search (embeddings) for better recall
        # than simple keyword matching
        if hasattr(self.memory_network.base_memory, 'search'):
            memory_results = self.memory_network.base_memory.search(query, n_results=10)
            results['memories'] = memory_results
        
        # Find similar problem patterns
        # This identifies previous similar issues and their solutions
        # Useful for "I have a similar problem to last time" scenarios
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
            
        CONTEXT BUILDING:
        This method assembles a complete picture of the current state:
        1. Session information (what conversation we're in)
        2. Recent exchanges (what we just discussed)
        3. Current topics (what we're talking about)
        4. Relevant memories (related past knowledge)
        
        This context is used for:
        - Informing response generation
        - Maintaining conversation coherence
        - Providing relevant examples
        - Avoiding repetition
        
        EXAMPLE OUTPUT:
        ```python
        {
            'timestamp': datetime.now(),
            'session_id': 'session_20240115_100000_abc',
            'recent_exchanges': [
                {'query': '...', 'response': '...', 'topics': [...]}
            ],
            'session_topics': ['python', 'debugging', 'async'],
            'relevant_memory_ids': ['mem_123', 'mem_456']
        }
        ```
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
            # Limited to top 3 topics and 5 memories per topic to avoid
            # overwhelming the context window while maintaining relevance
            relevant_memories = []
            for topic in context['session_topics'][:3]:  # Top 3 topics
                memories = list(self.memory_network.topic_to_memories.get(topic, []))[:5]
                relevant_memories.extend(memories)
            
            # Deduplicate memory IDs
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
            
        ANALYTICS INSIGHTS:
        This provides a holistic view of ADAM's interactions:
        
        1. **Conversation Metrics**: 
           - Session frequency and duration
           - Topic distribution
           - Engagement patterns
           
        2. **Memory Network Health**:
           - Total memories and connections
           - Memory decay (forgotten information)
           - Topic coverage
           
        3. **Integration Metrics**:
           - Memories per conversation (value density)
           - Topics per conversation (breadth)
           - Memory health (active vs decaying)
        
        USE CASES:
        - Monitor ADAM's effectiveness
        - Identify knowledge gaps
        - Optimize memory retention policies
        - Understand user interaction patterns
        
        EXAMPLE OUTPUT:
        ```python
        {
            'conversations': {
                'total_sessions': 45,
                'avg_duration_minutes': 28.3,
                'memory_storage_rate': '38.5%'
            },
            'memory_network': {
                'total_memories': 1523,
                'avg_connections_per_memory': 3.2,
                'memory_health': '87.3%'
            },
            'integration_metrics': {
                'memories_per_conversation': 4.7,
                'topics_per_conversation': 3.2
            }
        }
        ```
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
            # Memories decay based on age and access patterns
            # This identifies memories at risk of being forgotten
            decaying_count = 0
            for node_id in self.memory_network.memory_graph.nodes():
                node_data = self.memory_network.memory_graph.nodes[node_id]['data']
                decay_factor = self.memory_network._calculate_memory_decay(
                    node_data.timestamp,
                    node_data.access_count,
                    node_data.last_accessed
                )
                # Decay factor < 0.5 means memory is fading
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
            
        EVALUATION CRITERIA:
        This method implements a multi-factor decision algorithm:
        
        1. **Cost-based**: High generation cost = expensive to recreate
           Threshold: $0.01 (about 500 tokens with GPT-4)
           
        2. **Content-based**: Certain content types are inherently valuable
           - Error solutions (helps avoid repeated debugging)
           - Code implementations (reusable artifacts)
           - Complex explanations (knowledge transfer)
           
        3. **Complexity-based**: Information density indicators
           - Long responses (>100 words)
           - Multi-topic discussions (>=3 topics)
           
        BALANCING ACT:
        Too strict: Miss valuable information
        Too lenient: Memory bloat and noise
        
        Current thresholds are tuned for a good balance but
        can be adjusted based on storage constraints and use patterns.
        
        EXAMPLES:
        ```python
        # Memory-worthy:
        - "How do I fix this error?" -> Detailed debugging steps
        - "Explain async/await" -> Long educational response
        - "Write a function to..." -> Code implementation
        
        # Not memory-worthy:
        - "What time is it?" -> Simple factual response
        - "Thanks!" -> Social pleasantry
        - "Continue" -> Navigation command
        ```
        """
        # High-cost responses are always worth storing
        # This threshold represents roughly 500 tokens with GPT-4
        # or 2000 tokens with GPT-3.5-turbo
        if generation_cost > 0.01:
            return True
        
        # Error solutions are valuable
        # These help avoid repeated debugging of similar issues
        if any(word in query.lower() for word in ['error', 'issue', 'problem', 'fail']):
            return True
        
        # Code implementations
        # Code blocks and function definitions are reusable artifacts
        if any(marker in response for marker in ['```', 'def ', 'class ', 'function']):
            return True
        
        # Complex explanations (long responses)
        # Detailed explanations represent significant knowledge transfer
        if len(response.split()) > 100:
            return True
        
        # Multiple topics indicate substantive discussion
        # Multi-faceted conversations often contain valuable insights
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
            
        CLASSIFICATION CATEGORIES:
        - **error_solution**: Debugging help and fixes
        - **code_implementation**: Actual code artifacts
        - **how_to_guide**: Step-by-step instructions
        - **explanation**: Conceptual understanding
        - **analysis**: Code review, architecture discussion
        - **general**: Everything else
        
        WHY CLASSIFY:
        1. Enables type-specific retrieval ("show me all error solutions")
        2. Helps with memory organization and clustering
        3. Can apply different retention policies by type
        4. Improves search relevance by type matching
        
        CLASSIFICATION LOGIC:
        Uses simple keyword and pattern matching. Could be enhanced
        with ML classification for better accuracy.
        
        The order matters - we check from most specific to least specific
        to ensure accurate classification.
        """
        query_lower = query.lower()
        
        # Error solutions - highest priority classification
        # These are often the most immediately valuable memories
        if any(word in query_lower for word in ['error', 'fail', 'issue', 'bug']):
            return "error_solution"
        
        # Code implementations - concrete artifacts
        # Marked by code blocks or function definitions
        if '```' in response or 'def ' in response:
            return "code_implementation"
        
        # How-to guides - instructional content
        # These queries explicitly ask for procedures
        if query_lower.startswith(('how to', 'how do', 'how can')):
            return "how_to_guide"
        
        # Explanations - conceptual understanding
        # These help build mental models
        if any(word in query_lower for word in ['what is', 'explain', 'why']):
            return "explanation"
        
        # Analysis - evaluative content
        # Code reviews, architecture discussions, etc.
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
            
        FORMATTING RULES:
        - < 1 minute: "just now"
        - < 1 hour: "X minutes ago"
        - < 24 hours: "X hours ago"
        - 1 day: "yesterday"
        - > 1 day: "X days ago"
        
        This provides natural language timing that helps users
        orient themselves in their conversation history.
        
        EXAMPLE:
        ```python
        now = datetime.now()
        past = now - timedelta(hours=3, minutes=30)
        print(_format_time_ago(past))  # "3 hours ago"
        ```
        """
        delta = datetime.now() - timestamp
        
        if delta.days > 1:
            return f"{delta.days} days ago"
        elif delta.days == 1:
            return "yesterday"
        elif delta.seconds > 3600:
            hours = delta.seconds // 3600
            # Proper pluralization for natural language
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif delta.seconds > 60:
            minutes = delta.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "just now"