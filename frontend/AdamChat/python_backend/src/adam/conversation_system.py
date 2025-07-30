#!/usr/bin/env python3
"""
Conversation System - Managing ADAM's dialogue sessions and context

This system tracks conversations across time, enabling continuity and context preservation.

ARCHITECTURE OVERVIEW:
====================
The conversation system is designed with three key concepts:

1. **ConversationExchange**: The atomic unit - a single query/response pair
2. **ConversationSession**: A collection of exchanges forming a coherent dialogue
3. **ConversationSystem**: The orchestrator managing all sessions and their relationships

KEY FEATURES:
- Session continuity: Link related conversations across time
- Topic indexing: Fast retrieval of conversations by subject matter
- Persistence: All conversations are saved for future reference
- Analytics: Understand conversation patterns and user engagement

DESIGN DECISIONS:
- Sessions auto-save periodically to prevent data loss
- Topics are extracted and indexed for efficient search
- Parent-child session relationships enable conversation threading
- Exchanges track whether they were stored in the memory system

INTEGRATION POINTS:
- Works with the memory network to store important exchanges
- Provides context for the conversation-aware memory system
- Can be queried by other ADAM components for historical context
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import uuid
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConversationExchange:
    """
    Represents a single exchange (query + response) within a conversation
    
    This is the atomic unit of conversation - one back-and-forth between
    user and ADAM. Multiple exchanges form a complete conversation session.
    
    WHY THIS DESIGN:
    - Exchanges are immutable once created (dataclass provides this)
    - Each exchange has a unique ID for reference
    - We track whether each exchange was "memory-worthy" to understand
      what kinds of interactions get preserved long-term
    - Context field allows storing environmental information (screen content,
      active files, etc.) that might be relevant for understanding the exchange
    
    EXAMPLE:
    exchange = ConversationExchange(
        exchange_id="exc_123",
        query="How do I optimize my SQL query?",
        response="Here are 3 ways to optimize...",
        timestamp=datetime.now(),
        topics=["SQL", "optimization"],
        stored_in_memory=True,
        memory_id="mem_456",
        context={"active_file": "query.sql", "error_present": True}
    )
    """
    exchange_id: str
    query: str
    response: str
    timestamp: datetime
    topics: List[str]
    # Was this exchange worthy of memory storage?
    stored_in_memory: bool = False
    # If stored, what was the memory ID?
    memory_id: Optional[str] = None
    # Context at time of exchange (screen content, etc.)
    context: Dict[str, any] = field(default_factory=dict)


@dataclass
class ConversationSession:
    """
    Represents a complete conversation session from start to end
    
    A session is like a chapter in ADAM's interaction history. It has:
    - Clear boundaries (start/end time)
    - A coherent flow of topics  
    - Multiple exchanges that build upon each other
    
    WHY THIS DESIGN:
    - Sessions can be in different states (active/paused/completed) to handle
      interruptions gracefully
    - Parent-child relationships enable conversation threading ("continue where
      we left off" functionality)
    - Topics are stored as a Set for efficient deduplication and search
    - We track both total exchanges and memories created to understand the
      "value density" of conversations
    
    SESSION LIFECYCLE:
    1. Created when user starts interacting (state="active")
    2. Accumulates exchanges as conversation progresses
    3. Can be paused (user steps away) and resumed later
    4. Completed when user explicitly ends or starts new session
    
    EXAMPLE:
    session = ConversationSession(
        session_id="session_20240115_143022_abc123",
        start_time=datetime.now(),
        title="Debugging Python async issues",
        state="active"
    )
    """
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    # All exchanges in chronological order
    exchanges: List[ConversationExchange] = field(default_factory=list)
    # Primary topics discussed (extracted from exchanges)
    topics: Set[str] = field(default_factory=set)
    # Session metadata
    total_exchanges: int = 0
    memories_created: int = 0
    # Session state: active, paused, completed
    state: str = "active"
    # Optional session title (e.g., "Debugging dbt timeout issues")
    title: Optional[str] = None
    # Link to previous related session
    parent_session_id: Optional[str] = None


class ConversationSystem:
    """
    Manages all conversation sessions and provides continuity across interactions
    
    This is ADAM's "social memory" - it knows not just what was said, but
    when it was said, in what context, and how conversations relate to each other.
    
    CORE RESPONSIBILITIES:
    1. Session Management: Create, pause, resume, and end conversations
    2. Exchange Recording: Capture every interaction with full context
    3. Topic Indexing: Enable fast retrieval by subject matter
    4. Persistence: Save/load all conversation data
    5. Analytics: Provide insights into conversation patterns
    
    DATA STRUCTURES:
    - self.current_session: The active conversation (only one at a time)
    - self.sessions: All sessions indexed by ID for fast lookup
    - self.topic_to_sessions: Inverted index for topic-based search
    - self.session_chains: Track parent-child relationships
    
    WHY THIS ARCHITECTURE:
    - Single active session prevents context confusion
    - Topic indexing enables "what did we discuss about X?" queries
    - Session chains support multi-day problem-solving workflows
    - Auto-save prevents data loss during long conversations
    
    INTEGRATION EXAMPLE:
    ```python
    conv_system = ConversationSystem()
    
    # Start discussing a problem
    session_id = conv_system.start_session("Debug: Database timeouts")
    
    # Record the exchange
    conv_system.record_exchange(
        query="My queries timeout after 30s",
        response="Let's check your connection pooling...",
        topics=["database", "timeout", "debugging"]
    )
    
    # Later, continue the conversation
    parent_id, new_id = conv_system.continue_conversation("database")
    ```
    """
    
    def __init__(self, storage_path: str = "./adam_memory_advanced/conversations"):
        """
        Initialize the conversation system
        
        Args:
            storage_path: Where to persist conversation data
            
        INITIALIZATION FLOW:
        1. Create storage directory if it doesn't exist
        2. Initialize empty data structures
        3. Load any existing sessions from disk
        4. Auto-start a session if none exists (user-friendly)
        
        WHY AUTO-START:
        - Users can immediately start talking without setup
        - Prevents null pointer issues in integration code
        - Mimics natural conversation flow (no "start conversation" in real life)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Current active session
        self.current_session: Optional[ConversationSession] = None
        
        # All sessions indexed by ID
        self.sessions: Dict[str, ConversationSession] = {}
        
        # Topic index for fast topic-based search
        self.topic_to_sessions: Dict[str, List[str]] = defaultdict(list)
        
        # Session continuity tracking
        self.session_chains: Dict[str, List[str]] = defaultdict(list)
        
        # Load existing sessions from disk
        # This restores the complete conversation history and rebuilds
        # all indexes for fast search
        self._load_sessions()
        
        # Auto-start a session if none exists
        # This ensures ADAM is always ready to converse without requiring
        # explicit session management from the user
        if not self.current_session:
            self.start_session()
    
    def start_session(self, title: Optional[str] = None, 
                     parent_session_id: Optional[str] = None) -> str:
        """
        Start a new conversation session
        
        Args:
            title: Optional descriptive title for the session
            parent_session_id: ID of related previous session (for continuity)
            
        Returns:
            The new session ID
            
        BEHAVIOR:
        - If a session is already active, it's automatically ended first
        - Session IDs include timestamp for easy chronological sorting
        - Parent-child relationships create conversation threads
        
        SESSION ID FORMAT:
        "session_YYYYMMDD_HHMMSS_[8-char-hex]"
        Example: "session_20240115_143022_abc12345"
        
        This format ensures:
        - Sessions sort chronologically by ID
        - IDs are globally unique (timestamp + random)
        - Human-readable timestamp for debugging
        
        USAGE PATTERNS:
        1. Fresh start: start_session("Working on SQL optimization")
        2. Continuation: start_session("Continuing SQL work", parent_id)
        3. Auto-titled: start_session() # Title generated from topics
        """
        # End current session if one is active
        if self.current_session and self.current_session.state == "active":
            self.end_session()
        
        # Generate unique session ID with timestamp for easy sorting
        # Format: session_YYYYMMDD_HHMMSS_randomhex
        # This ensures chronological ordering when listing sessions
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create new session
        self.current_session = ConversationSession(
            session_id=session_id,
            start_time=datetime.now(),
            title=title,
            parent_session_id=parent_session_id,
            state="active"
        )
        
        # Store in sessions dict
        self.sessions[session_id] = self.current_session
        
        # Track session chains for continuity
        # This creates a tree structure of related conversations
        # enabling "show me all follow-up discussions" queries
        if parent_session_id:
            self.session_chains[parent_session_id].append(session_id)
        
        logger.info(f"Started new conversation session: {session_id}")
        
        # Auto-save on session start
        # This ensures even empty sessions are persisted, preventing
        # edge cases where a session is created but no exchanges recorded
        self._save_current_session()
        
        return session_id
    
    def record_exchange(self, query: str, response: str, 
                       topics: List[str], context: Optional[Dict] = None,
                       stored_in_memory: bool = False, 
                       memory_id: Optional[str] = None) -> str:
        """
        Record a conversation exchange in the current session
        
        This is called after every user-ADAM interaction to maintain
        a complete record of the conversation.
        
        Args:
            query: The user's question/input
            response: ADAM's response  
            topics: Topics discussed in this exchange
            context: Optional context (screen content, etc.)
            stored_in_memory: Whether this was stored in memory system
            memory_id: ID if stored in memory
            
        Returns:
            Exchange ID
            
        INTEGRATION FLOW:
        1. Conversation-aware memory decides if exchange is memory-worthy
        2. If yes, stores in memory network and gets memory_id
        3. Records exchange here with memory linkage
        4. Updates all indexes for fast retrieval
        
        AUTO-SAVE STRATEGY:
        - Saves every 5 exchanges to balance performance and data safety
        - Also saves on session end, pause, or system shutdown
        
        WHY TRACK MEMORY STORAGE:
        - Helps understand which interactions are most valuable
        - Enables "show me memorable conversations" queries
        - Provides metrics on information density
        
        EXAMPLE:
        ```python
        exchange_id = record_exchange(
            query="My app crashes with AttributeError",
            response="That error means... Here's how to fix it...",
            topics=["python", "debugging", "AttributeError"],
            context={"error_trace": "...", "file": "app.py"},
            stored_in_memory=True,
            memory_id="mem_python_error_001"
        )
        ```
        """
        if not self.current_session:
            self.start_session()
        
        # Generate exchange ID
        exchange_id = f"exchange_{uuid.uuid4().hex[:12]}"
        
        # Create exchange record
        exchange = ConversationExchange(
            exchange_id=exchange_id,
            query=query,
            response=response,
            timestamp=datetime.now(),
            topics=topics,
            stored_in_memory=stored_in_memory,
            memory_id=memory_id,
            context=context or {}
        )
        
        # Add to current session
        self.current_session.exchanges.append(exchange)
        self.current_session.total_exchanges += 1
        
        # Update session topics
        self.current_session.topics.update(topics)
        
        # Update topic index for fast topic-based search
        # This inverted index enables queries like "find all conversations about SQL"
        # We store lowercase topics for case-insensitive search
        for topic in topics:
            if self.current_session.session_id not in self.topic_to_sessions[topic.lower()]:
                self.topic_to_sessions[topic.lower()].append(self.current_session.session_id)
        
        # Track memories created
        if stored_in_memory:
            self.current_session.memories_created += 1
        
        # Auto-save periodically (every 5 exchanges)
        # This prevents data loss during long conversations while avoiding
        # excessive disk I/O. The number 5 is chosen as a balance between
        # safety (don't lose much) and performance (don't save too often)
        if self.current_session.total_exchanges % 5 == 0:
            self._save_current_session()
        
        return exchange_id
    
    def end_session(self) -> Optional[str]:
        """
        End the current conversation session
        
        Returns:
            Session ID of the ended session
            
        BEHAVIOR:
        - Marks the session as completed
        - Auto-generates a title if none provided (based on topics)
        - Performs final save to disk
        - Clears current_session to prepare for next conversation
        
        AUTO-TITLING:
        If no title was set, generates one from the top 3 topics.
        Example: "Discussion about SQL, optimization, indexes"
        
        This helps users identify sessions in history without
        requiring manual naming.
        """
        if not self.current_session:
            return None
        
        # Mark end time
        self.current_session.end_time = datetime.now()
        self.current_session.state = "completed"
        
        # Generate session title if not provided
        # This creates human-readable titles for sessions that weren't
        # explicitly named, making conversation history more navigable
        if not self.current_session.title and self.current_session.topics:
            # Use most common topics for title (up to 3 for readability)
            self.current_session.title = f"Discussion about {', '.join(list(self.current_session.topics)[:3])}"
        
        # Final save
        self._save_current_session()
        
        session_id = self.current_session.session_id
        logger.info(f"Ended conversation session: {session_id}")
        
        # Clear current session
        self.current_session = None
        
        return session_id
    
    def pause_session(self) -> Optional[str]:
        """
        Pause the current session (can be resumed later)
        
        Useful for temporary interruptions without losing context
        
        Returns:
            Session ID of the paused session
            
        USE CASES:
        - User needs to step away but wants to continue later
        - Switching context temporarily (e.g., urgent bug fix)
        - End of workday with intention to resume tomorrow
        
        BEHAVIOR:
        - Session state changes to "paused"
        - Session is saved immediately
        - Can be resumed with resume_session()
        - Auto-resume logic in _load_sessions() for recent pauses
        
        DESIGN RATIONALE:
        Pause/resume is separate from end/start to preserve the
        semantic connection of exchanges within a logical conversation.
        """
        if not self.current_session:
            return None
        
        self.current_session.state = "paused"
        self._save_current_session()
        
        session_id = self.current_session.session_id
        logger.info(f"Paused conversation session: {session_id}")
        
        return session_id
    
    def resume_session(self, session_id: str) -> bool:
        """
        Resume a paused session
        
        Args:
            session_id: ID of session to resume
            
        Returns:
            True if resumed successfully
            
        BEHAVIOR:
        - Only paused sessions can be resumed (not completed ones)
        - If another session is active, it's automatically paused first
        - Session state changes back to "active"
        - All context (topics, exchanges, etc.) is preserved
        
        ERROR CASES:
        - Session doesn't exist: returns False
        - Session is completed: returns False (use continue_conversation instead)
        - Session is already active: returns True (no-op)
        
        USAGE:
        ```python
        # User returns after lunch
        if conv_system.resume_session("session_20240115_120000_abc123"):
            print("Welcome back! Let's continue...")
        else:
            print("Couldn't resume. Starting fresh.")
        ```
        """
        if session_id not in self.sessions:
            logger.warning(f"Session not found: {session_id}")
            return False
        
        session = self.sessions[session_id]
        
        if session.state != "paused":
            logger.warning(f"Session {session_id} is not paused (state: {session.state})")
            return False
        
        # End current session if active
        if self.current_session and self.current_session.state == "active":
            self.pause_session()
        
        # Resume the session
        self.current_session = session
        self.current_session.state = "active"
        
        logger.info(f"Resumed conversation session: {session_id}")
        
        return True
    
    def find_recent_sessions_by_topic(self, topic: str, 
                                     days_back: int = 7,
                                     limit: int = 5) -> List[ConversationSession]:
        """
        Find recent sessions about a specific topic
        
        This enables "let's continue our SQL discussion from yesterday"
        
        Args:
            topic: Topic to search for
            days_back: How many days to look back
            limit: Maximum number of sessions to return
            
        Returns:
            List of matching sessions, newest first
            
        ALGORITHM:
        1. Use topic index for O(1) lookup of relevant session IDs
        2. Filter by date range to get recent sessions
        3. Sort by start time (newest first)
        4. Return top N sessions
        
        CASE SENSITIVITY:
        Topics are matched case-insensitively to improve recall.
        "SQL", "sql", and "Sql" all match the same sessions.
        
        USE CASES:
        - "Continue our debugging session from yesterday"
        - "What did we discuss about Python last week?"
        - "Show me recent conversations about database optimization"
        
        EXAMPLE:
        ```python
        # Find recent Python discussions
        python_sessions = find_recent_sessions_by_topic(
            "python", 
            days_back=14,  # Last 2 weeks
            limit=10       # Top 10 sessions
        )
        for session in python_sessions:
            print(f"{session.title} - {session.start_time}")
        ```
        """
        # Get all sessions for this topic
        session_ids = self.topic_to_sessions.get(topic.lower(), [])
        
        if not session_ids:
            return []
        
        # Filter by recency
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_sessions = []
        
        for session_id in session_ids:
            session = self.sessions.get(session_id)
            if session and session.start_time >= cutoff_date:
                recent_sessions.append(session)
        
        # Sort by start time (newest first)
        recent_sessions.sort(key=lambda s: s.start_time, reverse=True)
        
        return recent_sessions[:limit]
    
    def get_session_summary(self, session_id: str) -> Dict[str, any]:
        """
        Get a comprehensive summary of a conversation session
        
        Args:
            session_id: Session to summarize
            
        Returns:
            Dictionary with session analysis
            
        SUMMARY INCLUDES:
        - Basic info: ID, title, state, timestamps
        - Duration in minutes (if completed)
        - Exchange count and memory storage rate
        - Topic frequency analysis
        - Conversation threading (parent/child sessions)
        
        ANALYTICS:
        The summary provides insights into:
        - Conversation value (memory storage rate)
        - Topic focus (what was discussed most)
        - Conversation flow (parent/child relationships)
        
        USE CASES:
        - Display session history to user
        - Analyze conversation patterns
        - Generate conversation reports
        
        EXAMPLE OUTPUT:
        ```python
        {
            'session_id': 'session_20240115_100000_abc',
            'title': 'Debugging SQL Performance',
            'state': 'completed',
            'duration_minutes': 45.5,
            'total_exchanges': 12,
            'memory_storage_rate': '41.7%',
            'top_topics': [('SQL', 8), ('performance', 6)],
            'parent_session': 'session_20240114_150000_xyz',
            'child_sessions': ['session_20240116_093000_def']
        }
        ```
        """
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        # Calculate duration
        duration = None
        if session.end_time:
            duration = (session.end_time - session.start_time).total_seconds() / 60  # minutes
        
        # Find most discussed topics
        topic_counts = defaultdict(int)
        for exchange in session.exchanges:
            for topic in exchange.topics:
                topic_counts[topic] += 1
        
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Memory storage rate - a key metric for conversation value
        # High rate (>50%) suggests information-dense conversation
        # Low rate (<20%) might be casual chat or simple queries
        memory_rate = (session.memories_created / session.total_exchanges * 100 
                      if session.total_exchanges > 0 else 0)
        
        summary = {
            'session_id': session_id,
            'title': session.title or 'Untitled Session',
            'state': session.state,
            'start_time': session.start_time,
            'end_time': session.end_time,
            'duration_minutes': duration,
            'total_exchanges': session.total_exchanges,
            'memories_created': session.memories_created,
            'memory_storage_rate': f"{memory_rate:.1f}%",
            'topics': list(session.topics),
            'top_topics': top_topics,
            'parent_session': session.parent_session_id,
            'child_sessions': self.session_chains.get(session_id, [])
        }
        
        return summary
    
    def continue_conversation(self, topic: str) -> Tuple[Optional[str], str]:
        """
        Continue a previous conversation about a topic
        
        This is the main entry point for "let's continue where we left off"
        
        Args:
            topic: The topic to continue discussing
            
        Returns:
            Tuple of (parent_session_id, new_session_id)
            
        BEHAVIOR:
        1. Searches for recent sessions about the topic
        2. If found, creates new session linked to most recent
        3. If not found, creates fresh session
        
        CONTINUITY MODEL:
        Rather than resuming old sessions, we create new "child" sessions.
        This preserves the temporal boundary of each conversation while
        maintaining logical connections.
        
        WHY NOT JUST RESUME:
        - Each session represents a distinct time period
        - Metrics (duration, exchange rate) remain meaningful
        - User can see conversation evolution over time
        
        INTEGRATION:
        This method works with conversation_aware_memory to provide:
        - Session linkage (parent/child)
        - Memory thread continuation
        - Context preservation
        
        EXAMPLE:
        ```python
        # User: "Let's continue our discussion about the database issues"
        parent_id, new_id = continue_conversation("database")
        
        if parent_id:
            print(f"Continuing from {parent_id}")
            # Load relevant memories, show recap, etc.
        else:
            print("Starting fresh discussion about database")
        ```
        """
        # Find recent sessions about this topic
        recent_sessions = self.find_recent_sessions_by_topic(topic, days_back=30)
        
        if not recent_sessions:
            # No previous sessions, start fresh
            new_session_id = self.start_session(title=f"New discussion about {topic}")
            return (None, new_session_id)
        
        # Use the most recent session as parent
        parent_session = recent_sessions[0]
        
        # Start new session with continuity
        new_session_id = self.start_session(
            title=f"Continuing: {parent_session.title or topic}",
            parent_session_id=parent_session.session_id
        )
        
        logger.info(f"Continuing conversation from {parent_session.session_id}")
        
        return (parent_session.session_id, new_session_id)
    
    def get_conversation_context(self, lookback_exchanges: int = 5) -> List[Dict[str, str]]:
        """
        Get recent conversation context for the current session
        
        Useful for maintaining context in multi-turn conversations
        
        Args:
            lookback_exchanges: How many recent exchanges to return
            
        Returns:
            List of recent exchanges as dicts
            
        PURPOSE:
        Provides recent conversation history for:
        - LLM context windows ("based on our discussion...")
        - UI display ("conversation so far")
        - Context-aware responses
        
        FORMAT:
        Returns simplified exchange data suitable for serialization:
        ```python
        [
            {
                'query': 'How do I fix this error?',
                'response': 'Try checking your imports...',
                'timestamp': '2024-01-15T10:30:00',
                'topics': ['python', 'debugging']
            },
            ...
        ]
        ```
        
        USAGE IN PROMPTS:
        ```python
        context = get_conversation_context(3)
        prompt = "Previous context:\n"
        for exc in context:
            prompt += f"User: {exc['query']}\n"
            prompt += f"Assistant: {exc['response']}\n\n"
        prompt += f"User: {new_query}"
        ```
        """
        if not self.current_session:
            return []
        
        recent_exchanges = self.current_session.exchanges[-lookback_exchanges:]
        
        context = []
        for exchange in recent_exchanges:
            context.append({
                'query': exchange.query,
                'response': exchange.response,
                'timestamp': exchange.timestamp.isoformat(),
                'topics': exchange.topics
            })
        
        return context
    
    def _save_current_session(self):
        """
        Save the current session to disk
        
        PERSISTENCE STRATEGY:
        - Each session saved to individual JSON file
        - Filename matches session ID for easy lookup
        - JSON format for human readability and debugging
        
        FILE FORMAT:
        ```json
        {
            "session_id": "session_20240115_100000_abc",
            "start_time": "2024-01-15T10:00:00",
            "state": "active",
            "exchanges": [
                {
                    "query": "...",
                    "response": "...",
                    "timestamp": "...",
                    "topics": [...]
                }
            ]
        }
        ```
        
        ERROR HANDLING:
        - Creates directory if needed
        - Silently continues on write errors (logged)
        - Atomic writes would be better for production
        """
        if not self.current_session:
            return
        
        session_file = self.storage_path / f"{self.current_session.session_id}.json"
        
        # Convert session to serializable format
        session_data = {
            'session_id': self.current_session.session_id,
            'start_time': self.current_session.start_time.isoformat(),
            'end_time': self.current_session.end_time.isoformat() if self.current_session.end_time else None,
            'state': self.current_session.state,
            'title': self.current_session.title,
            'parent_session_id': self.current_session.parent_session_id,
            'topics': list(self.current_session.topics),
            'total_exchanges': self.current_session.total_exchanges,
            'memories_created': self.current_session.memories_created,
            'exchanges': []
        }
        
        # Serialize exchanges
        for exchange in self.current_session.exchanges:
            session_data['exchanges'].append({
                'exchange_id': exchange.exchange_id,
                'query': exchange.query,
                'response': exchange.response,
                'timestamp': exchange.timestamp.isoformat(),
                'topics': exchange.topics,
                'stored_in_memory': exchange.stored_in_memory,
                'memory_id': exchange.memory_id,
                'context': exchange.context
            })
        
        # Write to file
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def _load_sessions(self):
        """
        Load all sessions from disk
        
        LOADING PROCESS:
        1. Scan storage directory for session files
        2. Load and parse each JSON file
        3. Reconstruct session objects
        4. Rebuild all indexes (topic, chain)
        5. Identify and resume active/recent sessions
        
        AUTO-RESUME LOGIC:
        - Active sessions are resumed immediately
        - Recently paused sessions (< 24 hours) are resumed
        - Older paused sessions remain paused
        
        This provides a natural conversation flow where ADAM
        "remembers" what you were just discussing.
        
        ERROR RECOVERY:
        - Corrupted files are skipped (logged)
        - Missing fields use defaults
        - Partial loads don't break the system
        """
        # Load session files
        session_files = list(self.storage_path.glob("session_*.json"))
        
        for session_file in session_files:
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct session
                session = ConversationSession(
                    session_id=data['session_id'],
                    start_time=datetime.fromisoformat(data['start_time']),
                    end_time=datetime.fromisoformat(data['end_time']) if data['end_time'] else None,
                    state=data['state'],
                    title=data.get('title'),
                    parent_session_id=data.get('parent_session_id'),
                    topics=set(data.get('topics', [])),
                    total_exchanges=data.get('total_exchanges', 0),
                    memories_created=data.get('memories_created', 0)
                )
                
                # Reconstruct exchanges
                for exc_data in data.get('exchanges', []):
                    exchange = ConversationExchange(
                        exchange_id=exc_data['exchange_id'],
                        query=exc_data['query'],
                        response=exc_data['response'],
                        timestamp=datetime.fromisoformat(exc_data['timestamp']),
                        topics=exc_data['topics'],
                        stored_in_memory=exc_data.get('stored_in_memory', False),
                        memory_id=exc_data.get('memory_id'),
                        context=exc_data.get('context', {})
                    )
                    session.exchanges.append(exchange)
                
                # Store session
                self.sessions[session.session_id] = session
                
                # Rebuild indices
                # Topic index uses lowercase for case-insensitive search
                for topic in session.topics:
                    self.topic_to_sessions[topic.lower()].append(session.session_id)
                
                # Rebuild session chains
                if session.parent_session_id:
                    self.session_chains[session.parent_session_id].append(session.session_id)
                
            except Exception as e:
                logger.error(f"Error loading session {session_file}: {e}")
        
        logger.info(f"Loaded {len(self.sessions)} conversation sessions")
        
        # Find active or most recent session
        # This auto-resume behavior makes ADAM feel more continuous
        # and context-aware across restarts
        active_sessions = [s for s in self.sessions.values() if s.state == "active"]
        if active_sessions:
            # Resume the most recent active session
            # (handles edge case of multiple active sessions from crashes)
            self.current_session = max(active_sessions, key=lambda s: s.start_time)
            logger.info(f"Resumed active session: {self.current_session.session_id}")
        else:
            # Find most recent paused session
            paused_sessions = [s for s in self.sessions.values() if s.state == "paused"]
            if paused_sessions:
                recent_paused = max(paused_sessions, key=lambda s: s.start_time)
                # Only auto-resume if paused recently (within 24 hours)
                # This prevents resuming very old conversations that
                # the user has likely forgotten about
                if (datetime.now() - recent_paused.start_time).days < 1:
                    self.resume_session(recent_paused.session_id)
    
    def get_analytics(self, days: int = 30) -> Dict[str, any]:
        """
        Get conversation analytics for the specified period
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with analytics data
            
        ANALYTICS PROVIDED:
        - Total sessions, exchanges, and memories
        - Average exchanges per session (conversation depth)
        - Average session duration (engagement time)
        - Memory storage rate (information density)
        - Top topics (what users discuss most)
        - Sessions by hour (when users are active)
        - Conversation chains (multi-session discussions)
        
        USE CASES:
        - Understand user engagement patterns
        - Identify popular topics for improvement
        - Track ADAM's value delivery (memory rate)
        - Optimize availability (peak hours)
        
        EXAMPLE OUTPUT:
        ```python
        {
            'period_days': 30,
            'total_sessions': 45,
            'total_exchanges': 523,
            'avg_exchanges_per_session': 11.6,
            'avg_session_duration_minutes': 28.3,
            'memory_storage_rate': '38.5%',
            'top_topics': [('python', 23), ('sql', 18), ('debugging', 15)],
            'sessions_by_hour': {9: 8, 10: 12, 11: 10, ...},
            'conversation_chains': 12  # Multi-session discussions
        }
        ```
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter sessions within date range
        recent_sessions = [s for s in self.sessions.values() 
                          if s.start_time >= cutoff_date]
        
        if not recent_sessions:
            return {
                'period_days': days,
                'total_sessions': 0,
                'total_exchanges': 0,
                'total_memories': 0
            }
        
        # Calculate metrics
        total_exchanges = sum(s.total_exchanges for s in recent_sessions)
        total_memories = sum(s.memories_created for s in recent_sessions)
        total_duration = sum(
            ((s.end_time or datetime.now()) - s.start_time).total_seconds() / 60
            for s in recent_sessions
        )
        
        # Topic frequency analysis
        # This helps understand what users talk about most
        all_topics = defaultdict(int)
        for session in recent_sessions:
            for topic in session.topics:
                all_topics[topic] += 1
        
        top_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Session patterns - when do users interact with ADAM?
        # This data helps optimize availability and resource allocation
        sessions_by_hour = defaultdict(int)
        for session in recent_sessions:
            hour = session.start_time.hour
            sessions_by_hour[hour] += 1
        
        analytics = {
            'period_days': days,
            'total_sessions': len(recent_sessions),
            'total_exchanges': total_exchanges,
            'total_memories': total_memories,
            'avg_exchanges_per_session': total_exchanges / len(recent_sessions),
            'avg_session_duration_minutes': total_duration / len(recent_sessions),
            'memory_storage_rate': f"{(total_memories / total_exchanges * 100):.1f}%",
            'top_topics': top_topics,
            'sessions_by_hour': dict(sessions_by_hour),
            'conversation_chains': len([s for s in recent_sessions if s.parent_session_id])
        }
        
        return analytics