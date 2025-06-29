#!/usr/bin/env python3
"""
Memory Network System - Connecting related conversations and memories
This creates a graph of interconnected memories that ADAM can traverse intelligently

OVERVIEW:
The Memory Network System is ADAM's "neural network" - it doesn't just store memories,
it understands how they relate to each other. Think of it like a brain where:
- Memories are neurons
- References are synapses
- Topics are regions
- Threads are pathways

KEY CONCEPTS:
1. Memory Nodes: Individual memories with rich metadata and relationships
2. Reference Weights: Not all connections are equal - some are stronger than others
3. Conversation Threads: Track how understanding evolves over multiple sessions
4. Memory Decay: Old, unused memories fade like human memory
5. Pattern Recognition: Identifies recurring problem-solution patterns

ARCHITECTURE:
- Built on NetworkX for graph operations
- Integrates with base memory system for storage
- Links to conversation system for context
- Persists to disk for continuity across sessions

USAGE:
The system is typically used through the ConversationAwareMemorySystem,
but can be used directly for advanced memory operations like visualization
or manual thread management.
"""

# We need these dataclasses (decorators) to structure our memory objects cleanly
from dataclasses import dataclass, field
# Type hints make our code self-documenting and catch errors early
from typing import List, Dict, Set, Optional, Tuple, Any
# For timestamp tracking - knowing WHEN memories were created is crucial
from datetime import datetime
# defaultdict creates dictionaries that auto-initialize missing keys
from collections import defaultdict
# NetworkX is our graph library - it handles the complex relationships between memories
import networkx as nx
# For saving/loading our memory network to disk
import json
from pathlib import Path
# For logging instead of undefined console
import logging
# For embedding-based semantic similarity
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryNode:
    """
    Represents a single memory with its relationships
    
    Think of this like a neuron in ADAM's brain:
    - It holds information (the memory itself)
    - It has connections to other neurons (references)
    - Those connections have different strengths (weights)
    
    DESIGN PHILOSOPHY:
    A memory is more than just stored information - it's a living entity that:
    1. Knows its origin (conversation_id, timestamp)
    2. Understands its purpose (memory_type, topics)
    3. Tracks its relationships (references, referenced_by)
    4. Monitors its relevance (access_count, last_accessed)
    5. Can decay over time if not reinforced
    
    FIELDS EXPLAINED:
    - memory_id: Unique identifier, typically from the base memory system
    - conversation_id: Links this memory to its originating conversation
    - timestamp: When created - used for recency calculations and decay
    - query/response: The actual Q&A content
    - topics: Semantic tags for categorization and search
    - memory_type: Classification (error_solution, explanation, etc.)
    - embedding: Vector representation for semantic similarity
    - access_count: How often this memory is retrieved (prevents decay)
    - last_accessed: When last used (for decay calculations)
    - references: Memories this builds upon (outgoing edges)
    - referenced_by: Memories that build upon this (incoming edges)
    - reference_weights: Strength of each reference connection
    """
    # Core identity - every memory needs a unique ID
    memory_id: str
    # Which conversation created this memory? Important for context
    conversation_id: str
    # When was this memory created? Recent memories often more relevant
    timestamp: datetime
    # What was asked? The original question/problem
    query: str
    # What did ADAM answer? The solution/response
    response: str
    # What topics does this cover? Like tags or categories
    topics: List[str]
    # Type classification: error_solution, explanation, etc.
    memory_type: str
    # Embedding vector for semantic similarity
    embedding: Optional[np.ndarray] = None
    # Access count for memory decay calculations
    access_count: int = 0
    # Last accessed timestamp for decay
    last_accessed: Optional[datetime] = None

    # HERE'S WHERE THE MAGIC HAPPENS - RELATIONSHIPS!

    # References = "This memory builds upon these older memories"
    references: List[str] = field(default_factory=list)
    # Memories that reference this one (newer memories that built upon this)
    referenced_by: List[str] = field(default_factory=list)

    # Not all references are equal - some are strongly related, others tangentially
    # reference_weights maps memory_id -> strenght (0.0 to 1.0)
    reference_weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class ConversationThread:
    """
    Represents a thread of related conversations on a topic
    
    Imagine you're debugging a problem over several days - this thread
    tracks the entire journey, not just individual conversations.
    
    Example: "dbt debugging thread" might span 4 conversations over 3 days,
    containing 15 individual memories, showing how the problem evolved.
    
    CONCEPTUAL MODEL:
    A thread is like a "story arc" in ADAM's experience:
    - Chapter 1: "User encounters timeout error" (Monday)
    - Chapter 2: "We try optimizing CTEs" (Tuesday)  
    - Chapter 3: "Discover it's missing indexes" (Wednesday)
    - Chapter 4: "Solution implemented successfully" (Thursday)
    
    WHY THREADS MATTER:
    1. Continuity: Users don't solve problems in one conversation
    2. Evolution: Solutions evolve through trial and error
    3. Learning: Patterns emerge across multiple attempts
    4. Context: Later conversations build on earlier ones
    
    FIELDS EXPLAINED:
    - thread_id: Unique identifier with timestamp for sorting
    - primary_topic: Main subject (e.g., "dbt optimization")
    - subtopics: Related subjects discovered along the way
    - conversation_ids: All conversations in this thread (newest first)
    - memory_ids: All memories created during this journey
    - last_updated: When we last worked on this problem
    - total_interactions: Measure of topic importance/complexity
    - evolution_summary: AI-generated story of how understanding evolved
    - pattern_signatures: Extracted patterns for future recognition
    """

    # Unique identifier for this thread
    thread_id: str
    # Main topic - like "dbt errors" or "SQL optimization"
    primary_topic: str
    # Related topics - like ["CTEs", "window functions", "performance"]
    subtopics: List[str]
    # All conversations in this thread, newest first, this lets us see the timeline of problem-solving
    conversation_ids: List[str]
    # Every memory created in this thread
    memory_ids: List[str]
    # When did we last work on this topic?
    last_updated: datetime
    # How many times have we discussed this? Shows topic importance
    # Like "Started with timeout errors, found it was missing indexes",
    # finally optimized with materialized views
    total_interactions: int
    # AI-generated summary of how the problem/solution evolved
    # Like "Started with timeout errors, found it was missing indexes"
    evolution_summary: Optional[str] = None
    # Pattern signatures extracted from this thread
    pattern_signatures: List[str] = field(default_factory=list)

class MemoryNetworkSystem:
    """
    Manages the network of interconnected memories and conversations

    This is ADAM's "brain structure" - it doesn't just store memories,
    it understands how they connect and relate to each other.
    """
    
    # Memory decay parameters
    DECAY_THRESHOLD_DAYS = 30  # Start decay after 30 days
    MIN_ACCESS_COUNT_FOR_PERSISTENCE = 3  # Memories accessed < 3 times decay faster
    DECAY_RATE = 0.1  # 10% decay per month after threshold
    
    def __init__(self, base_memory_system, conversation_system):
        """
        Initialize the memory network

        Args:
            base_memory_system: Your original ADAMMemoryAdvanced system
            conversation_system: The conversation tracking system
        """

        # Keep reference to the original memory system - we build on top of it!
        self.base_memory = base_memory_system
        # Keep reference to conversation tracker
        self.conversation_system = conversation_system
        
        # THE CORE DATA STRUCTURE: A directed graph where:
        # - Nodes = memories
        # - Edges = references between memories
        # - Edge weights = strenght of relationship
        # DiGraph - Directed Graph (references have direction: A references B)
        self.memory_graph = nx.DiGraph()
        
        # Quick lookup indices - like a library card catalog
        # topic_to_memories: "SQL" -> {mem1, mem2, mem5, mem8}
        # Allows fast "show me all memories about SQL"
        self.topic_to_memories: Dict[str, Set[str]] = defaultdict(set)

        # topic_to_threads: "dbt" -> ["thread_dbt_001", "thread_dbt_002"]
        # Allows finding all conversation threads about a topic
        self.topic_to_threads: Dict[str, List[str]] = defaultdict(list)
        
        # All active conversation threads
        self.threads: Dict[str, ConversationThread] = {}
        
        # Where to save the network structure on disk
        self.network_path = Path("./adam_memory_advanced/memory_network")
        # Create directorye if it doesn't exist
        self.network_path.mkdir(exist_ok=True)
        
        # Load any previously saved network
        self._load_network()
    
    def add_memory_with_references(self,
                                  query: str,
                                  response: str,
                                  memory_type: str,
                                  topics: List[str],
                                  potential_references: Optional[List[str]] = None,
                                  *,
                                  auto_save: bool = True) -> str:
        """
        Add a new memory and automatically find and link related memories.

        This is like adding a new research paper that cites previous work -
        we need to find what existing knowledge this builds upon.  After the
        new memory is inserted into the network, the current state of the
        network is automatically persisted to disk so it is available on the
        next load.  The auto-save behavior can be disabled when performing
        bulk updates by passing ``auto_save=False``.

        THE PROCESS:
        1. First, the base memory system evaluates worthiness
        2. If worthy, we create a MemoryNode with full metadata
        3. We find related memories to reference (if not provided)
        4. We create weighted edges to referenced memories
        5. We update the referenced memories to know about this reference
        6. We update indices for fast topic-based retrieval
        7. We update or create a conversation thread
        8. We persist everything to disk

        WHY THIS MATTERS:
        - Automatic reference discovery builds a self-organizing knowledge graph
        - Bidirectional references enable traversal in both directions
        - Weighted connections distinguish strong vs. weak relationships
        - Thread tracking shows how understanding evolves over time

        EXAMPLE SCENARIO:
        User: "My dbt model is slow" (Monday)
        -> Creates memory M1, starts thread T1
        
        User: "I tried indexes but still slow" (Tuesday)  
        -> Creates memory M2, references M1 (weight: 0.8)
        -> Updates thread T1 with evolution
        
        User: "Fixed it with incremental models!" (Wednesday)
        -> Creates memory M3, references M1 & M2
        -> Thread T1 now shows complete problem->solution journey

        Args:
            query: The question/problem
            response: ADAM's answer/solution
            memory_type: Classification (error_solution, explanation, etc.)
            topics: List of topics this memory covers
            potential_references: Optional list of memory IDs to reference.
                                  If ``None``, we'll find them automatically.
            auto_save: Persist the network to disk after adding the memory.

        Returns:
            memory_id: The ID of the newly created memory
        """
        
        # First, use the base memory system to store the memory
        # This handles embeddings, vector storage, worthiness checks
        memory_id = self.base_memory.remember_if_worthy(
            query=query,
            response=response,
            generation_cost=0.001,  # Will be calculated properly in real implementation
            model_used="mistral"
        )
        
        # If the base system rejected it as not worthy, we're done
        if not memory_id:
            return None
        
        # NOW THE NETWORK MAGIC: Find what this memory should reference
        # If references weren't provided, intelligently find related memories
        if potential_references is None:
            potential_references = self._find_related_memories(query, topics)
        
        # Get embedding if available
        embedding = None
        if hasattr(self.base_memory, 'get_embedding'):
            try:
                embedding = self.base_memory.get_embedding(query)
            except:
                pass  # Embedding generation failed, continue without it
        
        # Create our memory node with all its metadata
        memory_node = MemoryNode(
            memory_id=memory_id,
            conversation_id=self.conversation_system.current_session.session_id,
            timestamp=datetime.now(),
            query=query,
            response=response,
            topics=topics,
            memory_type=memory_type,
            references=potential_references, # Who we build upon
            embedding=embedding
        )
        
        # Add this node to our graph
        # The node stores the full MemoryMode object as its data
        self.memory_graph.add_node(memory_id, data=memory_node)
        
        # Create edges (connections) to referenced memories
        for ref_id in potential_references:
            # Make sure the referenced memory exists in our graph
            if self.memory_graph.has_node(ref_id):
                # Calculate relevance weight based on topic overlap and recency
                # Higher weight = stronger relationship
                weight = self._calculate_reference_weight(memory_node, ref_id)

                # Add directed edge: memory_id -> ref_id with weight
                # Direction matters! This memory references that one, not vice versa
                self.memory_graph.add_edge(memory_id, ref_id, weight=weight)

                # Store the weight for quick access
                memory_node.reference_weights[ref_id] = weight
                
                # BIDIRECTIONAL TRACKING: Update the refrenced memory
                # to know that this new memory refrences it
                ref_node = self.memory_graph.nodes[ref_id]['data']
                ref_node.referenced_by.append(memory_id)
        
        # Update our topic indices for fast topic-based search
        for topic in topics:
            # Ensure the topic exists in our dictionary
            if topic not in self.topic_to_memories:
                self.topic_to_memories[topic] = set()
            self.topic_to_memories[topic].add(memory_id)
        
        # Update or the conversation thread for this topic
        self._update_conversation_thread(memory_node)

        if auto_save:
            self._save_network()

        return memory_id
    
    def _find_related_memories(self, query: str, topics: List[str], 
                             max_references: int = 5) -> List[str]:
        """
        Intelligently find memories that this new memory should reference

        This is like a research assistant finding relevant prior work -
        we look for memories that cover similar topics and might provide
        context or foundation for this new memory.

        THE ALGORITHM:
        1. Find all memories sharing at least one topic (candidate pool)
        2. Score each candidate by multiple factors:
           - Topic overlap (how many topics in common)
           - Recency (newer memories often more relevant)
           - Importance (how often others reference this memory)
           - Semantic similarity (if embeddings available)
        3. Return top N highest scoring memories

        SCORING PHILOSOPHY:
        We use a weighted multi-factor approach because:
        - Topic overlap is the strongest signal (50% weight)
        - Recent memories likely have updated solutions (30% weight)
        - Popular memories are probably important (20% weight)
        
        This creates a balanced scoring that finds truly relevant memories,
        not just the newest or most popular ones.

        EXAMPLE:
        New memory: "How to optimize dbt models with CTEs"
        Topics: ["dbt", "optimization", "CTE"]
        
        Finds and scores:
        - Memory A: "Basic dbt tutorial" [topics: "dbt"] -> Score: 0.3
        - Memory B: "Optimizing SQL CTEs" [topics: "SQL", "CTE", "optimization"] -> Score: 0.7
        - Memory C: "Recent dbt performance fix" [topics: "dbt", "performance"] -> Score: 0.8
        
        Returns: [Memory C, Memory B] (top 2)

        Args:
            query: The current question (helps find semantic similarity)
            topics: Topics of the new memory
            max_references: Maximum number of references to create

        Returns:
            List of memory IDs that this memory should reference
        """
        related_memories = []
        
        # Step 1: Find all memories that share at leas one topic
        # This is our candidate pool
        topic_memories = set()
        for topic in topics:
            # Union operation: add all memories for this topic
            topic_memories.update(self.topic_to_memories.get(topic, set()))
        
        # If no memories share topics, nothing to reference
        if not topic_memories:
            return []
        
        # Step 2: Score each candidate by relevance
        # We'll consider multiple factors to find the BEST references
        scored_memories = []

        for mem_id in topic_memories:
            # Skip if memory doesn't exist in graph (defensive programming)
            if not self.memory_graph.has_node(mem_id):
                continue
                
            # Get the full memory node
            mem_node = self.memory_graph.nodes[mem_id]['data']
            
            # SCORING ALGORITHM - multiple factors contribute:
            
            # 1. Topic overlap: How many topics do we share?
            # More shared topics = more relevant
            # Example: if both discuss "dbt" and "testing" vs just "dbt"
            shared_topics = set(topics) & set(mem_node.topics)
            topic_overlap = len(shared_topics) / len(topics)

            # 2. Recency: Newer memories often more relevant
            # Recent memories might have updated solutions
            recency_score = self._calculate_recency_score(mem_node.timestamp)

            # 3. Importante: How often has this memory been referenced?
            # If many memories reference it, it's probably important
            # Cap at 0.2 to prevent over-weighting popular memories
            importance_score = min(len(mem_node.referenced_by) * 0.1, 0.2)  # Boost popular memories
            
            # 4. Semantic similarity using embeddings
            semantic_score = 0.0
            if hasattr(self.base_memory, 'get_embedding'):
                # Get embeddings for semantic comparison
                query_embedding = self.base_memory.get_embedding(query)
                if mem_node.embedding is not None and query_embedding is not None:
                    # Cosine similarity between embeddings
                    semantic_score = np.dot(query_embedding, mem_node.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(mem_node.embedding)
                    )

            # Combine scores with weights
            # These weights (0.5, 0.3, 0.2) are tunable hyperparameters
            total_score = (topic_overlap * 0.4 +          # Topic match important
                          recency_score * 0.2 +           # Then recency
                          semantic_score * 0.3 +          # Semantic similarity
                          min(importance_score, 0.1))     # Then popularity
            
            scored_memories.append((mem_id, total_score))
        
        # Step 3: Sort by score and return top N
        # Higher scores = more relevant = should be referenced
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # Extract just memory IDs from the top scored memories
        return [mem_id for mem_id, _ in scored_memories[:max_references]]
    
    def _calculate_reference_weight(self, new_memory: MemoryNode, 
                                  ref_memory_id: str) -> float:
        """
        Calculate the strength of the reference relationship

        Not all references are equal - some memories are strongly related,
        others just tangentially. This weight helps ADAM understand which
        references are most important when reconstructing context.

        Args:   
            new_memory: The memory doing the referencing
            ref_memory_id: The memory being referenced
        
        Returns:
            Weight between 0.0 (weak) and 1.0 (strong)
        """
        # Get the referenced memory's data
        ref_node = self.memory_graph.nodes[ref_memory_id]['data']
        
        # Factor 1: Topic similarity
        # If memories share many topics, they're strongly related
        topic_overlap = len(set(new_memory.topics) & set(ref_node.topics))
        # Normalize by the number of topics in the new memory
        topic_score = topic_overlap / max(len(new_memory.topics), 1)
        
        # Factor 2: Temporal distance
        # Recent memories are more likely to be directly related
        # Calculate time difference in seconds
        time_diff = (new_memory.timestamp - ref_node.timestamp).total_seconds()
        # Convert to hours for easier reasoning
        hours_diff = time_diff / 3600

        # Use exponential decay: relevance decrease over time
        # After 24 hours, socre is ~0.5; after a week, ~0.14
        temporal_score = 1.0 / (1.0 + hours_diff / 24)  # Decay over days
        
        # Factor 3: Conversation continuity
        # Memories from the same conversation are more likely related
        # Same conversation = discussing the same problem
        continuity_score = 1.0 if new_memory.conversation_id == ref_node.conversation_id else 0.5
        
        # Combine factors with weights
        # These weights determine relative importance:
        # - Topic match is most important (50%)
        # - Recency matters (30%)
        # - Same conversation helps (20%)
        return (topic_score * 0.5 + temporal_score * 0.3 + continuity_score * 0.2)
    
    def _calculate_recency_score(self, timestamp: datetime) -> float:
        """
        Calculate how recent a memory is (0-1 scale)

        Recent memories are often more relevant because:
        1. They might contain updated solutions
        2. They reflect current understanding
        3. The context is likely still similar

        Args:
            timestamp: When the memory was created
        Returns:
            Score from 0.0 (very old) to 1.0 (brand new)
        """

        # Calculate age in hours
        age = (datetime.now() - timestamp).total_seconds() / 3600  # Hours

        # Exponential decay over a week (168 hours)
        # After 1 day: ~0.86
        # After 3 days: ~0.67
        # After 1 week: ~0.50
        # After 1 month: ~0.15
        
        # Apply memory decay for old, unused memories
        # Get memory node to access its decay-related fields
        decay_factor = 1.0
        for node_id in self.memory_graph.nodes():
            node = self.memory_graph.nodes[node_id]['data']
            if node.timestamp == timestamp:
                decay_factor = self._calculate_memory_decay(
                    timestamp, 
                    node.access_count, 
                    node.last_accessed
                )
                break
        
        return (1.0 / (1.0 + age / 168)) * decay_factor  # Decay over a week with additional decay
    
    def _calculate_memory_decay(self, timestamp: datetime, access_count: int = 0, 
                               last_accessed: Optional[datetime] = None) -> float:
        """
        Calculate memory decay factor based on age and usage patterns
        
        Implements forgetting curve: memories decay unless reinforced through access
        
        THE DECAY MODEL:
        Based on Ebbinghaus's forgetting curve with modifications:
        1. Recent memories (< 30 days) don't decay
        2. Base decay follows exponential curve: e^(-rate * time)
        3. Frequently accessed memories decay slower
        4. Long-unused memories decay faster
        
        DECAY FORMULA:
        decay = base_decay * access_boost * recency_penalty
        
        Where:
        - base_decay = e^(-0.1 * months_old)
        - access_boost = min(access_count / 3, 2.0)
        - recency_penalty = 0.5 if not accessed in 60+ days
        
        WHY THIS MATTERS:
        1. Prevents memory bloat - old, unused memories fade away
        2. Preserves important knowledge - frequently used memories persist
        3. Mimics human memory - we forget what we don't use
        4. Maintains system performance - fewer memories to search
        
        EXAMPLE SCENARIOS:
        - New memory (5 days old): decay = 1.0 (no decay)
        - Old but popular (6 months, accessed 10 times): decay = 0.8
        - Old and forgotten (1 year, never accessed): decay = 0.05
        
        THRESHOLDS:
        - decay < 0.1: Memory should be removed
        - decay < 0.5: Memory is fading, needs reinforcement
        - decay > 0.8: Memory is fresh and relevant
        
        Args:
            timestamp: When memory was created
            access_count: How many times memory was accessed
            last_accessed: When memory was last accessed
            
        Returns:
            Decay factor from 0.0 (forgotten) to 1.0 (fresh)
        """
        # Calculate age in days
        age_days = (datetime.now() - timestamp).days
        
        # No decay for recent memories
        if age_days < self.DECAY_THRESHOLD_DAYS:
            return 1.0
        
        # Calculate base decay
        months_old = (age_days - self.DECAY_THRESHOLD_DAYS) / 30
        base_decay = np.exp(-self.DECAY_RATE * months_old)
        
        # Boost for frequently accessed memories
        access_boost = min(access_count / self.MIN_ACCESS_COUNT_FOR_PERSISTENCE, 2.0)
        
        # Additional decay if not accessed recently
        if last_accessed:
            days_since_access = (datetime.now() - last_accessed).days
            if days_since_access > 60:  # Not accessed in 2 months
                base_decay *= 0.5
        
        return min(base_decay * access_boost, 1.0)
    
    def _update_conversation_thread(self, memory_node: MemoryNode):
        """
        Update or create a conversation thread for this topic

        Thread track the volution of dicussions about a topic across
        multiple conversations. Like chapters in a book about solving
        a particular problem.

        Args:
            memory_node: The new memory to add to a thread
        """

        # Use the first topic as primary (could be smarter about this)
        primary_topic = memory_node.topics[0] if memory_node.topics else "general"
        
        # Find existing thread or create new one
        thread_id = None
        for tid, thread in self.threads.items():
            if primary_topic == thread.primary_topic:
                thread_id = tid
                break
        
        if thread_id:
            # Update existing thread
            thread = self.threads[thread_id]

            # Add this conversation if it's new to the thread
            if memory_node.conversation_id not in thread.conversation_ids:
                # Insert at beginning - newest conversation first
                thread.conversation_ids.insert(0, memory_node.conversation_id)

            # Add this memory to the thread
            thread.memory_ids.append(memory_node.memory_id)

            # Update metadata
            thread.last_updated = datetime.now()
            thread.total_interactions += 1
        else:
            # Create new thread
            # Generate unique ID with timestamp
            thread_id = f"thread_{primary_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            thread = ConversationThread(
                thread_id=thread_id,
                primary_topic=primary_topic,
                subtopics=memory_node.topics[1:], # Rest are subtopics
                conversation_ids=[memory_node.conversation_id],
                memory_ids=[memory_node.memory_id],
                last_updated=datetime.now(),
                total_interactions=1
            )

            # Store the thread
            self.threads[thread_id] = thread
            # Index it by topic for fast lookup
            self.topic_to_threads[primary_topic].append(thread_id)
    
    def get_memory_context_chain(self, memory_id: str, 
                               max_depth: int = 5) -> List[MemoryNode]:
        """
        Get the chain of related memories, traversing references in reverse chronological order

        This is what ADAM uses to understand the full context of a problem.
        Like following citations in a research paper to understand the
        complete background

        THE TRAVERSAL ALGORITHM:
        Uses depth-first search to follow reference chains, prioritizing
        stronger connections. This creates a "context chain" that tells
        the complete story of how we arrived at current understanding.

        WHY THIS MATTERS:
        When ADAM needs to understand a complex problem, he doesn't just
        look at the most recent memory - he traces back through the entire
        learning journey to build complete context.

        EXAMPLE TRAVERSAL:
        Starting from: "Solution: Use incremental models"
        
        Traverses to find:
        1. "Solution: Use incremental models" (starting point)
        2. "Tried indexes but still slow" (weight: 0.9)
        3. "Initial problem: dbt model timeout" (weight: 0.8)
        4. "General dbt optimization tips" (weight: 0.6)
        
        This gives ADAM the full context of the problem evolution.

        DESIGN DECISIONS:
        - max_depth prevents infinite traversal in circular references
        - Visited set prevents processing same memory twice
        - Sorting by weight ensures most relevant references first
        - Recursive approach naturally builds chronological chain

        Args:
            memory_id: Starting memory (usually the most recent)
            max_depth: How many levels of references to follow
        Returns:
            List of MemoryNodes, starting with the requested memory
        """

        # Check if memory exists
        if not self.memory_graph.has_node(memory_id):
            return []
        
        # Track what we find
        context_chain = []
        # Prevent infinite loops if there are circular references
        visited = set()
        
        def traverse_references(mem_id: str, depth: int):
            """
            Recursive function to follow reference chain

            Think of this like archeology - we dig deeper into the past,
            following references to understand the full history.
            """

            # Stop conditions: too deep or already seen this memory
            if depth > max_depth or mem_id in visited:
                return
            
            # Mark as visited to prevent loops
            visited.add(mem_id)

            # Get the memory node
            node = self.memory_graph.nodes[mem_id]['data']
            # Add to our context chain
            context_chain.append(node)
            
            # Sort references by weight (strongest relationships first)
            # This ensures we follow the most relevant references
            sorted_refs = sorted(
                node.references,
                key=lambda ref: node.reference_weights.get(ref, 0),
                reverse=True
            )
            
            # Recursively traverse each reference
            for ref_id in sorted_refs:
                traverse_references(ref_id, depth + 1)
        
        # Start the traversal
        traverse_references(memory_id, 0)

        return context_chain
    
    def get_thread_summary(self, topic: str) -> Dict[str, any]:
        """
        Get a comprehensive summary of all conversations about a topic

        This gives ADAM a bird's-eye view of an entire problem-solving journey,
        not just individual memories.

        Args:
            topic: The topic to summarize (e.g., "dbt debugging")
        Returns:
            Dictionary with thread analysis
        """

        # Find the most recent thread for this topic
        thread_ids = self.topic_to_threads.get(topic, [])
        if not thread_ids:
            return None
        
        # Get the most recently updated thread
        latest_thread = max(
            [self.threads[tid] for tid in thread_ids],
            key=lambda t: t.last_updated
        )
        
        # Gather all memories in the thread
        thread_memories = []
        for mem_id in latest_thread.memory_ids:
            if self.memory_graph.has_node(mem_id):
                thread_memories.append(self.memory_graph.nodes[mem_id]['data'])
        
        # Sort by timestamp (newest first)
        thread_memories.sort(key=lambda m: m.timestamp, reverse=True)
        
        # Build summary
        summary = {
            'thread_id': latest_thread.thread_id,
            'primary_topic': latest_thread.primary_topic,
            'total_conversations': len(latest_thread.conversation_ids),
            'total_memories': len(thread_memories),
            'last_updated': latest_thread.last_updated,
            # Trace how the problem evolved over time
            'evolution': self._trace_topic_evolution(thread_memories),
            # Extract the most important learnings
            'key_insights': self._extract_key_insights(thread_memories),
            # Find what's still not resolved
            'unresolved_issues': self._find_unresolved_issues(thread_memories),
            # The full memory chain for detailed exploration
            'memory_chain': [m.memory_id for m in thread_memories]
        }
        
        return summary
    
    def _trace_topic_evolution(self, memories: List[MemoryNode]) -> str:
        """
        Trace how a topic/problem evolved over time

        This helps ADAM understand the journey, not just the destination.
        Like reading a detective's case notes to see how they solved it.

        Args:
            memories: List of memories in chronological order
        Returns:
            Natural language description of the evolution
        """
        if not memories:
            return "No evolution history"
        
        # This would use more sophisticated NLP in practice
        evolution_points = []
        
        # Analyze memories to find key transitions
        # In a real implementation, this would use NLP
        for i, mem in enumerate(memories):
            if i == 0:
                # First memory - how it all started
                evolution_points.append(f"Started with: {mem.query[:50]}...")
            elif "error" in mem.query.lower() and "error" not in memories[i-1].query.lower():
                # Transition to error state
                evolution_points.append(f"Encountered error: {mem.query[:50]}...")
            elif "work" in mem.query.lower() or "success" in mem.response.lower():
                # Found a solution
                evolution_points.append(f"Found solution: {mem.response[:50]}...")
        
        # Join with arrows to show progression
        evolution = " → ".join(evolution_points)
        
        # Update thread with evolution summary if we have one
        thread_id = None
        for tid, thread in self.threads.items():
            if any(m.memory_id in thread.memory_ids for m in memories):
                thread_id = tid
                break
        
        if thread_id:
            self.threads[thread_id].evolution_summary = evolution
            # Extract patterns from this evolution
            patterns = self._extract_patterns_from_memories(memories)
            self.threads[thread_id].pattern_signatures = patterns
        
        return evolution
    
    def _extract_key_insights(self, memories: List[MemoryNode]) -> List[str]:
        """
        Extract the most important learnings from a thread

        Insights are memories that proved particularly valuable -
        ofter referenced by many other memories.

        Args:
            memories: All memories in the thread
        Returns:
            List of important insights
        """
        insights = []
        
        # Look for memories that were referenced many times (important insights)
        for mem in memories:
            if len(mem.referenced_by) > 2: # Referenced by 3+ other memories
                insights.append(f"Important insight (referenced {len(mem.referenced_by)} times): {mem.response[:100]}...")
        
        # Return top 3 most important insights
        return insights[:3]  # Top 3 insights
    
    def _find_unresolved_issues(self, memories: List[MemoryNode]) -> List[str]:
        """
        Find issues that were raised but not resolved

        These are questions without answers or problems without solutions -
        important for ADAM to track what's still pending

        Args:
            memories: All memories in the thread
        Returns:
            List of unresolved questions/issues
        """
        unresolved = []
        
        # Simple heuristic: questions without successful follow-ups
        for i, mem in enumerate(memories):
            if "?" in mem.query and i > 0:  # It's a question
                # Check if any later memory indicates resolution
                next_memories = memories[i+1:]  # Later memories
                resolved = any(
                    "work" in m.query.lower() or 
                    "success" in m.response.lower() or
                    "solved" in m.response.lower()
                    for m in next_memories
                )

                # If not resolved, track it
                if not resolved:
                    unresolved.append(mem.query[:100])
        
        return unresolved
    
    def generate_contextual_recap(self, topic: str) -> str:
        """
        Generate a natural language recap of a topic thread

        This is what ADAM shows the user - a friendly, comprehensive
        summary of their joruney with a particular problem or topic.

        Args:
            topic: The topic to recap
        Returns:
            Human-friendly recap text
        """

        # Get the thread summary data
        summary = self.get_thread_summary(topic)
        if not summary:
            return f"I don't have any previous conversations about {topic}."
        
        # Build the recap piece by piece
        recap_parts = [
            f"Let me recap our journey with {summary['primary_topic']}:",
            f"\nWe've had {summary['total_conversations']} conversations about this, ",
            f"spanning {summary['total_memories']} specific points of discussion.",
            f"\n\nEvolution of the problem:\n{summary['evolution']}",
        ]
        
        # Add key insights if we found any
        if summary['key_insights']:
            recap_parts.append("\n\nKey insights we've discovered:")
            for insight in summary['key_insights']:
                recap_parts.append(f"- {insight}")
        
        # Add unresolved issues if any remain
        if summary['unresolved_issues']:
            recap_parts.append("\n\nOpen questions we haven't fully resolved:")
            for issue in summary['unresolved_issues']:
                recap_parts.append(f"- {issue}")
        
        # Most recent memory for continuity
        if summary['memory_chain']:
            latest_memory_id = summary['memory_chain'][0]
            latest_memory = self.memory_graph.nodes[latest_memory_id]['data']
            recap_parts.append(
                f"\n\nMost recently, we were discussing:\n'{latest_memory.query}'"
            )
        
        # Join all parts into a cohesive recap
        return "\n".join(recap_parts)
    
    def continue_thread(self, topic: str) -> Tuple[str, List[str]]:
        """
        Get the context needed to continue a conversation thread
        
        This is called when the users want to resume working on a problem.
        It provides both a human-friendly recap and the memory IDs for ADAM
        to load full context.

        Args:
            topic: The topic to continue
        Returns:
            Tuple of (recap_text, list_of_memory_id)
        """

        # Get thread summary
        summary = self.get_thread_summary(topic)
        if not summary:
            return "Let's start fresh with this topic.", []
        
        # Get the most recent memory
        latest_memory_id = summary['memory_chain'][0]
        
        # Get its context chain
        context_chain = self.get_memory_context_chain(latest_memory_id, max_depth=3)
        
        # Generate human-friendly recap
        recap = self.generate_contextual_recap(topic)
        # Extract memory IDs for ADAM to load
        memory_ids = [mem.memory_id for mem in context_chain]
        
        return recap, memory_ids
    
    def visualize_memory_network(self, topic: Optional[str] = None, 
                               show_decay: bool = True,
                               highlight_patterns: bool = True):
        """
        Create an enhanced visual representation of the memory network
        
        This visualization helps debug and understand ADAM's knowledge structure,
        showing decay states, access patterns, and thread connections.

        VISUALIZATION FEATURES:
        1. Node Colors: Indicate memory type
           - Red: Error solutions
           - Blue: Explanations  
           - Green: Patterns
           - Gray: Other
        
        2. Node Size: Shows importance
           - Larger = more accessed + more referenced
           - Small = rarely used
        
        3. Node Transparency: Shows decay status
           - Opaque = fresh memory
           - Transparent = decaying memory
        
        4. Edge Styles: Show connection strength
           - Solid thick = strong connection (>0.7)
           - Dashed thin = weak connection (<0.7)
        
        5. Labels: Show memory preview
           - First 6 chars of ID
           - First 3 words of query
        
        WHY VISUALIZE:
        - Debug memory connections
        - Identify memory clusters
        - Spot decaying memories
        - Understand topic relationships
        - Verify pattern recognition
        
        USAGE TIPS:
        - Start with topic-specific views for clarity
        - Look for disconnected memories (might be orphaned)
        - Check for overly connected nodes (might be too general)
        - Identify fading clusters (might need reinforcement)

        Args:
            topic: Optional - visualize only memories about this topic
            show_decay: Show memory decay status with node transparency
            highlight_patterns: Highlight memories that are part of patterns
        Returns:
            matplotlib figure object
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        # Decide what to visualize
        if topic:
            # Filter to specific topic
            relevant_memories = self.topic_to_memories.get(topic, set())
            # Create subgraph with only these memories
            subgraph = self.memory_graph.subgraph(relevant_memories)
        else:
            # Visualize entire network
            subgraph = self.memory_graph
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Calculate node positions using hierarchical layout for better structure
        # This groups related memories together
        if len(subgraph) > 1:
            pos = nx.spring_layout(subgraph, k=3, iterations=100, seed=42)
        else:
            pos = {list(subgraph.nodes())[0]: (0.5, 0.5)}
        
        # Prepare node visualization data
        node_colors = []
        node_sizes = []
        node_alphas = []
        
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]['data']
            
            # Color by memory type
            if node_data.memory_type == "error_solution":
                color = 'red'
            elif node_data.memory_type == "explanation":
                color = 'blue'
            elif node_data.memory_type == "pattern":
                color = 'green'
            else:
                color = 'gray'
            node_colors.append(color)
            
            # Size by importance (access count + reference count)
            importance = node_data.access_count + len(node_data.referenced_by)
            node_sizes.append(300 + importance * 50)
            
            # Alpha by decay if enabled
            if show_decay:
                decay_factor = self._calculate_memory_decay(
                    node_data.timestamp,
                    node_data.access_count,
                    node_data.last_accessed
                )
                node_alphas.append(max(0.3, decay_factor))
            else:
                node_alphas.append(0.8)
        
        # Draw nodes
        for i, node in enumerate(subgraph.nodes()):
            nx.draw_networkx_nodes(
                subgraph, pos,
                nodelist=[node],
                node_color=[node_colors[i]],
                node_size=[node_sizes[i]],
                alpha=node_alphas[i],
                ax=ax
            )
        
        # Draw edges with varying styles
        for u, v in subgraph.edges():
            weight = subgraph[u][v].get('weight', 0.5)
            
            # Strong connections are solid, weak are dashed
            if weight > 0.7:
                style = 'solid'
                width = weight * 4
            else:
                style = 'dashed'
                width = weight * 2
                
            nx.draw_networkx_edges(
                subgraph, pos,
                edgelist=[(u, v)],
                width=width,
                alpha=0.6,
                edge_color='gray',
                style=style,
                ax=ax
            )
        
        # Add detailed labels
        labels = {}
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]['data']
            # Show first few words of query
            query_preview = ' '.join(node_data.query.split()[:3]) + "..."
            labels[node] = f"{node[:6]}\n{query_preview}"
        
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=6, ax=ax)
        
        # Create legend
        legend_elements = [
            mpatches.Patch(color='red', label='Error Solutions'),
            mpatches.Patch(color='blue', label='Explanations'),
            mpatches.Patch(color='green', label='Patterns'),
            mpatches.Patch(color='gray', label='Other'),
        ]
        
        if show_decay:
            legend_elements.extend([
                mpatches.Patch(color='white', label=''),  # Spacer
                mpatches.Patch(color='black', alpha=1.0, label='Fresh Memory'),
                mpatches.Patch(color='black', alpha=0.3, label='Decaying Memory'),
            ])
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        # Add title with statistics
        memory_count = len(subgraph)
        edge_count = len(subgraph.edges())
        avg_connections = edge_count / memory_count if memory_count > 0 else 0
        
        title = f"Memory Network {'for ' + topic if topic else '(All Topics)'}\n"
        title += f"{memory_count} memories, {edge_count} connections (avg: {avg_connections:.1f})"
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.axis('off')
        plt.tight_layout()
        
        return fig
    
    def _save_network(self):
        """
        Persist the memory network to disk

        This snures ADAM's knowledge graph survives restarts.
        We save both the structure and the metadata.
        """
        # Save the graph structure using NetworkX's pickle format
        # This preserves all node and edge data
        # Note: In NetworkX 3.x, write_gpickle is deprecated, use pickle directly
        import pickle
        with open(self.network_path / "memory_graph.gpickle", 'wb') as f:
            pickle.dump(self.memory_graph, f)
        
        # Save our indices as JSON for human readability
        with open(self.network_path / "indices.json", 'w') as f:
            json.dump({
                # Convert sets to lists for JSON serialization
                'topic_to_memories': {k: list(v) for k, v in self.topic_to_memories.items()},
                'topic_to_threads': dict(self.topic_to_threads)
            }, f, indent=2)
        
        # Save threads
        threads_data = {}
        for tid, thread in self.threads.items():
            # Convert thread object to dictionary
            threads_data[tid] = {
                'thread_id': thread.thread_id,
                'primary_topic': thread.primary_topic,
                'subtopics': thread.subtopics,
                'conversation_ids': thread.conversation_ids,
                'memory_ids': thread.memory_ids,
                'last_updated': thread.last_updated.isoformat(),
                'total_interactions': thread.total_interactions,
                'evolution_summary': thread.evolution_summary,
                'pattern_signatures': thread.pattern_signatures
            }
        
        with open(self.network_path / "threads.json", 'w') as f:
            json.dump(threads_data, f, indent=2)
    
    def _load_network(self):
        """
        Load the memory network from disk

        This method is THE BRIDGE between ADAM's sessions. Without it, ADAM would forget everything
        when restarted. With it, ADAM's knowledge accumulates over days, weeks, and months.

        Think of this like waking up in the morning and remembering everything from yesterday -
        not just facts, but how those facts connect to each other.
        
        THE LOADING PROCESS:
        1. Load the graph structure (nodes and edges)
        2. Rebuild bidirectional references (referenced_by lists)
        3. Load topic indices for fast search
        4. Load conversation threads
        5. Resume any active sessions
        
        DATA STRUCTURES LOADED:
        - memory_graph.gpickle: The core NetworkX graph
        - indices.json: Topic-to-memory mappings
        - threads.json: Conversation thread metadata
        
        ERROR HANDLING:
        If any component fails to load, we log the error but continue.
        Better to have partial memory than no memory at all.
        
        CRITICAL RECONSTRUCTION:
        The bidirectional reference rebuild is crucial because:
        - Graph edges only store one direction (A->B)
        - But we need both directions for traversal
        - So we rebuild B.referenced_by = [A] from edges
        
        PERFORMANCE CONSIDERATIONS:
        Loading can be slow with thousands of memories, but it's
        a one-time cost at startup. The payoff is instant access
        to all historical knowledge.
        """
        # First, let's understand what we're loading:
        # 1. The graph structure (nodes = memories, edges = refereneces)
        # 2. Topic indices (for fast topic-based searches)
        # 3. Conversation threads (ongoing problem-solving journeys)

        # STEP 1: LOAD THE MEMORY GRAPH
        # This is the core structure - without it, we have no memories at all
        graph_file = self.network_path / "memory_graph.gpickle"
        
        # Check if we have a saved brain to load
        if graph_file.exists():
            try:
                # NetworkX's read_gpickle deserializes the entire graph structure
                # This includes all nodes (memories) and edges (references between memories)
                # Note: In NetworkX 3.x, read_gpickle is deprecated, use pickle directly
                import pickle
                with open(graph_file, 'rb') as f:
                    self.memory_graph = pickle.load(f)

                # Give feedback so we know the load succeeded
                # len(self.memory_graph.nodes) tells us how many memories ADAM has accumulated
                logger.info(f"Loaded memory graph with {len(self.memory_graph.nodes)} memories")
                
                # CRITICAL STEP: Rebuild the biderectional references
                # Here's why this is necessary:
                # - The graph edges store "Memory A references Memory B"
                # - But we also need to know "Memory B is referenced by Memory A"
                # - The edges only store one direction, so we rebuild the other

                # First, clear all referenced_by lists to start fresh
                # This prevents duplicates if the lists somehow got corrupted
                for node_id in self.memory_graph.nodes():
                    node_data = self.memory_graph.nodes[node_id]['data']
                    # Clear and rebuild referenced_by lists
                    node_data.referenced_by = []
                
                # Now, traverse every edge and build the reverse references
                for source, target in self.memory_graph.edges():
                    # If source references target, then target is referenced by source
                    target_node = self.memory_graph.nodes[target]['data']
                    target_node.referenced_by.append(source)
                
                # At this point, our memory network is fully reconstructed!
                # Every memory knows both:
                # - What memories it builds upon (references)
                # - What memories build upon it (referenced_by)

            except Exception as e:
                # If ANYTHING goes wrong loading the graph, we don't want ADAM to crash
                # Better to start fresh than to fail completely
                logger.error(f"Error loading graph: {e}")
                # Initialize empty graph - ADAM won't remember but can still work
                self.memory_graph = nx.DiGraph()
        else:
            # No saved graph exists - this might be ADAM's first run
            logger.info("No existing memory graph found - starting fresh!")
        
        # STEP 2: LOAD THE TOPIC INDICES
        # These are like the index in a book - they help us quickly find all memories
        # about a specific topic without searching through every single memory
        indices_file = self.network_path / "indices.json"
        
        if indices_file.exists():
            try:
                with open(indices_file, 'r') as f:
                    indices_data = json.load(f)
                    
                    # topic_to_memories: Maps each topic to all memories about that topic
                    # Example: "SQL" -> {"mem001", "mem047", "mem112", ...}
                    # We convert list back to sets for 0(1) lookup performance
                    # Use defaultdict to handle new topics automatically
                    self.topic_to_memories = defaultdict(set)
                    for k, v in indices_data.get('topic_to_memories', {}).items():
                        self.topic_to_memories[k] = set(v)

                    # topic_to_threads: Maps topics to conversation threads about them
                    # Example: "dbt debugging" -> ["thread_001", "thread_002"]
                    # defaultdict ensures we get empty list for new topics automatically
                    self.topic_to_threads = defaultdict(list, indices_data.get('topic_to_threads', {}))
                    
                logger.info(f"Loaded indices for {len(self.topic_to_memories)} topics")
            except Exception as e:
                logger.error(f"Error loading indices: {e}")
        
        # Load threads
        threads_file = self.network_path / "threads.json"
        if threads_file.exists():
            try:
                with open(threads_file, 'r') as f:
                    threads_data = json.load(f)
                    
                for tid, thread_dict in threads_data.items():
                    # Reconstruct thread objects from saved data
                    thread = ConversationThread(
                        thread_id=thread_dict['thread_id'],
                        primary_topic=thread_dict['primary_topic'],
                        subtopics=thread_dict.get('subtopics', []),
                        conversation_ids=thread_dict.get('conversation_ids', []),
                        memory_ids=thread_dict.get('memory_ids', []),
                        last_updated=datetime.fromisoformat(thread_dict['last_updated']),
                        total_interactions=thread_dict.get('total_interactions', 0),
                        evolution_summary=thread_dict.get('evolution_summary')
                    )
                    self.threads[tid] = thread
                    
                logger.info(f"Loaded {len(self.threads)} conversation threads")
                
                # Show a summary of active threads
                active_threads = [t for t in self.threads.values() 
                                if (datetime.now() - t.last_updated).days < 14]
                if active_threads:
                    logger.info(f"Active threads: {', '.join(t.primary_topic for t in active_threads[:5])}")
                    
            except Exception as e:
                logger.error(f"Error loading threads: {e}")

    # Additional helper method to ensure persistence after major operations
    def save_checkpoint(self):
        """
        Save the current state of the network
        Call this after significant changes to ensure persistence
        """
        try:
            self._save_network()
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def _extract_patterns_from_memories(self, memories: List[MemoryNode]) -> List[str]:
        """
        Extract common patterns from a sequence of memories
        
        Patterns are recurring problem-solution signatures that can be
        recognized in future scenarios
        
        Args:
            memories: List of memories to analyze
            
        Returns:
            List of pattern signatures
        """
        patterns = []
        
        # Pattern 1: Error → Solution sequences
        for i in range(len(memories) - 1):
            current = memories[i]
            next_mem = memories[i + 1]
            
            # Check for error-solution pattern
            if "error" in current.query.lower() and (
                "work" in next_mem.response.lower() or 
                "success" in next_mem.response.lower() or
                "fixed" in next_mem.response.lower()
            ):
                # Extract error type and solution approach
                error_keywords = self._extract_keywords(current.query, ["error", "fail", "issue"])
                solution_keywords = self._extract_keywords(next_mem.response, ["fix", "solve", "work"])
                
                if error_keywords and solution_keywords:
                    pattern = f"ERROR:{error_keywords[0]}→SOLUTION:{solution_keywords[0]}"
                    patterns.append(pattern)
        
        # Pattern 2: Progressive refinement (multiple attempts)
        attempt_count = 0
        for mem in memories:
            if "try" in mem.query.lower() or "attempt" in mem.query.lower():
                attempt_count += 1
        
        if attempt_count > 2:
            patterns.append("PATTERN:iterative_refinement")
        
        # Pattern 3: Tool/Technology specific patterns
        tech_keywords = ["sql", "dbt", "python", "javascript", "docker", "git"]
        for keyword in tech_keywords:
            if sum(1 for m in memories if keyword in m.query.lower()) > 2:
                patterns.append(f"TECH:{keyword}_recurring")
        
        return patterns
    
    def _extract_keywords(self, text: str, keyword_types: List[str]) -> List[str]:
        """
        Extract relevant keywords from text based on keyword types
        
        Args:
            text: Text to analyze
            keyword_types: Types of keywords to look for
            
        Returns:
            List of extracted keywords
        """
        # Simple keyword extraction - in production, use NLP
        words = text.lower().split()
        keywords = []
        
        for word in words:
            for kw_type in keyword_types:
                if kw_type in word:
                    # Get the full word/phrase containing the keyword
                    keywords.append(word.strip('.,!?;:'))
        
        return keywords
    
    def find_similar_patterns(self, current_query: str, current_context: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """
        Find threads with similar problem patterns to current query
        
        This enables ADAM to say "This looks similar to the issue we solved in March"
        
        THE PATTERN MATCHING PROCESS:
        1. Extract keywords from current query (error types, issues)
        2. Compare against stored pattern signatures in threads
        3. Check topic overlap between query and threads
        4. Apply recency boost (prefer recent similar problems)
        5. Return top matches above similarity threshold
        
        PATTERN SIGNATURES EXPLAINED:
        Threads store patterns like:
        - "ERROR:timeout→SOLUTION:indexes" (specific error-solution pair)
        - "PATTERN:iterative_refinement" (took multiple attempts)
        - "TECH:dbt_recurring" (technology-specific pattern)
        
        SIMILARITY SCORING:
        - Pattern match: +0.3 per matching pattern
        - Topic overlap: +0.5 * (shared_topics / total_topics)
        - Recency boost: +0.2 * (1 / (1 + days_old/30))
        
        WHY THIS MATTERS:
        Users often encounter similar problems. By recognizing patterns,
        ADAM can immediately suggest relevant past solutions, saving time
        and preventing repeated troubleshooting.
        
        EXAMPLE:
        User: "Getting timeout errors in my SQL query"
        
        Finds threads with:
        - Pattern: "ERROR:timeout→SOLUTION:indexes" (score +0.3)
        - Topics: ["SQL", "performance"] (score +0.5)
        - Recent: 5 days old (score +0.18)
        Total: 0.98 similarity
        
        ADAM: "This looks similar to the timeout issue we solved last week
                by adding indexes. Here's what worked..."
        
        Args:
            current_query: Current user query
            current_context: Current context (screen content, etc.)
            
        Returns:
            List of (thread_id, similarity_score) tuples
        """
        similar_threads = []
        
        # Extract patterns from current query
        current_keywords = self._extract_keywords(
            current_query, 
            ["error", "fail", "issue", "problem", "not working"]
        )
        
        # Check each thread for pattern matches
        for thread_id, thread in self.threads.items():
            similarity_score = 0.0
            
            # Check pattern signatures
            for pattern in thread.pattern_signatures:
                for keyword in current_keywords:
                    if keyword in pattern.lower():
                        similarity_score += 0.3
            
            # Check topic overlap
            query_topics = set(current_query.lower().split())
            thread_topics = set(thread.primary_topic.lower().split()) | set(
                topic.lower() for topic in thread.subtopics
            )
            
            topic_overlap = len(query_topics & thread_topics) / max(len(query_topics), 1)
            similarity_score += topic_overlap * 0.5
            
            # Check recency (prefer recent similar problems)
            days_old = (datetime.now() - thread.last_updated).days
            recency_boost = 1.0 / (1.0 + days_old / 30)
            similarity_score += recency_boost * 0.2
            
            if similarity_score > 0.3:  # Threshold for relevance
                similar_threads.append((thread_id, similarity_score))
        
        # Sort by similarity score
        similar_threads.sort(key=lambda x: x[1], reverse=True)
        
        return similar_threads[:5]  # Top 5 similar threads
    
    def update_memory_access(self, memory_id: str):
        """
        Update access count and timestamp when a memory is retrieved
        
        This is crucial for memory decay calculations - frequently accessed
        memories persist longer
        
        Args:
            memory_id: ID of the accessed memory
        """
        if self.memory_graph.has_node(memory_id):
            memory_node = self.memory_graph.nodes[memory_id]['data']
            memory_node.access_count += 1
            memory_node.last_accessed = datetime.now()
            
            # Also update access for strongly connected memories
            # This reinforces related knowledge
            for ref_id in memory_node.references:
                if self.memory_graph.has_node(ref_id):
                    ref_weight = memory_node.reference_weights.get(ref_id, 0)
                    if ref_weight > 0.7:  # Strong connection
                        ref_node = self.memory_graph.nodes[ref_id]['data']
                        ref_node.access_count += 0.5  # Partial reinforcement
                        ref_node.last_accessed = datetime.now()
    
    def cleanup_decayed_memories(self, decay_threshold: float = 0.1) -> int:
        """
        Remove memories that have decayed below the threshold
        
        This prevents the memory network from growing indefinitely with
        obsolete information
        
        Args:
            decay_threshold: Memories with decay factor below this are removed
            
        Returns:
            Number of memories removed
        """
        memories_to_remove = []
        
        for node_id in self.memory_graph.nodes():
            node_data = self.memory_graph.nodes[node_id]['data']
            
            # Calculate current decay factor
            decay_factor = self._calculate_memory_decay(
                node_data.timestamp,
                node_data.access_count,
                node_data.last_accessed
            )
            
            # Mark for removal if below threshold
            if decay_factor < decay_threshold:
                memories_to_remove.append(node_id)
        
        # Remove decayed memories
        for mem_id in memories_to_remove:
            # Remove from graph
            self.memory_graph.remove_node(mem_id)
            
            # Remove from topic indices
            for memories in self.topic_to_memories.values():
                memories.discard(mem_id)
            
            # Remove from threads
            for thread in self.threads.values():
                if mem_id in thread.memory_ids:
                    thread.memory_ids.remove(mem_id)
        
        # Save if we removed anything
        if memories_to_remove:
            self._save_network()
            logger.info(f"Removed {len(memories_to_remove)} decayed memories")
        
        return len(memories_to_remove)