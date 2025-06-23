#!/usr/bin/env python3
"""
Memory Network System - Connecting related conversations and memories
This creates a graph of interconnected memories that ADAM can traverse intelligently
"""

# We need these dataclasses (decorators) to structure our memory objects cleanly
from dataclasses import dataclass, field
# Type hints make our code self-documenting and catch errors early
from typing import List, Dict, Set, Optional, Tuple
# For timestamp tracking - knowing WHEN memories were created is crucial
from datetime import datetime
# defaultdict creates dictionaries that auto-initialize missing keys
from collections import defaultdict
# NetworkX is our graph library - it handles the complex relationships between memories
import networkx as nx
# For saving/loading our memory network to disk
import json
from pathlib import Path

@dataclass
class MemoryNode:
    """
    Represents a single memory with its relationships
    
    Think of this like a neuron in ADAM's brain:
    - It holds information (the memory itself)
    - It has connections to other neurons (references)
    - Those connections have different strengths (weights)
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
    # Like "Started with tiemout errors, found it was missing indexes"
    evolution_summary: Optional[str] = None

class MemoryNetworkSystem:
    """
    Manages the network of interconnected memories and conversations

    This is ADAM's "brain structure" - it doesn't just store memories,
    it understand show they connect and relate to each other.
    """
    
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
                                  potential_references: Optional[List[str]] = None) -> str:
        """
        Add a new memory and automatically find and link related memories

        This is like adding a new research paper that cites previous work -
        we need to find what existing knowledge this builds upon.

        Args:
            query: The question/problem
            response: ADAM's answer/solution
            memory_type: Classification (error_solution, explanation, etc.)
            topics: List of topics this memory covers
            potential_references: Optional list of memory IDS to reference
                                  If None, we'll find them automatically

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
        
        # Create our memory node with all its metadata
        memory_node = MemoryNode(
            memory_id=memory_id,
            conversation_id=self.conversation_system.current_session.session_id,
            timestamp=datetime.now(),
            query=query,
            response=response,
            topics=topics,
            memory_type=memory_type,
            references=potential_references # Who we build upon
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
            self.topic_to_memories[topic].add(memory_id)
        
        # Update or the conversation thread for this topic
        self._update_conversation_thread(memory_node)
        
        return memory_id
    
    def _find_related_memories(self, query: str, topics: List[str], 
                             max_references: int = 5) -> List[str]:
        """
        Intelligently find memories that this new memory should reference

        This is like a research assistant finding relevant prior work -
        we look for memories that cover similar topics and might provide
        context or foundation for this new memory.

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
            # In full implementation, we'd use vector similarity too

            #Combine scores with weights
            # These weights (0.5, 0.3, 0.2) are tunable hyperparameters
            total_score = (topic_overlap * 0.5 +          # Topic match most important
                          recency_score * 0.3 +           # Then recency
                          min(importance_score, 0.2))     # Then popularity
            
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
        return 1.0 / (1.0 + age / 168)  # Decay over a week
    
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
        return " → ".join(evolution_points)
    
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
    
    def visualize_memory_network(self, topic: Optional[str] = None):
        """
        Create a visual representation of the memory network
        
        This is incredibly useful for debugging and understanding how ADAM's knowledge
        is structured. You can literally SEE the connections between memories.

        Args:
            topic: Optional - visualize only memories about this topic
        Returns:
            matplotlib figure object
        """
        import matplotlib.pyplot as plt
        
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
        plt.figure(figsize=(12, 8))
        # Calculate node positions using spring layout
        # This creates a nice organic layout where connected nodes cluster
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
        
        # Color nodes by age (newer = brighter)
        node_colors = []
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]['data']
            age_hours = (datetime.now() - node_data.timestamp).total_seconds() / 3600
            node_colors.append(age_hours)
        
        nx.draw_networkx_nodes(
            subgraph, pos, 
            node_color=node_colors,
            cmap='viridis_r',
            node_size=500,
            alpha=0.8
        )
        
        # Draw edges with weights
        edges = subgraph.edges()
        weights = [subgraph[u][v].get('weight', 0.5) for u, v in edges]
        nx.draw_networkx_edges(
            subgraph, pos,
            width=[w * 3 for w in weights],
            alpha=0.5,
            edge_color='gray'
        )
        
        # Add labels (memory IDs)
        labels = {node: node[:8] for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
        
        plt.title(f"Memory Network {'for ' + topic if topic else '(All Topics)'}")
        plt.colorbar(label='Age (hours)')
        plt.axis('off')
        plt.tight_layout()
        
        return plt
    
    def _save_network(self):
        """
        Persist the memory network to disk

        This snures ADAM's knowledge graph survives restarts.
        We save both the structure and the metadata.
        """
        # Save the graph structure using NetworkX's pickle format
        # This preservers all node and edge data
        nx.write_gpickle(self.memory_graph, self.network_path / "memory_graph.gpickle")
        
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
                'evolution_summary': thread.evolution_summary
            }
        
        with open(self.network_path / "threads.json", 'w') as f:
            json.dump(threads_data, f, indent=2)
    
    def _load_network(self):
        """
        Load the memory network from disk
        
        This restores ADAM's knowledge graph from previous sessions.
        """
        # Check if saved network exists
        graph_file = self.network_path / "memory_graph.gpickle"
        if graph_file.exists():
            # Load the graph structure
            self.memory_graph = nx.read_gpickle(graph_file)
            
            # Load indices
            indices_file = self.network_path / "indices.json"
            if indices_file.exists():
                with open(indices_file, 'r') as f:
                    indices_data = json.load(f)
                    # Convert lists back to sets
                    self.topic_to_memories = {
                        k: set(v) for k, v in indices_data['topic_to_memories'].items()
                    }
                    self.topic_to_threads = defaultdict(list, indices_data['topic_to_threads'])
            
            # Load threads
            threads_file = self.network_path / "threads.json"
            if threads_file.exists():
                with open(threads_file, 'r') as f:
                    threads_data = json.load(f)
                    for tid, thread_dict in threads_data.items():
                        # Reconstruct thread objects
                        thread = ConversationThread(
                            thread_id=thread_dict['thread_id'],
                            primary_topic=thread_dict['primary_topic'],
                            subtopics=thread_dict['subtopics'],
                            conversation_ids=thread_dict['conversation_ids'],
                            memory_ids=thread_dict['memory_ids'],
                            last_updated=datetime.fromisoformat(thread_dict['last_updated']),
                            total_interactions=thread_dict['total_interactions'],
                            evolution_summary=thread_dict.get('evolution_summary')
                        )
                        self.threads[tid] = thread
