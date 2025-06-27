#!/usr/bin/env python3
"""
Example usage of the conversation system with memory network

Shows how ADAM tracks conversations and builds connected memories.

PURPOSE:
========
This example demonstrates real-world usage patterns of ADAM's
conversation and memory systems. It serves as both:
1. **Documentation**: Shows developers how to integrate these systems
2. **Reference**: Provides copy-paste examples for common tasks

KEY CONCEPTS DEMONSTRATED:
- Starting and ending conversation sessions
- Recording exchanges with memory decisions
- Continuing conversations across time
- Searching conversation history
- Generating analytics
- Integrating conversation and memory systems

RUNNING THIS EXAMPLE:
```bash
python examples/conversation_usage.py
```

This will create demo conversations in ./demo_conversations/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam import ConversationSystem, ConversationAwareMemorySystem


def demonstrate_conversation_flow():
    """Demonstrate a typical conversation flow with ADAM
    
    This function shows a complete multi-day conversation workflow:
    1. User encounters a problem (Day 1)
    2. ADAM provides initial solutions
    3. User returns with updates (Day 2)
    4. ADAM provides additional help
    5. Problem is resolved
    
    This pattern mirrors real debugging workflows where complex
    issues require multiple iterations to solve.
    """
    
    print("=== ADAM Conversation System Demo ===\n")
    
    # Initialize conversation system with dedicated storage
    # In production, this would be a persistent path
    conv_system = ConversationSystem(storage_path="./demo_conversations")
    
    # Day 1: User starts discussing a problem
    # This simulates the beginning of a debugging journey
    print("Day 1: Initial Problem Discussion")
    print("-" * 40)
    
    # Start a named session for easy identification
    # Good practice: Use descriptive titles for sessions
    session1 = conv_system.start_session(title="DBT Performance Issues")
    print(f"Started session: {session1}")
    
    # First exchange - User describes the problem
    # This exchange is memory-worthy because it contains:
    # 1. A specific problem description
    # 2. Technical details (CTEs, joins)
    # 3. Performance metrics (45 minutes)
    exchange1 = conv_system.record_exchange(
        query="My dbt model is taking 45 minutes to run. It has multiple CTEs and joins.",
        response="I can help optimize your dbt model. The long runtime suggests inefficient queries. Let me analyze the common causes:\n1. Unoptimized CTEs\n2. Missing indexes\n3. Inefficient joins",
        topics=["dbt", "performance", "optimization", "CTE"],
        stored_in_memory=True,      # This is valuable problem-solution knowledge
        memory_id="mem_001"         # ID from memory system
    )
    print(f"Recorded exchange: {exchange1}")
    
    # Second exchange - User provides more details, ADAM gives specific solution
    # This is a high-value exchange with concrete optimization strategies
    exchange2 = conv_system.record_exchange(
        query="Here's my model code with 5 CTEs doing aggregations",
        response="I see the issue. Your CTEs are scanning the full table multiple times. Try:\n1. Materialize intermediate CTEs as views\n2. Add WHERE clauses early\n3. Use incremental models",
        topics=["dbt", "CTE", "materialization"],
        stored_in_memory=True,      # Specific optimization techniques
        memory_id="mem_002"
    )
    
    # User tries solution - transitional exchange
    # Not every exchange needs to be stored in memory
    # This is a simple follow-up that doesn't contain new knowledge
    exchange3 = conv_system.record_exchange(
        query="I materialized the CTEs but it's still slow",
        response="Let's check if you have proper indexes. Run: EXPLAIN ANALYZE on your query",
        topics=["dbt", "indexes", "performance"],
        stored_in_memory=False  # Transitional exchanges often aren't memory-worthy
    )
    
    # End day 1 session
    # Always end sessions when the conversation naturally concludes
    # This helps with analytics and session boundaries
    conv_system.end_session()
    
    # Get session summary for reporting
    # This shows what was accomplished in the conversation
    print(f"\nSession 1 Summary:")
    summary1 = conv_system.get_session_summary(session1)
    print(f"- Duration: {summary1['duration_minutes']:.1f} minutes")
    print(f"- Exchanges: {summary1['total_exchanges']}")
    print(f"- Memories created: {summary1['memories_created']}")
    print(f"- Topics: {', '.join(summary1['topics'])}")
    
    print("\n" + "="*50 + "\n")
    
    # Day 2: User continues the conversation
    # This demonstrates ADAM's ability to maintain context across time
    print("Day 2: Continuing the Discussion")
    print("-" * 40)
    
    # Continue previous conversation about dbt
    # The system finds the most recent dbt session and links to it
    parent_id, session2 = conv_system.continue_conversation("dbt")
    print(f"Continuing from session: {parent_id}")
    print(f"New session: {session2}")
    
    # User provides update - Problem solved!
    # This is a valuable resolution that should definitely be stored
    exchange4 = conv_system.record_exchange(
        query="I found the issue! No indexes on the join columns. Added them and query dropped to 5 minutes!",
        response="Excellent! That's a 90% performance improvement. For future dbt models:\n1. Always check indexes on join/filter columns\n2. Use dbt's pre-hook to create indexes\n3. Document index requirements in schema.yml",
        topics=["dbt", "indexes", "solution", "best-practices"],
        stored_in_memory=True,      # Solutions and best practices are always valuable
        memory_id="mem_003"
    )
    
    conv_system.end_session()
    
    print("\n" + "="*50 + "\n")
    
    # Search for all DBT-related sessions
    # This demonstrates how users can find previous discussions
    print("Searching for DBT-related conversations:")
    print("-" * 40)
    
    # Search for dbt sessions in the last 30 days
    # Results are ordered newest first for convenience
    dbt_sessions = conv_system.find_recent_sessions_by_topic("dbt", days_back=30)
    for session in dbt_sessions:
        print(f"\n- {session.title}")
        print(f"  Started: {session.start_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Exchanges: {session.total_exchanges}")
        print(f"  Topics: {', '.join(list(session.topics)[:3])}")
    
    # Get analytics
    # Analytics help understand usage patterns and value delivery
    print("\n" + "="*50 + "\n")
    print("Conversation Analytics (Last 30 days):")
    print("-" * 40)
    
    analytics = conv_system.get_analytics(days=30)
    
    # Display key metrics
    print(f"Total sessions: {analytics['total_sessions']}")
    print(f"Total exchanges: {analytics['total_exchanges']}")
    print(f"Memories created: {analytics['total_memories']}")
    print(f"Memory storage rate: {analytics['memory_storage_rate']}")
    
    # Show popular topics
    print(f"\nTop topics:")
    for topic, count in analytics['top_topics'][:5]:
        print(f"  - {topic}: {count} sessions")


def demonstrate_conversation_aware_memory():
    """Show how conversation system integrates with memory network
    
    This demonstrates the ConversationAwareMemorySystem which:
    1. Decides what to store in memory
    2. Links memories to conversations
    3. Enables context-aware responses
    
    In production, this would use the actual memory network
    instead of our mock implementation.
    """
    
    print("\n\n=== Conversation-Aware Memory Demo ===\n")
    
    # Mock memory system for demo purposes
    # In production, this would be the actual ADAMMemoryAdvanced system
    class MockMemorySystem:
        """Simplified memory system for demonstration"""
        def remember_if_worthy(self, **kwargs):
            """Mock memory storage - returns a fake memory ID"""
            return f"mem_{kwargs.get('query', '')[:10]}"
        
        def search(self, query, n_results=5):
            """Mock memory search - returns fake results"""
            return [{"id": f"mem_{i}", "score": 0.9-i*0.1} for i in range(3)]
    
    # Initialize conversation-aware memory
    # This wraps the base memory system with conversation tracking
    base_memory = MockMemorySystem()
    cam_system = ConversationAwareMemorySystem(base_memory)
    
    # Process interactions
    print("Processing user interactions:")
    print("-" * 40)
    
    # Interaction 1: Complex problem (memory-worthy)
    # This demonstrates a high-value technical exchange
    exchange_id1, memory_id1 = cam_system.process_interaction(
        query="How do I optimize a slow dbt model with window functions?",
        response="To optimize dbt models with window functions:\n1. Partition wisely\n2. Order only when needed\n3. Consider materialization",
        topics=["dbt", "window-functions", "optimization"],
        generation_cost=0.02,  # High cost indicates complex/valuable response
        model_used="gpt-4"
    )
    print(f"Exchange {exchange_id1} -> Memory: {memory_id1}")
    
    # Interaction 2: Simple query (not memory worthy)
    # This demonstrates the filtering - not everything is stored
    exchange_id2, memory_id2 = cam_system.process_interaction(
        query="What time is it?",
        response="I don't have access to real-time data.",
        topics=["time"],
        generation_cost=0.001,  # Low cost, simple response
        model_used="mistral"
    )
    print(f"Exchange {exchange_id2} -> Memory: {memory_id2}")
    
    # Get current context
    # This shows what information is available for response generation
    print("\nCurrent Context:")
    context = cam_system.get_current_context()
    print(f"Session ID: {context['session_id']}")
    print(f"Topics: {context.get('session_topics', [])}")
    
    # Continue previous conversation
    # This demonstrates how ADAM can resume discussions with full context
    print("\n" + "="*50 + "\n")
    print("Continuing Previous Conversation:")
    print("-" * 40)
    
    # The system will:
    # 1. Find previous dbt conversations
    # 2. Generate a recap
    # 3. Identify relevant memories to load
    recap, memory_ids = cam_system.continue_conversation("dbt")
    print(f"Recap: {recap}")
    print(f"Relevant memories to load: {memory_ids}")


if __name__ == "__main__":
    # Run demonstrations
    # These examples show typical usage patterns that developers
    # can adapt for their own applications
    
    # Demo 1: Pure conversation tracking
    demonstrate_conversation_flow()
    
    # Demo 2: Integrated conversation + memory
    demonstrate_conversation_aware_memory()
    
    print("\n\nDemo completed! Check ./demo_conversations for persisted data.")
    print("\nNOTE: In production, use proper paths and error handling.")
    print("This demo creates files in the current directory for inspection.")