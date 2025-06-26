#!/usr/bin/env python3
"""
Example usage of the conversation system with memory network
Shows how ADAM tracks conversations and builds connected memories
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam import ConversationSystem, ConversationAwareMemorySystem


def demonstrate_conversation_flow():
    """Demonstrate a typical conversation flow with ADAM"""
    
    print("=== ADAM Conversation System Demo ===\n")
    
    # Initialize conversation system
    conv_system = ConversationSystem(storage_path="./demo_conversations")
    
    # Day 1: User starts discussing a problem
    print("Day 1: Initial Problem Discussion")
    print("-" * 40)
    
    session1 = conv_system.start_session(title="DBT Performance Issues")
    print(f"Started session: {session1}")
    
    # First exchange
    exchange1 = conv_system.record_exchange(
        query="My dbt model is taking 45 minutes to run. It has multiple CTEs and joins.",
        response="I can help optimize your dbt model. The long runtime suggests inefficient queries. Let me analyze the common causes:\n1. Unoptimized CTEs\n2. Missing indexes\n3. Inefficient joins",
        topics=["dbt", "performance", "optimization", "CTE"],
        stored_in_memory=True,
        memory_id="mem_001"
    )
    print(f"Recorded exchange: {exchange1}")
    
    # Second exchange
    exchange2 = conv_system.record_exchange(
        query="Here's my model code with 5 CTEs doing aggregations",
        response="I see the issue. Your CTEs are scanning the full table multiple times. Try:\n1. Materialize intermediate CTEs as views\n2. Add WHERE clauses early\n3. Use incremental models",
        topics=["dbt", "CTE", "materialization"],
        stored_in_memory=True,
        memory_id="mem_002"
    )
    
    # User tries solution
    exchange3 = conv_system.record_exchange(
        query="I materialized the CTEs but it's still slow",
        response="Let's check if you have proper indexes. Run: EXPLAIN ANALYZE on your query",
        topics=["dbt", "indexes", "performance"],
        stored_in_memory=False  # Not storing this transitional exchange
    )
    
    # End day 1 session
    conv_system.end_session()
    print(f"\nSession 1 Summary:")
    summary1 = conv_system.get_session_summary(session1)
    print(f"- Duration: {summary1['duration_minutes']:.1f} minutes")
    print(f"- Exchanges: {summary1['total_exchanges']}")
    print(f"- Memories created: {summary1['memories_created']}")
    print(f"- Topics: {', '.join(summary1['topics'])}")
    
    print("\n" + "="*50 + "\n")
    
    # Day 2: User continues the conversation
    print("Day 2: Continuing the Discussion")
    print("-" * 40)
    
    # Continue previous conversation
    parent_id, session2 = conv_system.continue_conversation("dbt")
    print(f"Continuing from session: {parent_id}")
    print(f"New session: {session2}")
    
    # User provides update
    exchange4 = conv_system.record_exchange(
        query="I found the issue! No indexes on the join columns. Added them and query dropped to 5 minutes!",
        response="Excellent! That's a 90% performance improvement. For future dbt models:\n1. Always check indexes on join/filter columns\n2. Use dbt's pre-hook to create indexes\n3. Document index requirements in schema.yml",
        topics=["dbt", "indexes", "solution", "best-practices"],
        stored_in_memory=True,
        memory_id="mem_003"
    )
    
    conv_system.end_session()
    
    print("\n" + "="*50 + "\n")
    
    # Search for all DBT-related sessions
    print("Searching for DBT-related conversations:")
    print("-" * 40)
    
    dbt_sessions = conv_system.find_recent_sessions_by_topic("dbt", days_back=30)
    for session in dbt_sessions:
        print(f"\n- {session.title}")
        print(f"  Started: {session.start_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Exchanges: {session.total_exchanges}")
        print(f"  Topics: {', '.join(list(session.topics)[:3])}")
    
    # Get analytics
    print("\n" + "="*50 + "\n")
    print("Conversation Analytics (Last 30 days):")
    print("-" * 40)
    
    analytics = conv_system.get_analytics(days=30)
    print(f"Total sessions: {analytics['total_sessions']}")
    print(f"Total exchanges: {analytics['total_exchanges']}")
    print(f"Memories created: {analytics['total_memories']}")
    print(f"Memory storage rate: {analytics['memory_storage_rate']}")
    print(f"\nTop topics:")
    for topic, count in analytics['top_topics'][:5]:
        print(f"  - {topic}: {count} sessions")


def demonstrate_conversation_aware_memory():
    """Show how conversation system integrates with memory network"""
    
    print("\n\n=== Conversation-Aware Memory Demo ===\n")
    
    # Mock memory system for demo
    class MockMemorySystem:
        def remember_if_worthy(self, **kwargs):
            return f"mem_{kwargs.get('query', '')[:10]}"
        
        def search(self, query, n_results=5):
            return [{"id": f"mem_{i}", "score": 0.9-i*0.1} for i in range(3)]
    
    # Initialize conversation-aware memory
    base_memory = MockMemorySystem()
    cam_system = ConversationAwareMemorySystem(base_memory)
    
    # Process interactions
    print("Processing user interactions:")
    print("-" * 40)
    
    # Interaction 1: Complex problem
    exchange_id1, memory_id1 = cam_system.process_interaction(
        query="How do I optimize a slow dbt model with window functions?",
        response="To optimize dbt models with window functions:\n1. Partition wisely\n2. Order only when needed\n3. Consider materialization",
        topics=["dbt", "window-functions", "optimization"],
        generation_cost=0.02,  # Expensive query
        model_used="gpt-4"
    )
    print(f"Exchange {exchange_id1} -> Memory: {memory_id1}")
    
    # Interaction 2: Simple query (not memory worthy)
    exchange_id2, memory_id2 = cam_system.process_interaction(
        query="What time is it?",
        response="I don't have access to real-time data.",
        topics=["time"],
        generation_cost=0.001,
        model_used="mistral"
    )
    print(f"Exchange {exchange_id2} -> Memory: {memory_id2}")
    
    # Get current context
    print("\nCurrent Context:")
    context = cam_system.get_current_context()
    print(f"Session ID: {context['session_id']}")
    print(f"Topics: {context.get('session_topics', [])}")
    
    # Continue previous conversation
    print("\n" + "="*50 + "\n")
    print("Continuing Previous Conversation:")
    print("-" * 40)
    
    recap, memory_ids = cam_system.continue_conversation("dbt")
    print(f"Recap: {recap}")
    print(f"Relevant memories to load: {memory_ids}")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_conversation_flow()
    demonstrate_conversation_aware_memory()
    
    print("\n\nDemo completed! Check ./demo_conversations for persisted data.")