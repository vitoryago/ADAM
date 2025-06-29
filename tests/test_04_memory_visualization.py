#!/usr/bin/env python3
"""
Test 4: Memory Network Visualization
Tests the visual representation of ADAM's memory network
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.adam.memory_network import MemoryNetworkSystem
from src.adam.conversation_system import ConversationSystem
import tempfile
import shutil
import numpy as np
from datetime import datetime, timedelta


class MockMemory:
    """Mock memory system for testing"""
    def __init__(self):
        self.memory_count = 0
        
    def remember_if_worthy(self, **kwargs):
        self.memory_count += 1
        return f"mem_{kwargs.get('query', '')[:10].replace(' ', '_')}_{self.memory_count}"
    
    def get_embedding(self, text):
        # Generate consistent embeddings based on text content
        np.random.seed(sum(ord(c) for c in text) % 1000)
        return np.random.rand(384)


def test_memory_visualization():
    """Test memory network visualization with different scenarios"""
    print("=== Memory Network Visualization Test ===\n")
    
    # Create temporary directory for test data
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize systems
        conv_system = ConversationSystem(temp_dir)
        mock_memory = MockMemory()
        memory_network = MemoryNetworkSystem(mock_memory, conv_system)
        
        # Test 1: Simple topic network
        print("1. Creating simple topic network...")
        simple_memories = [
            (["SQL", "optimization"], "How to optimize SQL queries?", "Use indexes and EXPLAIN"),
            (["SQL", "joins"], "Best practices for SQL joins?", "Start with smaller tables"),
            (["SQL", "indexes"], "When to use indexes?", "On columns used in WHERE and JOIN"),
        ]
        
        memory_ids = []
        for topics, query, response in simple_memories:
            mem_id = memory_network.add_memory_with_references(
                query=query,
                response=response,
                memory_type="explanation",
                topics=topics,
                auto_save=False
            )
            memory_ids.append(mem_id)
        
        # Create visualization
        fig = memory_network.visualize_memory_network(topic="SQL", show_decay=True, highlight_patterns=True)
        fig.savefig("test_simple_network.png", dpi=150, bbox_inches='tight')
        print("✓ Saved simple network to test_simple_network.png")
        
        # Test 2: Complex interconnected network
        print("\n2. Creating complex interconnected network...")
        
        # Add memories that reference each other
        complex_memories = [
            (["dbt", "SQL", "performance"], "dbt model running slow", "Check materialization strategy"),
            (["dbt", "CTE"], "Using CTEs in dbt", "CTEs can impact performance"),
            (["dbt", "optimization"], "Optimize dbt models", "Use incremental models"),
            (["SQL", "CTE", "performance"], "CTE performance issues", "CTEs are not indexed"),
            (["debugging", "SQL"], "Debug slow queries", "Use query profiler"),
        ]
        
        for i, (topics, query, response) in enumerate(complex_memories):
            # Add references to previous memories
            potential_refs = memory_ids[:max(0, i-1)] if i > 0 else []
            
            mem_id = memory_network.add_memory_with_references(
                query=query,
                response=response,
                memory_type="explanation",
                topics=topics,
                potential_references=potential_refs,
                auto_save=False
            )
            memory_ids.append(mem_id)
        
        # Visualize the complex network
        fig = memory_network.visualize_memory_network(show_decay=True, highlight_patterns=True)
        fig.savefig("test_complex_network.png", dpi=150, bbox_inches='tight')
        print("✓ Saved complex network to test_complex_network.png")
        
        # Test 3: Memory types visualization
        print("\n3. Testing different memory types...")
        
        typed_memories = [
            (["Python", "error"], "ImportError in Python", "Check module path", "error_solution"),
            (["Python", "code"], "Write async function", "async def example()...", "code_implementation"),
            (["Python", "guide"], "Setup Python project", "1. Create venv 2. Install deps", "how_to_guide"),
            (["Python", "concept"], "What is a decorator?", "Function that wraps another", "explanation"),
        ]
        
        for topics, query, response, mem_type in typed_memories:
            memory_network.add_memory_with_references(
                query=query,
                response=response,
                memory_type=mem_type,
                topics=topics,
                auto_save=False
            )
        
        fig = memory_network.visualize_memory_network(topic="Python", show_decay=True, highlight_patterns=True)
        fig.savefig("test_typed_network.png", dpi=150, bbox_inches='tight')
        print("✓ Saved typed network to test_typed_network.png")
        
        # Test 4: Memory decay visualization
        print("\n4. Testing memory decay visualization...")
        
        # Add memories with different ages
        base_time = datetime.now()
        decay_memories = [
            (["decay", "test"], "Fresh memory", "Just created", 0),
            (["decay", "test"], "Week old memory", "One week ago", 7),
            (["decay", "test"], "Month old memory", "One month ago", 30),
            (["decay", "test"], "Old memory", "Six months ago", 180),
        ]
        
        for topics, query, response, days_old in decay_memories:
            # Create memory with specific timestamp
            mem_id = memory_network.add_memory_with_references(
                query=query,
                response=response,
                memory_type="test",
                topics=topics,
                auto_save=False
            )
            
            # Manually set timestamp for testing
            if mem_id in memory_network.memory_graph.nodes:
                node = memory_network.memory_graph.nodes[mem_id]['data']
                node.timestamp = base_time - timedelta(days=days_old)
                
                # Simulate access patterns
                if days_old < 30:
                    node.access_count = 10 - days_old // 3
                    node.last_accessed = base_time - timedelta(days=days_old // 2)
        
        fig = memory_network.visualize_memory_network(topic="decay", show_decay=True, highlight_patterns=True)
        fig.savefig("test_decay_network.png", dpi=150, bbox_inches='tight')
        print("✓ Saved decay network to test_decay_network.png")
        
        # Test 5: Large network performance
        print("\n5. Testing visualization performance with large network...")
        
        import time
        start_time = time.time()
        
        # Add many memories
        for i in range(50):
            topics = ["performance", f"subtopic_{i % 5}"]
            memory_network.add_memory_with_references(
                query=f"Query {i}",
                response=f"Response {i}",
                memory_type="test",
                topics=topics,
                auto_save=False
            )
        
        # Time the visualization
        viz_start = time.time()
        fig = memory_network.visualize_memory_network(topic="performance")
        fig.savefig("test_large_network.png", dpi=150, bbox_inches='tight')
        viz_time = time.time() - viz_start
        
        print(f"✓ Created 50-node network in {time.time() - start_time:.2f}s")
        print(f"✓ Visualization rendered in {viz_time:.2f}s")
        
        # Print network statistics
        print("\n=== Network Statistics ===")
        print(f"Total memories: {memory_network.memory_graph.number_of_nodes()}")
        print(f"Total connections: {memory_network.memory_graph.number_of_edges()}")
        print(f"Topics tracked: {len(memory_network.topic_to_memories)}")
        
        topics_with_counts = [
            (topic, len(memories)) 
            for topic, memories in memory_network.topic_to_memories.items()
        ]
        topics_with_counts.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop topics:")
        for topic, count in topics_with_counts[:5]:
            print(f"  - {topic}: {count} memories")
        
        print("\n✅ All visualization tests completed successfully!")
        print("\nGenerated visualizations:")
        print("  - test_simple_network.png: Basic SQL topic network")
        print("  - test_complex_network.png: Interconnected multi-topic network")
        print("  - test_typed_network.png: Different memory types with colors")
        print("  - test_decay_network.png: Memory decay visualization")
        print("  - test_large_network.png: Performance test with 50+ nodes")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_memory_visualization()