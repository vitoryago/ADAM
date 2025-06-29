#!/usr/bin/env python3
"""
Test 5: Performance & Scalability
Tests ADAM's performance with large-scale memory networks
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import tempfile
import shutil
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from src.adam import ConversationSystem, MemoryNetworkSystem


class MockMemory:
    """Mock memory system optimized for performance testing"""
    def __init__(self):
        self.memory_count = 0
        self.embeddings_cache = {}
        
    def remember_if_worthy(self, **kwargs):
        self.memory_count += 1
        return f"mem_{self.memory_count:06d}"
    
    def get_embedding(self, text):
        # Cache embeddings for performance
        if text not in self.embeddings_cache:
            np.random.seed(hash(text) % 10000)
            self.embeddings_cache[text] = np.random.rand(384)
        return self.embeddings_cache[text]


def benchmark_operation(operation_name, func, iterations=1):
    """Benchmark a single operation"""
    start_time = time.time()
    result = None
    
    for _ in range(iterations):
        result = func()
    
    elapsed = time.time() - start_time
    avg_time = elapsed / iterations
    
    print(f"{operation_name}:")
    print(f"  Total: {elapsed:.3f}s")
    print(f"  Average: {avg_time*1000:.1f}ms")
    print(f"  Per second: {iterations/elapsed:.1f}")
    
    return avg_time, result


def test_performance_scalability():
    """Test ADAM's performance at various scales"""
    print("=== Performance & Scalability Test ===\n")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize systems
        conv_system = ConversationSystem(temp_dir)
        mock_memory = MockMemory()
        memory_network = MemoryNetworkSystem(mock_memory, conv_system)
        
        # Test 1: Memory Addition Performance
        print("1. Memory Addition Performance")
        print("-" * 40)
        
        memory_counts = [10, 50, 100, 500, 1000]
        addition_times = []
        
        for count in memory_counts:
            def add_memories():
                mem_ids = []
                for i in range(count):
                    topics = [f"topic_{i % 10}", f"subtopic_{i % 20}"]
                    mem_id = memory_network.add_memory_with_references(
                        query=f"Query {i} about {topics[0]}",
                        response=f"Response {i} explaining {topics[0]}",
                        memory_type="explanation",
                        topics=topics,
                        auto_save=False
                    )
                    mem_ids.append(mem_id)
                return mem_ids
            
            avg_time, _ = benchmark_operation(
                f"Add {count} memories", 
                add_memories, 
                iterations=1
            )
            addition_times.append(avg_time)
            print()
        
        # Test 2: Reference Resolution Performance
        print("\n2. Reference Resolution Performance")
        print("-" * 40)
        
        # Add interconnected memories
        print("Creating interconnected memory network...")
        base_memories = []
        for i in range(100):
            mem_id = memory_network.add_memory_with_references(
                query=f"Base query {i}",
                response=f"Base response {i}",
                memory_type="explanation",
                topics=[f"ref_topic_{i % 5}"],
                auto_save=False
            )
            base_memories.append(mem_id)
        
        # Add memories with references
        ref_counts = [1, 3, 5, 10]
        reference_times = []
        
        for ref_count in ref_counts:
            def add_with_refs():
                # Select random references
                refs = np.random.choice(base_memories, 
                                      min(ref_count, len(base_memories)), 
                                      replace=False).tolist()
                
                return memory_network.add_memory_with_references(
                    query=f"Query with {ref_count} references",
                    response=f"Response referencing {ref_count} memories",
                    memory_type="analysis",
                    topics=["reference_test"],
                    potential_references=refs,
                    auto_save=False
                )
            
            avg_time, _ = benchmark_operation(
                f"Add memory with {ref_count} references",
                add_with_refs,
                iterations=10
            )
            reference_times.append(avg_time)
            print()
        
        # Test 3: Search Performance
        print("\n3. Search Performance")
        print("-" * 40)
        
        search_sizes = [10, 50, 100, 500, 1000]
        search_times = []
        
        # Clear and rebuild with specific sizes
        for size in search_sizes:
            # Create fresh network for each test
            memory_network = MemoryNetworkSystem(mock_memory, conv_system)
            
            # Add memories
            for i in range(size):
                memory_network.add_memory_with_references(
                    query=f"Search test query {i}",
                    response=f"Search test response {i}",
                    memory_type="test",
                    topics=[f"search_{i % 10}", "performance"],
                    auto_save=False
                )
            
            def search_memories():
                return memory_network.find_similar_patterns(
                    "test query about performance",
                    {"search_5": 0.8}
                )
            
            avg_time, results = benchmark_operation(
                f"Search in {size} memories",
                search_memories,
                iterations=10
            )
            search_times.append(avg_time)
            print(f"  Found: {len(results)} results")
            print()
        
        # Test 4: Memory Decay Calculation
        print("\n4. Memory Decay Performance")
        print("-" * 40)
        
        # Add memories with various ages
        print("Creating aged memory network...")
        aged_memories = []
        for days_old in [0, 7, 30, 90, 180, 365]:
            for i in range(50):
                mem_id = memory_network.add_memory_with_references(
                    query=f"Aged query {days_old}d-{i}",
                    response=f"Aged response",
                    memory_type="test",
                    topics=["decay_test"],
                    auto_save=False
                )
                
                # Set age
                if mem_id in memory_network.memory_graph.nodes:
                    node = memory_network.memory_graph.nodes[mem_id]['data']
                    node.timestamp = datetime.now() - timedelta(days=days_old)
                    node.access_count = max(0, 10 - days_old // 30)
                    aged_memories.append((mem_id, days_old))
        
        def calculate_all_decay():
            decay_scores = []
            for mem_id, _ in aged_memories:
                if mem_id in memory_network.memory_graph.nodes:
                    node = memory_network.memory_graph.nodes[mem_id]['data']
                    decay = memory_network._calculate_memory_decay(
                        node.timestamp,
                        node.access_count,
                        node.last_accessed
                    )
                    decay_scores.append(decay)
            return decay_scores
        
        avg_time, decay_scores = benchmark_operation(
            f"Calculate decay for {len(aged_memories)} memories",
            calculate_all_decay,
            iterations=5
        )
        print(f"  Memories to be removed: {sum(1 for d in decay_scores if d < 0.1)}")
        print()
        
        # Test 5: Graph Operations
        print("\n5. Graph Operation Performance")
        print("-" * 40)
        
        def get_graph_stats():
            stats = {
                'nodes': memory_network.memory_graph.number_of_nodes(),
                'edges': memory_network.memory_graph.number_of_edges(),
                'avg_degree': np.mean([d for n, d in memory_network.memory_graph.degree()]),
                'topics': len(memory_network.topic_to_memories)
            }
            return stats
        
        avg_time, stats = benchmark_operation(
            "Calculate graph statistics",
            get_graph_stats,
            iterations=100
        )
        print(f"  Nodes: {stats['nodes']}")
        print(f"  Edges: {stats['edges']}")
        print(f"  Avg degree: {stats['avg_degree']:.2f}")
        print(f"  Topics: {stats['topics']}")
        print()
        
        # Test 6: Persistence Performance
        print("\n6. Persistence Performance")
        print("-" * 40)
        
        def save_network():
            memory_network._save_network()
        
        def load_network():
            memory_network._load_network()
        
        avg_save_time, _ = benchmark_operation(
            f"Save network ({stats['nodes']} nodes)",
            save_network,
            iterations=3
        )
        
        avg_load_time, _ = benchmark_operation(
            f"Load network ({stats['nodes']} nodes)",
            load_network,
            iterations=3
        )
        print()
        
        # Generate performance plots
        print("\n7. Generating Performance Plots")
        print("-" * 40)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Memory addition scaling
        ax1.plot(memory_counts, [t*1000 for t in addition_times], 'b-o')
        ax1.set_xlabel('Number of Memories')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Memory Addition Performance')
        ax1.grid(True)
        
        # Reference resolution scaling
        ax2.plot(ref_counts, [t*1000 for t in reference_times], 'g-o')
        ax2.set_xlabel('Number of References')
        ax2.set_ylabel('Time per Addition (ms)')
        ax2.set_title('Reference Resolution Performance')
        ax2.grid(True)
        
        # Search performance scaling
        ax3.plot(search_sizes, [t*1000 for t in search_times], 'r-o')
        ax3.set_xlabel('Network Size')
        ax3.set_ylabel('Search Time (ms)')
        ax3.set_title('Search Performance Scaling')
        ax3.grid(True)
        
        # Decay distribution
        ax4.hist(decay_scores, bins=50, edgecolor='black')
        ax4.axvline(x=0.1, color='red', linestyle='--', label='Removal threshold')
        ax4.set_xlabel('Decay Score')
        ax4.set_ylabel('Count')
        ax4.set_title('Memory Decay Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_performance_plots.png', dpi=150)
        print("✓ Saved performance plots to test_performance_plots.png")
        
        # Performance summary
        print("\n=== Performance Summary ===")
        print(f"✓ Memory addition: {addition_times[-1]*1000/memory_counts[-1]:.2f}ms per memory")
        print(f"✓ Reference resolution: {reference_times[-1]*1000:.1f}ms with 10 refs")
        print(f"✓ Search performance: {search_times[-1]*1000:.1f}ms in 1000 memories")
        print(f"✓ Decay calculation: {avg_time*1000/len(aged_memories):.3f}ms per memory")
        print(f"✓ Save performance: {avg_save_time*1000:.1f}ms")
        print(f"✓ Load performance: {avg_load_time*1000:.1f}ms")
        
        # Check if meeting performance targets
        print("\n=== Performance Targets ===")
        targets_met = True
        
        if addition_times[-1]*1000/memory_counts[-1] < 1.0:
            print("✅ Memory addition < 1ms per memory")
        else:
            print("❌ Memory addition exceeds 1ms per memory")
            targets_met = False
            
        if search_times[-1]*1000 < 100:
            print("✅ Search < 100ms in large network")
        else:
            print("❌ Search exceeds 100ms in large network")
            targets_met = False
            
        if avg_save_time < 1.0:
            print("✅ Save < 1s")
        else:
            print("❌ Save exceeds 1s")
            targets_met = False
        
        if targets_met:
            print("\n🎉 All performance targets met!")
        else:
            print("\n⚠️  Some performance targets not met")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_performance_scalability()