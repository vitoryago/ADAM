#!/usr/bin/env python3
"""
Test 6: Memory Evolution
Tests how memories evolve over time through decay, reinforcement, and pattern emergence
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import shutil

from src.adam import ConversationSystem, MemoryNetworkSystem, ConversationAwareMemorySystem


class MockMemory:
    """Mock memory system with embedding support"""
    def __init__(self):
        self.memory_count = 0
        self.stored_memories = {}
        
    def remember_if_worthy(self, **kwargs):
        self.memory_count += 1
        mem_id = f"mem_{kwargs.get('query', '')[:15].replace(' ', '_')}_{self.memory_count}"
        self.stored_memories[mem_id] = kwargs
        return mem_id
    
    def get_embedding(self, text):
        # Generate semantic embeddings based on content
        words = text.lower().split()
        embedding = np.zeros(384)
        
        # Simple semantic encoding
        topic_vectors = {
            'sql': 0, 'query': 1, 'database': 2, 'index': 3,
            'dbt': 10, 'model': 11, 'transformation': 12,
            'python': 20, 'code': 21, 'function': 22,
            'error': 30, 'bug': 31, 'fix': 32, 'solution': 33,
            'performance': 40, 'optimize': 41, 'slow': 42,
        }
        
        for word in words:
            if word in topic_vectors:
                idx = topic_vectors[word]
                embedding[idx] = 1.0
                # Add some noise for variation
                embedding[idx:idx+5] += np.random.normal(0, 0.1, min(5, 384-idx))
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def search(self, query, n_results=5):
        """Mock search that returns similar memories"""
        results = []
        query_embedding = self.get_embedding(query)
        
        for mem_id, mem_data in self.stored_memories.items():
            if 'response' in mem_data:
                mem_embedding = self.get_embedding(mem_data['response'])
                similarity = np.dot(query_embedding, mem_embedding)
                if similarity > 0.3:
                    results.append({
                        'id': mem_id,
                        'query': mem_data.get('query', ''),
                        'response': mem_data.get('response', ''),
                        'distance': 1 - similarity
                    })
        
        # Sort by similarity and return top n
        results.sort(key=lambda x: x['distance'])
        return results[:n_results]


def simulate_memory_evolution(memory_network, days=30, interactions_per_day=5):
    """Simulate memory evolution over time"""
    evolution_data = {
        'day': [],
        'total_memories': [],
        'active_memories': [],
        'decayed_memories': [],
        'avg_access_count': [],
        'avg_references': [],
        'topics_covered': []
    }
    
    # Common queries that might recur
    query_templates = [
        ("How to optimize {topic} query?", ["SQL", "optimization", "performance"]),
        ("Debug {topic} error", ["debugging", "error", "solution"]),
        ("Best practices for {topic}", ["best-practices", "guide"]),
        ("Implement {topic} feature", ["implementation", "code"]),
        ("Fix {topic} performance issue", ["performance", "debugging"]),
    ]
    
    topics_pool = ["SQL", "dbt", "Python", "database", "API", "testing"]
    
    print(f"Simulating {days} days of memory evolution...")
    
    for day in range(days):
        current_date = datetime.now() - timedelta(days=days-day)
        
        # Simulate daily interactions
        for interaction in range(interactions_per_day):
            # Sometimes reference old problems
            if day > 5 and np.random.rand() < 0.3:
                # Follow up on previous topic
                if memory_network.topic_to_memories:
                    topic = np.random.choice(list(memory_network.topic_to_memories.keys()))
                    query = f"Follow up on {topic} issue from before"
                    topics = [topic, "follow-up"]
                else:
                    continue
            else:
                # New query
                template, base_topics = query_templates[np.random.randint(len(query_templates))]
                topic = np.random.choice(topics_pool)
                query = template.format(topic=topic)
                topics = base_topics + [topic]
            
            response = f"Here's the solution for {query}: [detailed explanation]"
            
            # Add memory with some probability
            if np.random.rand() < 0.7:  # 70% chance of being worthy
                mem_id = memory_network.add_memory_with_references(
                    query=query,
                    response=response,
                    memory_type=np.random.choice(["explanation", "solution", "guide"]),
                    topics=topics,
                    auto_save=False
                )
                
                # Simulate access patterns
                if mem_id in memory_network.memory_graph.nodes:
                    node = memory_network.memory_graph.nodes[mem_id]['data']
                    node.timestamp = current_date
                    
                    # Popular topics get accessed more
                    if any(t in ["SQL", "performance", "error"] for t in topics):
                        node.access_count = np.random.randint(5, 15)
                    else:
                        node.access_count = np.random.randint(0, 5)
                    
                    if node.access_count > 0:
                        node.last_accessed = current_date + timedelta(
                            days=np.random.randint(0, min(3, days-day))
                        )
        
        # Calculate metrics for this day
        total_mems = memory_network.memory_graph.number_of_nodes()
        
        # Count active vs decayed memories
        active_count = 0
        decay_count = 0
        total_access = 0
        total_refs = 0
        
        for node_id in memory_network.memory_graph.nodes():
            node = memory_network.memory_graph.nodes[node_id]['data']
            decay = memory_network._calculate_memory_decay(
                node.timestamp,
                node.access_count,
                node.last_accessed
            )
            
            if decay > 0.1:
                active_count += 1
                total_access += node.access_count
                total_refs += len(node.references)
            else:
                decay_count += 1
        
        # Record evolution data
        evolution_data['day'].append(day)
        evolution_data['total_memories'].append(total_mems)
        evolution_data['active_memories'].append(active_count)
        evolution_data['decayed_memories'].append(decay_count)
        evolution_data['avg_access_count'].append(
            total_access / active_count if active_count > 0 else 0
        )
        evolution_data['avg_references'].append(
            total_refs / active_count if active_count > 0 else 0
        )
        evolution_data['topics_covered'].append(len(memory_network.topic_to_memories))
    
    return evolution_data


def test_memory_evolution():
    """Test memory evolution patterns"""
    print("=== Memory Evolution Test ===\n")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize systems
        conv_system = ConversationSystem(temp_dir)
        mock_memory = MockMemory()
        memory_network = MemoryNetworkSystem(mock_memory, conv_system)
        cam_system = ConversationAwareMemorySystem(mock_memory)
        cam_system.memory_network = memory_network
        
        # Test 1: Short-term evolution (7 days)
        print("1. Short-term Memory Evolution (7 days)")
        print("-" * 40)
        
        evolution_7d = simulate_memory_evolution(memory_network, days=7, interactions_per_day=10)
        
        print(f"After 7 days:")
        print(f"  Total memories: {evolution_7d['total_memories'][-1]}")
        print(f"  Active memories: {evolution_7d['active_memories'][-1]}")
        print(f"  Topics covered: {evolution_7d['topics_covered'][-1]}")
        print()
        
        # Test 2: Medium-term evolution (30 days)
        print("2. Medium-term Memory Evolution (30 days)")
        print("-" * 40)
        
        # Reset and simulate longer period
        memory_network = MemoryNetworkSystem(mock_memory, conv_system)
        evolution_30d = simulate_memory_evolution(memory_network, days=30, interactions_per_day=5)
        
        print(f"After 30 days:")
        print(f"  Total memories: {evolution_30d['total_memories'][-1]}")
        print(f"  Active memories: {evolution_30d['active_memories'][-1]}")
        print(f"  Decayed memories: {evolution_30d['decayed_memories'][-1]}")
        print(f"  Retention rate: {evolution_30d['active_memories'][-1]/evolution_30d['total_memories'][-1]*100:.1f}%")
        print()
        
        # Test 3: Pattern emergence
        print("3. Pattern Emergence Analysis")
        print("-" * 40)
        
        # Analyze conversation threads
        threads_by_topic = {}
        for thread_id, thread in memory_network.threads.items():
            topic = thread.primary_topic
            if topic not in threads_by_topic:
                threads_by_topic[topic] = []
            threads_by_topic[topic].append(thread)
        
        print(f"Conversation threads formed: {len(memory_network.threads)}")
        print("Top thread topics:")
        for topic, threads in sorted(threads_by_topic.items(), 
                                   key=lambda x: len(x[1]), 
                                   reverse=True)[:5]:
            avg_length = np.mean([len(t.memory_ids) for t in threads])
            print(f"  - {topic}: {len(threads)} threads, avg length: {avg_length:.1f}")
        print()
        
        # Test 4: Memory reinforcement patterns
        print("4. Memory Reinforcement Patterns")
        print("-" * 40)
        
        # Find most accessed memories
        access_counts = []
        for node_id in memory_network.memory_graph.nodes():
            node = memory_network.memory_graph.nodes[node_id]['data']
            access_counts.append((node_id, node.access_count, node.topics))
        
        access_counts.sort(key=lambda x: x[1], reverse=True)
        
        print("Most accessed memories:")
        for mem_id, count, topics in access_counts[:5]:
            print(f"  - {mem_id}: {count} accesses, topics: {', '.join(topics[:3])}")
        print()
        
        # Test 5: Reference network growth
        print("5. Reference Network Analysis")
        print("-" * 40)
        
        # Analyze reference patterns
        reference_stats = {
            'no_refs': 0,
            'few_refs': 0,  # 1-3
            'many_refs': 0,  # 4+
            'highly_referenced': 0  # referenced by 3+ others
        }
        
        for node_id in memory_network.memory_graph.nodes():
            node = memory_network.memory_graph.nodes[node_id]['data']
            ref_count = len(node.references)
            referenced_by_count = len(node.referenced_by)
            
            if ref_count == 0:
                reference_stats['no_refs'] += 1
            elif ref_count <= 3:
                reference_stats['few_refs'] += 1
            else:
                reference_stats['many_refs'] += 1
                
            if referenced_by_count >= 3:
                reference_stats['highly_referenced'] += 1
        
        total = sum(reference_stats.values()) - reference_stats['highly_referenced']
        print(f"Reference patterns:")
        print(f"  No references: {reference_stats['no_refs']} ({reference_stats['no_refs']/total*100:.1f}%)")
        print(f"  Few references (1-3): {reference_stats['few_refs']} ({reference_stats['few_refs']/total*100:.1f}%)")
        print(f"  Many references (4+): {reference_stats['many_refs']} ({reference_stats['many_refs']/total*100:.1f}%)")
        print(f"  Highly referenced: {reference_stats['highly_referenced']} memories")
        print()
        
        # Generate evolution plots
        print("6. Generating Evolution Visualizations")
        print("-" * 40)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Memory growth over time
        ax1.plot(evolution_30d['day'], evolution_30d['total_memories'], 'b-', label='Total')
        ax1.plot(evolution_30d['day'], evolution_30d['active_memories'], 'g-', label='Active')
        ax1.plot(evolution_30d['day'], evolution_30d['decayed_memories'], 'r-', label='Decayed')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Memory Count')
        ax1.set_title('Memory Population Over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Access patterns
        ax2.plot(evolution_30d['day'], evolution_30d['avg_access_count'], 'purple')
        ax2.set_xlabel('Days')
        ax2.set_ylabel('Average Access Count')
        ax2.set_title('Memory Access Patterns')
        ax2.grid(True)
        
        # Topic growth
        ax3.plot(evolution_30d['day'], evolution_30d['topics_covered'], 'orange')
        ax3.set_xlabel('Days')
        ax3.set_ylabel('Unique Topics')
        ax3.set_title('Topic Diversity Growth')
        ax3.grid(True)
        
        # Reference network density
        ax4.plot(evolution_30d['day'], evolution_30d['avg_references'], 'brown')
        ax4.set_xlabel('Days')
        ax4.set_ylabel('Average References per Memory')
        ax4.set_title('Reference Network Density')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('test_memory_evolution.png', dpi=150)
        print("✓ Saved evolution plots to test_memory_evolution.png")
        
        # Test 6: Semantic clustering
        print("\n7. Semantic Memory Clustering")
        print("-" * 40)
        
        # Find semantic clusters
        clusters = {}
        for node_id in list(memory_network.memory_graph.nodes())[:50]:  # Sample for performance
            node = memory_network.memory_graph.nodes[node_id]['data']
            if node.embedding is not None:
                # Find similar memories
                similar = []
                for other_id in memory_network.memory_graph.nodes():
                    if other_id != node_id:
                        other = memory_network.memory_graph.nodes[other_id]['data']
                        if other.embedding is not None:
                            similarity = np.dot(node.embedding, other.embedding)
                            if similarity > 0.7:
                                similar.append(other_id)
                
                if len(similar) >= 2:
                    # Found a cluster
                    cluster_key = tuple(sorted(node.topics))
                    if cluster_key not in clusters:
                        clusters[cluster_key] = set()
                    clusters[cluster_key].add(node_id)
                    clusters[cluster_key].update(similar)
        
        print(f"Found {len(clusters)} semantic clusters")
        for topics, members in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            print(f"  - {', '.join(topics)}: {len(members)} memories")
        
        print("\n✅ Memory evolution test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_memory_evolution()