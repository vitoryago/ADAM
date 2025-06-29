#!/usr/bin/env python3
"""
Test 7: Advanced Reference Resolution
Tests complex reference patterns, multi-hop traversal, and intelligent linking
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tempfile
import shutil
from datetime import datetime, timedelta

from src.adam import ConversationSystem, MemoryNetworkSystem


class AdvancedMockMemory:
    """Mock memory with advanced semantic understanding"""
    def __init__(self):
        self.memory_count = 0
        self.memories = {}
        
    def remember_if_worthy(self, **kwargs):
        self.memory_count += 1
        mem_id = f"mem_{self.memory_count:04d}"
        self.memories[mem_id] = kwargs
        return mem_id
    
    def get_embedding(self, text):
        """Generate embeddings that capture semantic relationships"""
        text_lower = text.lower()
        embedding = np.zeros(384)
        
        # Define semantic concepts and their vector positions
        concepts = {
            # SQL concepts
            'sql': (0, 10), 'query': (5, 10), 'join': (10, 10),
            'index': (15, 10), 'performance': (20, 10),
            
            # Programming concepts  
            'python': (30, 10), 'function': (35, 10), 'class': (40, 10),
            'async': (45, 10), 'error': (50, 10),
            
            # Data concepts
            'data': (60, 10), 'pipeline': (65, 10), 'etl': (70, 10),
            'transformation': (75, 10), 'dbt': (80, 10),
            
            # Problem-solving concepts
            'debug': (90, 10), 'fix': (95, 10), 'solution': (100, 10),
            'optimize': (105, 10), 'refactor': (110, 10),
            
            # Relationships
            'because': (120, 5), 'therefore': (125, 5), 'however': (130, 5),
            'related': (135, 5), 'similar': (140, 5)
        }
        
        # Encode concepts
        for concept, (start_idx, strength) in concepts.items():
            if concept in text_lower:
                embedding[start_idx:start_idx+5] = strength * (1 + np.random.normal(0, 0.1, 5))
        
        # Add contextual relationships
        if 'error' in text_lower and 'fix' in text_lower:
            embedding[150:155] = 10  # Error-solution pattern
        if 'slow' in text_lower and 'optimize' in text_lower:
            embedding[155:160] = 10  # Performance pattern
        if 'how' in text_lower and '?' in text:
            embedding[160:165] = 5   # Question pattern
            
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def search(self, query, n_results=5):
        """Search with semantic understanding"""
        results = []
        query_embedding = self.get_embedding(query)
        
        for mem_id, memory in self.memories.items():
            mem_text = f"{memory.get('query', '')} {memory.get('response', '')}"
            mem_embedding = self.get_embedding(mem_text)
            
            # Calculate similarity
            similarity = np.dot(query_embedding, mem_embedding)
            
            if similarity > 0.1:  # Low threshold to get more results
                results.append({
                    'id': mem_id,
                    'query': memory.get('query', ''),
                    'response': memory.get('response', ''),
                    'distance': 1 - similarity,
                    'similarity': similarity
                })
        
        results.sort(key=lambda x: x['distance'])
        return results[:n_results]


def create_knowledge_graph_scenario(memory_network):
    """Create a complex knowledge graph with multiple interconnected topics"""
    
    # Layer 1: Foundational concepts
    foundations = [
        ("What is SQL?", "SQL is a language for querying databases", ["SQL", "basics"], "explanation"),
        ("What is Python?", "Python is a programming language", ["Python", "basics"], "explanation"),
        ("What is dbt?", "dbt is a data transformation tool", ["dbt", "basics"], "explanation"),
        ("What are indexes?", "Indexes speed up data retrieval", ["SQL", "indexes", "basics"], "explanation"),
    ]
    
    foundation_ids = []
    for query, response, topics, mem_type in foundations:
        mem_id = memory_network.add_memory_with_references(
            query=query,
            response=response,
            memory_type=mem_type,
            topics=topics,
            auto_save=False
        )
        foundation_ids.append(mem_id)
    
    # Layer 2: Intermediate concepts (reference foundations)
    intermediate = [
        ("How do SQL joins work?", "Joins combine data from multiple tables", 
         ["SQL", "joins"], "explanation", [foundation_ids[0]]),
        ("Python async programming", "Use async/await for concurrent operations", 
         ["Python", "async"], "explanation", [foundation_ids[1]]),
        ("dbt models explained", "dbt models are SQL files with Jinja", 
         ["dbt", "models"], "explanation", [foundation_ids[2]]),
        ("When to use indexes?", "Use indexes on columns in WHERE and JOIN clauses", 
         ["SQL", "indexes", "optimization"], "guide", [foundation_ids[0], foundation_ids[3]]),
    ]
    
    intermediate_ids = []
    for query, response, topics, mem_type, refs in intermediate:
        mem_id = memory_network.add_memory_with_references(
            query=query,
            response=response,
            memory_type=mem_type,
            topics=topics,
            potential_references=refs,
            auto_save=False
        )
        intermediate_ids.append(mem_id)
    
    # Layer 3: Advanced concepts (reference intermediate)
    advanced = [
        ("Optimize slow SQL joins", "Add indexes on join columns, use EXPLAIN", 
         ["SQL", "optimization", "performance"], "solution", 
         [intermediate_ids[0], intermediate_ids[3]]),
        ("Debug dbt model timeout", "Check for missing indexes, use incremental models", 
         ["dbt", "debugging", "performance"], "solution", 
         [intermediate_ids[2], intermediate_ids[3]]),
        ("Async database queries in Python", "Use asyncpg or databases library", 
         ["Python", "async", "SQL"], "code_implementation", 
         [intermediate_ids[1], foundation_ids[0]]),
    ]
    
    advanced_ids = []
    for query, response, topics, mem_type, refs in advanced:
        mem_id = memory_network.add_memory_with_references(
            query=query,
            response=response,
            memory_type=mem_type,
            topics=topics,
            potential_references=refs,
            auto_save=False
        )
        advanced_ids.append(mem_id)
    
    # Layer 4: Cross-domain solutions (reference multiple layers)
    solutions = [
        ("Build async data pipeline", "Use Python async with dbt and optimized SQL", 
         ["Python", "dbt", "SQL", "pipeline"], "analysis", 
         foundation_ids + intermediate_ids[:2] + [advanced_ids[2]]),
        ("Fix slow data transformation", "Profile queries, add indexes, use incremental dbt", 
         ["performance", "debugging", "solution"], "solution", 
         [advanced_ids[0], advanced_ids[1]] + intermediate_ids[2:]),
    ]
    
    for query, response, topics, mem_type, refs in solutions:
        memory_network.add_memory_with_references(
            query=query,
            response=response,
            memory_type=mem_type,
            topics=topics,
            potential_references=refs,
            auto_save=False
        )
    
    return foundation_ids + intermediate_ids + advanced_ids


def test_advanced_reference_resolution():
    """Test advanced reference resolution capabilities"""
    print("=== Advanced Reference Resolution Test ===\n")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize systems
        conv_system = ConversationSystem(temp_dir)
        mock_memory = AdvancedMockMemory()
        memory_network = MemoryNetworkSystem(mock_memory, conv_system)
        
        # Test 1: Multi-layer knowledge graph
        print("1. Building Multi-layer Knowledge Graph")
        print("-" * 40)
        
        memory_ids = create_knowledge_graph_scenario(memory_network)
        
        print(f"Created knowledge graph with {len(memory_ids)} memories")
        print(f"Total edges: {memory_network.memory_graph.number_of_edges()}")
        
        # Analyze graph structure
        in_degrees = dict(memory_network.memory_graph.in_degree())
        out_degrees = dict(memory_network.memory_graph.out_degree())
        
        print(f"\nMost referenced memories:")
        for mem_id, degree in sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]:
            if degree > 0:
                node = memory_network.memory_graph.nodes[mem_id]['data']
                print(f"  - {node.query[:50]}: {degree} references")
        print()
        
        # Test 2: Multi-hop traversal
        print("2. Testing Multi-hop Reference Traversal")
        print("-" * 40)
        
        # Start from a solution and trace back to foundations
        solution_nodes = [
            n for n in memory_network.memory_graph.nodes() 
            if memory_network.memory_graph.nodes[n]['data'].memory_type == 'solution'
        ]
        
        if solution_nodes:
            start_node = solution_nodes[0]
            print(f"Starting from: {memory_network.memory_graph.nodes[start_node]['data'].query}")
            
            # Find all ancestors (memories this builds upon)
            ancestors = nx.ancestors(memory_network.memory_graph, start_node)
            print(f"Builds upon {len(ancestors)} other memories")
            
            # Find shortest paths to foundations
            foundation_nodes = [
                n for n in memory_network.memory_graph.nodes()
                if 'basics' in memory_network.memory_graph.nodes[n]['data'].topics
            ]
            
            print("\nPaths to foundational knowledge:")
            for foundation in foundation_nodes[:3]:
                if foundation in ancestors:
                    try:
                        path = nx.shortest_path(memory_network.memory_graph, foundation, start_node)
                        print(f"\n  Path length: {len(path)-1} hops")
                        for i, node_id in enumerate(path):
                            node = memory_network.memory_graph.nodes[node_id]['data']
                            print(f"    {i}: {node.query[:50]}")
                    except nx.NetworkXNoPath:
                        pass
        print()
        
        # Test 3: Semantic reference finding
        print("3. Testing Semantic Reference Finding")
        print("-" * 40)
        
        test_queries = [
            "How to fix timeout in data pipeline?",
            "Optimize Python database queries",
            "Debug slow SQL joins in dbt"
        ]
        
        for test_query in test_queries:
            print(f"\nQuery: {test_query}")
            
            # Find related memories using the internal method
            related = memory_network._find_related_memories(
                test_query, 
                ["performance", "optimization"], 
                max_references=5
            )
            
            print(f"Found {len(related)} related memories:")
            for mem_id in related[:3]:
                node = memory_network.memory_graph.nodes[mem_id]['data']
                print(f"  - {node.query[:50]}")
        print()
        
        # Test 4: Reference strength analysis
        print("4. Testing Reference Strength Patterns")
        print("-" * 40)
        
        # Add memories with varying reference strengths
        base_mem = memory_network.add_memory_with_references(
            query="Complex performance issue",
            response="Requires multiple optimizations",
            memory_type="analysis",
            topics=["performance", "complex"],
            auto_save=False
        )
        
        # Add memories that reference it with different strengths
        ref_patterns = [
            ("Directly related solution", 0.9),
            ("Somewhat related approach", 0.5),
            ("Tangentially related topic", 0.2)
        ]
        
        for desc, strength in ref_patterns:
            new_mem = memory_network.add_memory_with_references(
                query=desc,
                response=f"Reference with strength {strength}",
                memory_type="solution",
                topics=["reference_test"],
                potential_references=[base_mem],
                auto_save=False
            )
            
            # Set reference weight
            if new_mem in memory_network.memory_graph.nodes:
                node = memory_network.memory_graph.nodes[new_mem]['data']
                node.reference_weights[base_mem] = strength
        
        print("Created reference strength test pattern")
        print()
        
        # Test 5: Circular reference detection
        print("5. Testing Circular Reference Handling")
        print("-" * 40)
        
        # Create potential circular references
        mem_a = memory_network.add_memory_with_references(
            query="Concept A",
            response="Depends on B",
            memory_type="explanation",
            topics=["circular_test"],
            auto_save=False
        )
        
        mem_b = memory_network.add_memory_with_references(
            query="Concept B", 
            response="Builds on A",
            memory_type="explanation",
            topics=["circular_test"],
            potential_references=[mem_a],
            auto_save=False
        )
        
        # Try to create circular reference
        if mem_a in memory_network.memory_graph.nodes:
            node_a = memory_network.memory_graph.nodes[mem_a]['data']
            node_a.references.append(mem_b)
            memory_network.memory_graph.add_edge(mem_a, mem_b)
        
        # Check for cycles
        cycles = list(nx.simple_cycles(memory_network.memory_graph))
        print(f"Detected {len(cycles)} cycles in the graph")
        if cycles:
            print("Circular references found - this should be handled carefully")
        print()
        
        # Test 6: Reference network visualization
        print("6. Generating Reference Network Visualization")
        print("-" * 40)
        
        # Create a subgraph of highly connected memories
        high_degree_nodes = [
            n for n, d in memory_network.memory_graph.degree() 
            if d >= 3
        ]
        
        if len(high_degree_nodes) >= 3:
            subgraph = memory_network.memory_graph.subgraph(high_degree_nodes)
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(subgraph, k=2, iterations=50)
            
            # Node colors by memory type
            node_colors = []
            for node in subgraph.nodes():
                mem_type = memory_network.memory_graph.nodes[node]['data'].memory_type
                color_map = {
                    'explanation': 'lightblue',
                    'solution': 'lightgreen',
                    'analysis': 'lightyellow',
                    'code_implementation': 'lightcoral',
                    'guide': 'lightgray'
                }
                node_colors.append(color_map.get(mem_type, 'white'))
            
            # Draw network
            nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                                 node_size=1000, alpha=0.8)
            nx.draw_networkx_edges(subgraph, pos, edge_color='gray', 
                                 arrows=True, arrowsize=20, alpha=0.5)
            
            # Add labels
            labels = {}
            for node in subgraph.nodes():
                query = memory_network.memory_graph.nodes[node]['data'].query
                labels[node] = query[:20] + "..." if len(query) > 20 else query
            
            nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
            
            plt.title("Reference Network of Highly Connected Memories")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('test_reference_network.png', dpi=150, bbox_inches='tight')
            print("✓ Saved reference network to test_reference_network.png")
        
        # Test 7: Performance with deep references
        print("\n7. Testing Deep Reference Chain Performance")
        print("-" * 40)
        
        # Create a deep chain
        chain_length = 20
        chain_ids = []
        
        import time
        start_time = time.time()
        
        for i in range(chain_length):
            refs = chain_ids[-3:] if i > 0 else []  # Reference last 3
            mem_id = memory_network.add_memory_with_references(
                query=f"Chain element {i}",
                response=f"Part {i} of deep chain",
                memory_type="test",
                topics=["chain_test"],
                potential_references=refs,
                auto_save=False
            )
            chain_ids.append(mem_id)
        
        creation_time = time.time() - start_time
        print(f"Created {chain_length}-deep chain in {creation_time:.3f}s")
        
        # Test traversal performance
        start_time = time.time()
        descendants = nx.descendants(memory_network.memory_graph, chain_ids[0])
        traversal_time = time.time() - start_time
        
        print(f"Traversed {len(descendants)} descendants in {traversal_time:.3f}s")
        
        # Summary statistics
        print("\n=== Reference Network Statistics ===")
        print(f"Total memories: {memory_network.memory_graph.number_of_nodes()}")
        print(f"Total references: {memory_network.memory_graph.number_of_edges()}")
        print(f"Average degree: {np.mean([d for n, d in memory_network.memory_graph.degree()]):.2f}")
        
        # Find most connected component
        components = list(nx.weakly_connected_components(memory_network.memory_graph))
        largest_component = max(components, key=len)
        print(f"Largest connected component: {len(largest_component)} memories")
        
        print("\n✅ Advanced reference resolution test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_advanced_reference_resolution()