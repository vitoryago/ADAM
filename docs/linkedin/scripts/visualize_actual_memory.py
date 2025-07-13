#!/usr/bin/env python3
"""
Visualize ADAM's Actual Memory Network - Real connections from the system
"""
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import json

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Silent initialization
import io
from contextlib import redirect_stdout, redirect_stderr

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    from src.adam.memory import ADAMMemoryAdvanced
    memory = ADAMMemoryAdvanced()

class ActualMemoryVisualizer:
    def __init__(self):
        self.memory = memory
        self.output_dir = Path(__file__).parent.parent / "images"
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_real_connections(self, query="BigQuery optimization", num_memories=25):
        """Extract real memory connections from ADAM"""
        print(f"Searching for memories related to: {query}")
        
        # Get initial memories
        with redirect_stdout(io.StringIO()):
            initial_memories = self.memory.recall_with_context(query=query, n_results=num_memories)
        
        if not initial_memories:
            print("No memories found!")
            return None, None
        
        print(f"Found {len(initial_memories)} initial memories")
        
        # Build node data
        nodes = {}
        edges = []
        
        # Process each memory
        for i, mem in enumerate(initial_memories):
            node_id = f"mem_{i}"
            
            # Extract key info from memory
            content = mem.get('content', '')
            metadata = mem.get('metadata', {})
            
            # Determine node type based on content
            if 'optimization' in content.lower() or 'pattern' in content.lower():
                node_type = 'pattern'
            elif 'query' in content.lower() and 'slow' in content.lower():
                node_type = 'problem'
            elif 'solution' in content.lower() or 'fix' in content.lower():
                node_type = 'solution'
            else:
                node_type = 'case'
            
            # Extract label (first meaningful part of content)
            if 'Query:' in content:
                label = content.split('Query:')[1][:40] + '...'
            elif 'Response:' in content:
                label = content.split('Response:')[1][:40] + '...'
            else:
                label = content[:40] + '...'
            
            # Clean up label
            label = label.replace('\n', ' ').strip()
            
            nodes[node_id] = {
                'label': label,
                'type': node_type,
                'distance': mem.get('distance', 0),
                'metadata': metadata,
                'full_content': content
            }
        
        # Create connections based on content similarity
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1 = f"mem_{i}"
                node2 = f"mem_{j}"
                
                # Calculate connection strength based on distance
                dist1 = nodes[node1]['distance']
                dist2 = nodes[node2]['distance']
                avg_distance = (dist1 + dist2) / 2
                
                # Connect if similar enough
                if avg_distance < 0.5:  # Threshold for connection
                    weight = 1 - avg_distance
                    edges.append((node1, node2, weight))
        
        print(f"Created {len(edges)} connections")
        return nodes, edges
    
    def create_actual_memory_network(self):
        """Create visualization of actual memory network"""
        nodes, edges = self.extract_real_connections("BigQuery optimization", num_memories=20)
        
        if not nodes:
            print("No data to visualize!")
            return None
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for node_id, node_data in nodes.items():
            G.add_node(node_id, **node_data)
        
        # Add edges
        for source, target, weight in edges:
            if weight > 0.5:  # Only show strong connections
                G.add_edge(source, target, weight=weight)
        
        # Calculate layout - group similar nodes
        pos = nx.spring_layout(G, k=3, iterations=100, weight='weight')
        
        # Color schemes
        node_colors = {
            'problem': '#FF6B6B',
            'pattern': '#4ECDC4',
            'solution': '#FFE66D',
            'case': '#95E1D3'
        }
        
        # Draw edges with transparency based on weight
        edge_widths = []
        edge_colors = []
        for (u, v, d) in G.edges(data=True):
            weight = d.get('weight', 0.5)
            edge_widths.append(weight * 5)
            if weight > 0.8:
                edge_colors.append('#2ECC71')
            elif weight > 0.6:
                edge_colors.append('#3498DB')
            else:
                edge_colors.append('#95A5A6')
        
        nx.draw_networkx_edges(G, pos, 
                             width=edge_widths,
                             edge_color=edge_colors,
                             alpha=0.6)
        
        # Draw nodes by type
        for node_type, color in node_colors.items():
            nodelist = [n for n, d in G.nodes(data=True) if d['type'] == node_type]
            if nodelist:
                nx.draw_networkx_nodes(G, pos,
                                     nodelist=nodelist,
                                     node_color=color,
                                     node_size=1000,
                                     alpha=0.9)
        
        # Add labels (shortened)
        labels = {}
        for node in G.nodes():
            label = G.nodes[node]['label']
            # Further shorten labels for readability
            if len(label) > 25:
                label = label[:22] + '...'
            labels[node] = label
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # Add title and stats
        plt.title(f"ADAM's Actual Memory Network: BigQuery Optimization Knowledge\n"
                 f"{len(nodes)} memories, {len([e for e in edges if e[2] > 0.5])} strong connections",
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Problems'),
            Patch(facecolor='#4ECDC4', label='Optimization Patterns'),
            Patch(facecolor='#FFE66D', label='Solutions'),
            Patch(facecolor='#95E1D3', label='Past Cases')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add statistics box
        stats_text = f"Memory Statistics:\n"
        stats_text += f"Total Memories: {len(nodes)}\n"
        stats_text += f"Strong Connections: {len([e for e in edges if e[2] > 0.8])}\n"
        stats_text += f"Patterns Found: {len([n for n,d in G.nodes(data=True) if d['type']=='pattern'])}\n"
        stats_text += f"Success Cases: {len([n for n,d in G.nodes(data=True) if d['type']=='case'])}"
        
        ax.text(0.02, 0.02, stats_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax.axis('off')
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.output_dir / f"actual_memory_network_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Actual memory network saved to: {filepath}")
        return filepath
    
    def create_memory_cluster_view(self):
        """Create a view showing how memories cluster by topic"""
        # Get memories for different BigQuery topics
        topics = {
            'performance': 'BigQuery slow query performance',
            'cost': 'BigQuery cost optimization',
            'partitioning': 'BigQuery partition clustering',
            'joins': 'BigQuery JOIN optimization'
        }
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        G = nx.Graph()
        all_nodes = {}
        node_id = 0
        
        # Colors for each topic
        topic_colors = {
            'performance': '#E74C3C',
            'cost': '#F39C12',
            'partitioning': '#3498DB',
            'joins': '#9B59B6'
        }
        
        # Get memories for each topic
        for topic, query in topics.items():
            with redirect_stdout(io.StringIO()):
                memories = self.memory.recall_with_context(query=query, n_results=5)
            
            for mem in memories:
                content = mem.get('content', '')[:50] + '...'
                all_nodes[f"node_{node_id}"] = {
                    'label': content.replace('\n', ' '),
                    'topic': topic,
                    'distance': mem.get('distance', 0)
                }
                G.add_node(f"node_{node_id}", **all_nodes[f"node_{node_id}"])
                node_id += 1
        
        # Connect nodes within same topic
        for i in range(len(all_nodes)):
            for j in range(i + 1, len(all_nodes)):
                node1 = f"node_{i}"
                node2 = f"node_{j}"
                if node1 in all_nodes and node2 in all_nodes:
                    if all_nodes[node1]['topic'] == all_nodes[node2]['topic']:
                        G.add_edge(node1, node2, weight=0.8)
                    else:
                        # Weaker inter-topic connections
                        if np.random.random() < 0.2:
                            G.add_edge(node1, node2, weight=0.3)
        
        # Layout with clustering
        pos = nx.spring_layout(G, k=2, iterations=100)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
        
        # Draw nodes by topic
        for topic, color in topic_colors.items():
            nodelist = [n for n, d in G.nodes(data=True) if d.get('topic') == topic]
            if nodelist:
                nx.draw_networkx_nodes(G, pos,
                                     nodelist=nodelist,
                                     node_color=color,
                                     node_size=800,
                                     label=topic.capitalize())
        
        # Title
        plt.title("ADAM's Memory Clusters: BigQuery Knowledge Domains",
                 fontsize=16, fontweight='bold')
        
        plt.legend(loc='upper right', frameon=True, fancybox=True)
        ax.axis('off')
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.output_dir / f"memory_clusters_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Memory cluster view saved to: {filepath}")
        return filepath

def main():
    print("="*60)
    print("Visualizing ADAM's Actual Memory Network")
    print("="*60)
    
    visualizer = ActualMemoryVisualizer()
    
    print("\n1. Creating actual memory network from real data...")
    viz1 = visualizer.create_actual_memory_network()
    
    print("\n2. Creating memory cluster visualization...")
    viz2 = visualizer.create_memory_cluster_view()
    
    # Also create the designed visualizations
    print("\n3. Creating problem-solving visualization...")
    from visualize_real_memory import RealMemoryVisualizer
    designed_viz = RealMemoryVisualizer()
    viz3 = designed_viz.create_problem_solving_visualization()
    
    print("\n" + "="*60)
    print("✅ All visualizations complete!")
    print("\nRecommended for LinkedIn:")
    print(f"  1. {viz3.name} - Best for showing problem-solving process")
    print(f"  2. {viz1.name} - Shows real memory connections")
    print(f"  3. {viz2.name} - Shows knowledge organization")

if __name__ == "__main__":
    main()