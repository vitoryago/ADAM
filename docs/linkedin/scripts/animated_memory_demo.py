#!/usr/bin/env python3
"""
Animated Memory Demo - Shows ADAM solving BigQuery problem step by step
Creates multiple frames for video/GIF creation
"""
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.animation as animation

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

class AnimatedMemoryDemo:
    def __init__(self):
        self.output_dir = Path(__file__).parent.parent / "images" / "animation_frames"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def create_problem_solving_frames(self):
        """Create frames showing step-by-step problem solving"""
        
        # Define the story
        frames_data = [
            {
                'title': 'Step 1: Problem Detected',
                'highlight': ['problem'],
                'message': 'Dashboard query timeout: 185 seconds',
                'connections': []
            },
            {
                'title': 'Step 2: Searching Memory Network',
                'highlight': ['problem'],
                'message': 'Searching for similar cases...',
                'connections': ['problem-search']
            },
            {
                'title': 'Step 3: Similar Cases Found',
                'highlight': ['problem', 'case1', 'case2', 'case3'],
                'message': '3 similar cases found (92% match)',
                'connections': ['problem-case1', 'problem-case2', 'problem-case3']
            },
            {
                'title': 'Step 4: Extracting Patterns',
                'highlight': ['case1', 'case2', 'case3', 'pattern1', 'pattern2'],
                'message': 'Identifying optimization patterns...',
                'connections': ['case1-pattern1', 'case2-pattern1', 'case2-pattern2', 'case3-pattern2']
            },
            {
                'title': 'Step 5: Combining Solutions',
                'highlight': ['pattern1', 'pattern2', 'solution'],
                'message': 'Combining partition + clustering optimizations',
                'connections': ['pattern1-solution', 'pattern2-solution']
            },
            {
                'title': 'Step 6: Problem Solved!',
                'highlight': ['problem', 'solution'],
                'message': 'Query optimized: 185s → 4.2s (98% faster)',
                'connections': ['problem-solution'],
                'solved': True
            }
        ]
        
        # Node positions
        positions = {
            'problem': (0, 0),
            'search': (0, -2),
            'case1': (-3, -4),
            'case2': (0, -4),
            'case3': (3, -4),
            'pattern1': (-2, -6),
            'pattern2': (2, -6),
            'solution': (0, -8)
        }
        
        # Node details
        nodes = {
            'problem': {'label': 'Dashboard\nTimeout\n185s', 'color': '#FF6B6B', 'size': 1500},
            'search': {'label': 'Memory\nSearch', 'color': '#F39C12', 'size': 1000},
            'case1': {'label': 'Sales\nDashboard\n120s→8s', 'color': '#95E1D3', 'size': 1200},
            'case2': {'label': 'User\nAnalytics\n180s→12s', 'color': '#95E1D3', 'size': 1200},
            'case3': {'label': 'Revenue\nReport\n95s→5s', 'color': '#95E1D3', 'size': 1200},
            'pattern1': {'label': 'Partition\nFilter', 'color': '#4ECDC4', 'size': 1200},
            'pattern2': {'label': 'Table\nClustering', 'color': '#4ECDC4', 'size': 1200},
            'solution': {'label': 'SOLUTION\n4.2s', 'color': '#FFE66D', 'size': 1500}
        }
        
        # Create each frame
        for i, frame_data in enumerate(frames_data):
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Draw active connections
            for conn in frame_data['connections']:
                if '-' in conn:
                    start, end = conn.split('-')
                    if start in positions and end in positions:
                        x1, y1 = positions[start]
                        x2, y2 = positions[end]
                        
                        # Draw arrow
                        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                   arrowprops=dict(arrowstyle='->', lw=3, 
                                                 color='#2ECC71' if frame_data.get('solved') else '#3498DB',
                                                 alpha=0.8))
            
            # Draw all nodes (faded if not highlighted)
            for node_id, pos in positions.items():
                if node_id in nodes:
                    node = nodes[node_id]
                    is_highlighted = node_id in frame_data['highlight']
                    
                    # Draw node
                    circle = Circle(pos, radius=0.6, 
                                  facecolor=node['color'] if is_highlighted else '#CCCCCC',
                                  edgecolor='black' if is_highlighted else 'gray',
                                  linewidth=3 if is_highlighted else 1,
                                  alpha=1.0 if is_highlighted else 0.3)
                    ax.add_patch(circle)
                    
                    # Add label
                    ax.text(pos[0], pos[1], node['label'],
                           ha='center', va='center',
                           fontsize=10 if is_highlighted else 8,
                           fontweight='bold' if is_highlighted else 'normal',
                           alpha=1.0 if is_highlighted else 0.3)
            
            # Add title and message
            ax.text(0, 2, frame_data['title'], 
                   ha='center', fontsize=18, fontweight='bold')
            ax.text(0, 1.2, frame_data['message'],
                   ha='center', fontsize=14, style='italic')
            
            # Add frame number
            ax.text(0.95, 0.05, f"Frame {i+1}/6",
                   transform=ax.transAxes, ha='right', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Set limits and clean up
            ax.set_xlim(-5, 5)
            ax.set_ylim(-10, 3)
            ax.axis('off')
            
            # Save frame
            frame_path = self.output_dir / f"frame_{i+1:02d}.png"
            plt.tight_layout()
            plt.savefig(frame_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ Created frame {i+1}/6: {frame_path.name}")
        
        # Create instruction file
        instructions = """# Creating Animation from Frames

## Option 1: Create GIF (using ImageMagick)
```bash
convert -delay 150 -loop 0 frame_*.png adam_solving_problem.gif
```

## Option 2: Create MP4 (using ffmpeg)
```bash
ffmpeg -r 1 -i frame_%02d.png -vcodec libx264 -pix_fmt yuv420p adam_solving_problem.mp4
```

## Option 3: Manual slideshow
Upload frames to LinkedIn as a multi-image post
"""
        
        instruction_path = self.output_dir / "animation_instructions.md"
        with open(instruction_path, 'w') as f:
            f.write(instructions)
        
        print(f"\n📝 Instructions saved to: {instruction_path}")
        
        return self.output_dir
    
    def create_single_impact_visual(self):
        """Create a single powerful visualization showing impact"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Create the network
        G = nx.Graph()
        
        # Add nodes representing the problem-solution journey
        nodes = {
            'problem': {'pos': (-4, 0), 'label': 'SLOW QUERY\n185 seconds\n$45/run', 'color': '#E74C3C', 'size': 2000},
            'adam': {'pos': (0, 0), 'label': 'ADAM\nMemory\nNetwork', 'color': '#3498DB', 'size': 2500},
            'solution': {'pos': (4, 0), 'label': 'OPTIMIZED\n4.2 seconds\n$0.52/run', 'color': '#27AE60', 'size': 2000}
        }
        
        # Add memory nodes around ADAM
        memory_nodes = [
            {'id': 'm1', 'pos': (-1, 2), 'label': 'Partition\nPatterns'},
            {'id': 'm2', 'pos': (1, 2), 'label': 'Clustering\nStrategies'},
            {'id': 'm3', 'pos': (2, 1), 'label': 'Past\nSuccesses'},
            {'id': 'm4', 'pos': (2, -1), 'label': 'JOIN\nOptimizations'},
            {'id': 'm5', 'pos': (1, -2), 'label': 'Index\nPatterns'},
            {'id': 'm6', 'pos': (-1, -2), 'label': 'Query\nCache'},
            {'id': 'm7', 'pos': (-2, -1), 'label': 'Cost\nReductions'},
            {'id': 'm8', 'pos': (-2, 1), 'label': 'Similar\nCases'}
        ]
        
        # Draw the main flow
        arrow_props = dict(arrowstyle='->', lw=5, color='#34495E')
        ax.annotate('', xy=(0, 0), xytext=(-4, 0), arrowprops=arrow_props)
        ax.annotate('', xy=(4, 0), xytext=(0, 0), arrowprops=arrow_props)
        
        # Draw main nodes
        for node_id, node in nodes.items():
            circle = Circle(node['pos'], radius=0.8,
                          facecolor=node['color'],
                          edgecolor='black',
                          linewidth=3)
            ax.add_patch(circle)
            ax.text(node['pos'][0], node['pos'][1], node['label'],
                   ha='center', va='center',
                   fontsize=12, fontweight='bold',
                   color='white' if node_id != 'adam' else 'black')
        
        # Draw memory nodes
        for mem in memory_nodes:
            # Connection to center
            ax.plot([0, mem['pos'][0]], [0, mem['pos'][1]], 
                   'gray', alpha=0.3, lw=2, linestyle='--')
            
            # Node
            circle = Circle(mem['pos'], radius=0.4,
                          facecolor='#ECF0F1',
                          edgecolor='gray',
                          linewidth=2)
            ax.add_patch(circle)
            ax.text(mem['pos'][0], mem['pos'][1], mem['label'],
                   ha='center', va='center',
                   fontsize=8)
        
        # Add impact metrics
        metrics_text = "98% Faster • 99% Cost Reduction • 44x Performance"
        ax.text(0, -3.5, metrics_text,
               ha='center', fontsize=16, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#F1C40F', alpha=0.9))
        
        # Add title
        ax.text(0, 3.5, "ADAM's Memory Network in Action",
               ha='center', fontsize=20, fontweight='bold')
        
        # Clean up
        ax.set_xlim(-6, 6)
        ax.set_ylim(-4, 4)
        ax.axis('off')
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.output_dir.parent / f"memory_impact_visual_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Impact visualization saved to: {filepath}")
        return filepath

def main():
    print("="*60)
    print("Creating Animated Memory Demonstration")
    print("="*60)
    
    demo = AnimatedMemoryDemo()
    
    print("\n1. Creating problem-solving animation frames...")
    frames_dir = demo.create_problem_solving_frames()
    
    print("\n2. Creating single impact visualization...")
    impact_viz = demo.create_single_impact_visual()
    
    print("\n" + "="*60)
    print("✅ Animation frames created!")
    print(f"\nFrames location: {frames_dir}")
    print(f"Impact visual: {impact_viz}")
    print("\nNext steps:")
    print("1. Create GIF/video from frames (see animation_instructions.md)")
    print("2. Use impact visual as hero image for LinkedIn")
    print("3. Or upload frames as multi-image post")

if __name__ == "__main__":
    main()