#!/usr/bin/env python3
"""
Visualize the LangGraph conversation state machine
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np


def create_langgraph_visualization():
    """Create a visual representation of the LangGraph state machine"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define node positions
    nodes = {
        'start': (1, 5),
        'analyze_query': (3, 5),
        'check_memory': (5, 5),
        'verify_freshness': (7, 7),
        'route_llm': (9, 5),
        'generate_response': (11, 5),
        'handle_error': (11, 3),
        'store_result': (13, 5),
        'end': (13, 5)
    }
    
    # Node colors based on function
    node_colors = {
        'start': '#90EE90',  # Light green
        'analyze_query': '#87CEEB',  # Sky blue
        'check_memory': '#FFB6C1',  # Light pink
        'verify_freshness': '#DDA0DD',  # Plum
        'route_llm': '#F0E68C',  # Khaki
        'generate_response': '#FFA07A',  # Light salmon
        'handle_error': '#FF6B6B',  # Red
        'store_result': '#98FB98',  # Pale green
        'end': '#90EE90'  # Light green
    }
    
    # Draw nodes
    for node, (x, y) in nodes.items():
        if node in ['start', 'end']:
            # Circle for start/end
            circle = Circle((x, y), 0.3, color=node_colors[node], ec='black', linewidth=2)
            ax.add_patch(circle)
            if node == 'start':
                ax.text(x, y-0.6, 'START', ha='center', fontsize=10, weight='bold')
        else:
            # Rectangle for process nodes
            width = 1.8
            height = 0.6
            rect = FancyBboxPatch(
                (x - width/2, y - height/2), width, height,
                boxstyle="round,pad=0.1",
                facecolor=node_colors[node],
                edgecolor='black',
                linewidth=2
            )
            ax.add_patch(rect)
            
            # Node labels
            labels = {
                'analyze_query': 'Analyze\nComplexity',
                'check_memory': 'Check\nMemory',
                'verify_freshness': 'Verify\nFreshness',
                'route_llm': 'Route to\nLLM',
                'generate_response': 'Generate\nResponse',
                'handle_error': 'Handle\nError',
                'store_result': 'Store\nResult'
            }
            ax.text(x, y, labels[node], ha='center', va='center', fontsize=9, weight='bold')
    
    # Draw edges
    edges = [
        ('start', 'analyze_query', 'straight'),
        ('analyze_query', 'check_memory', 'straight'),
        ('check_memory', 'verify_freshness', 'conditional'),
        ('check_memory', 'route_llm', 'conditional'),
        ('verify_freshness', 'route_llm', 'straight'),
        ('route_llm', 'generate_response', 'straight'),
        ('generate_response', 'handle_error', 'conditional'),
        ('generate_response', 'store_result', 'conditional'),
        ('handle_error', 'generate_response', 'retry'),
        ('store_result', 'end', 'straight')
    ]
    
    for start, end, edge_type in edges:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        
        if edge_type == 'straight':
            arrow = FancyArrowPatch(
                (x1 + 0.9, y1), (x2 - 0.9, y2),
                arrowstyle='->', mutation_scale=20,
                color='black', linewidth=2
            )
        elif edge_type == 'conditional':
            # Curved arrow for conditional
            if end == 'verify_freshness':
                mid_x, mid_y = (x1 + x2) / 2, y1 + 1.5
            else:
                mid_x, mid_y = (x1 + x2) / 2, y1
            
            arrow = FancyArrowPatch(
                (x1 + 0.9, y1), (x2 - 0.9, y2),
                connectionstyle=f"arc3,rad=.3",
                arrowstyle='->', mutation_scale=20,
                color='blue', linewidth=2, linestyle='dashed'
            )
        elif edge_type == 'retry':
            # Loop back arrow
            arrow = FancyArrowPatch(
                (x1, y1 - 0.3), (x2, y2 - 0.3),
                connectionstyle="arc3,rad=-.5",
                arrowstyle='->', mutation_scale=20,
                color='red', linewidth=2
            )
        
        ax.add_patch(arrow)
    
    # Add edge labels
    ax.text(6, 6.2, 'if confidence > 0.7', fontsize=8, style='italic', color='blue')
    ax.text(6.5, 4.5, 'else', fontsize=8, style='italic', color='blue')
    ax.text(11, 4.2, 'if error', fontsize=8, style='italic', color='blue')
    ax.text(11.5, 3.8, 'retry', fontsize=8, style='italic', color='red')
    
    # Add state information boxes
    state_info = [
        "State Fields:",
        "• query: str",
        "• complexity: simple/moderate/complex",
        "• memory_confidence: 0.0-1.0",
        "• memory_found: bool",
        "• should_verify: bool",
        "• selected_model: mistral/gpt-3.5/gpt-4/claude",
        "• response: str",
        "• total_cost: float"
    ]
    
    info_box = FancyBboxPatch(
        (0.2, 0.5), 3.5, 2.5,
        boxstyle="round,pad=0.1",
        facecolor='lightyellow',
        edgecolor='black',
        linewidth=1
    )
    ax.add_patch(info_box)
    
    for i, line in enumerate(state_info):
        ax.text(0.4, 2.8 - i*0.3, line, fontsize=8, 
               weight='bold' if i == 0 else 'normal')
    
    # Add complexity routing info
    routing_info = [
        "Model Selection Logic:",
        "Simple → Grok-3-mini",
        "Moderate → Grok-3-mini",
        "Complex → O3",
        "Complex + Code → Claude Opus 4",
        "High Memory → Grok-3-mini"
    ]
    
    routing_box = FancyBboxPatch(
        (10, 0.5), 3.5, 1.8,
        boxstyle="round,pad=0.1",
        facecolor='lightblue',
        edgecolor='black',
        linewidth=1
    )
    ax.add_patch(routing_box)
    
    for i, line in enumerate(routing_info):
        ax.text(10.2, 2.0 - i*0.3, line, fontsize=8,
               weight='bold' if i == 0 else 'normal')
    
    # Title
    ax.text(7, 9, 'ADAM LangGraph Conversation State Machine', 
            fontsize=16, weight='bold', ha='center')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='black', label='Sequential Flow'),
        mpatches.Patch(color='blue', label='Conditional Flow'),
        mpatches.Patch(color='red', label='Error/Retry Flow')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig


def create_cost_optimization_diagram():
    """Create a diagram showing cost optimization strategies"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Query distribution and model usage
    ax1.set_title('Query Complexity Distribution & Model Routing', fontsize=14, weight='bold')
    
    # Query distribution pie chart
    sizes = [70, 20, 10]
    labels = ['Simple (70%)', 'Moderate (20%)', 'Complex (10%)']
    colors = ['#90EE90', '#FFD700', '#FF6B6B']
    explode = (0.1, 0, 0)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.0f%%', startangle=90)
    
    # Model usage text
    model_text = [
        "Model Usage Strategy:",
        "• Simple → Grok-3-mini ($0.0002)",
        "• Moderate → Grok-3-mini ($0.0002)",  
        "• Complex → O3 ($37.50/1K tokens)",
        "• Coding → Claude Opus 4 ($0.025)",
        "",
        "Average cost per query: $0.0018",
        "Monthly cost (1500 queries): $2.70"
    ]
    
    for i, text in enumerate(model_text):
        ax1.text(-1.5, -1.5 - i*0.15, text, fontsize=10,
                weight='bold' if i == 0 else 'normal')
    
    # Right plot: Memory impact on cost
    ax2.set_title('Memory Cache Impact on Costs', fontsize=14, weight='bold')
    
    # Cost comparison bars
    scenarios = ['No Memory', 'With Memory\n(50% hit rate)', 'With Memory\n(80% hit rate)']
    costs = [5.0, 2.75, 1.5]
    colors = ['#FF6B6B', '#FFD700', '#90EE90']
    
    bars = ax2.bar(scenarios, costs, color=colors, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Monthly Cost ($)', fontsize=12)
    ax2.set_ylim(0, 6)
    
    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'${cost:.2f}', ha='center', va='bottom', fontsize=11, weight='bold')
    
    # Add savings annotations
    ax2.annotate('45% savings', xy=(1, 2.75), xytext=(1, 4),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green', weight='bold', ha='center')
    
    ax2.annotate('70% savings', xy=(2, 1.5), xytext=(2, 3.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green', weight='bold', ha='center')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Generate visualizations
    print("Generating LangGraph visualization...")
    fig1 = create_langgraph_visualization()
    fig1.savefig('langgraph_state_machine.png', dpi=300, bbox_inches='tight')
    print("Saved: langgraph_state_machine.png")
    
    print("Generating cost optimization diagram...")
    fig2 = create_cost_optimization_diagram()
    fig2.savefig('cost_optimization_diagram.png', dpi=300, bbox_inches='tight')
    print("Saved: cost_optimization_diagram.png")
    
    # Show plots
    plt.show()