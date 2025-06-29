#!/usr/bin/env python3
"""
Test 8: Memory Network Analytics
Comprehensive analytics and insights from ADAM's memory network
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tempfile
import shutil
from collections import defaultdict, Counter

from src.adam import ConversationSystem, MemoryNetworkSystem, ConversationAwareMemorySystem


class AnalyticsMockMemory:
    """Mock memory system with rich data generation"""
    def __init__(self):
        self.memory_count = 0
        self.memories = {}
        
    def remember_if_worthy(self, **kwargs):
        self.memory_count += 1
        mem_id = f"mem_{self.memory_count:05d}"
        self.memories[mem_id] = kwargs
        return mem_id
    
    def get_embedding(self, text):
        # Generate consistent embeddings
        np.random.seed(hash(text) % 10000)
        return np.random.rand(384)
    
    def search(self, query, n_results=10):
        # Return some mock results
        results = []
        for i, (mem_id, mem) in enumerate(list(self.memories.items())[:n_results]):
            results.append({
                'id': mem_id,
                'query': mem.get('query', ''),
                'response': mem.get('response', ''),
                'distance': 0.1 * (i + 1)
            })
        return results


def generate_realistic_memory_network(memory_network, days=60, users=5):
    """Generate a realistic memory network with various patterns"""
    
    # User profiles with different expertise areas
    user_profiles = {
        'data_engineer': {
            'topics': ['SQL', 'dbt', 'ETL', 'pipeline', 'optimization'],
            'problem_rate': 0.7,
            'learning_rate': 0.3
        },
        'ml_engineer': {
            'topics': ['Python', 'ML', 'tensorflow', 'model', 'training'],
            'problem_rate': 0.5,
            'learning_rate': 0.5
        },
        'backend_dev': {
            'topics': ['API', 'Python', 'async', 'database', 'performance'],
            'problem_rate': 0.6,
            'learning_rate': 0.4
        },
        'analyst': {
            'topics': ['SQL', 'visualization', 'pandas', 'analysis', 'reporting'],
            'problem_rate': 0.4,
            'learning_rate': 0.6
        },
        'devops': {
            'topics': ['deployment', 'docker', 'kubernetes', 'monitoring', 'CI/CD'],
            'problem_rate': 0.8,
            'learning_rate': 0.2
        }
    }
    
    # Common problem patterns
    problem_patterns = [
        ("timeout", "performance issue", ["performance", "optimization"]),
        ("error", "bug fix", ["debugging", "error", "solution"]),
        ("slow", "optimization needed", ["performance", "optimization"]),
        ("implement", "new feature", ["implementation", "feature"]),
        ("understand", "explanation needed", ["explanation", "learning"]),
        ("fix", "problem solution", ["debugging", "solution"]),
        ("best practice", "guidance needed", ["best-practices", "guide"])
    ]
    
    # Simulate daily activity
    all_memories = []
    
    for day in range(days):
        current_date = datetime.now() - timedelta(days=days-day)
        
        # Vary activity by day of week (less on weekends)
        is_weekend = current_date.weekday() >= 5
        daily_activity = np.random.poisson(3 if is_weekend else 10)
        
        for _ in range(daily_activity):
            # Select random user
            user_type = np.random.choice(list(user_profiles.keys()))
            profile = user_profiles[user_type]
            
            # Determine interaction type
            is_problem = np.random.rand() < profile['problem_rate']
            
            if is_problem:
                # Generate problem-solution pair
                pattern_key, pattern_desc, extra_topics = problem_patterns[
                    np.random.randint(len(problem_patterns))
                ]
                main_topic = np.random.choice(profile['topics'])
                
                query = f"{pattern_key} with {main_topic} - {pattern_desc}"
                response = f"Here's the solution for {main_topic} {pattern_desc}"
                topics = [main_topic] + extra_topics
                memory_type = "solution" if "solution" in extra_topics else "explanation"
                
            else:
                # Learning/exploration query
                main_topic = np.random.choice(profile['topics'])
                learn_types = ["How to", "What is", "Explain", "Best way to"]
                query_start = np.random.choice(learn_types)
                
                query = f"{query_start} {main_topic}?"
                response = f"Detailed explanation about {main_topic}"
                topics = [main_topic, "learning", "explanation"]
                memory_type = "explanation"
            
            # Add context metadata
            context = {
                'user_type': user_type,
                'interaction_type': 'problem' if is_problem else 'learning',
                'time_of_day': current_date.hour,
                'day_of_week': current_date.strftime('%A')
            }
            
            # Create memory with references to related past memories
            references = []
            if len(all_memories) > 5 and np.random.rand() < 0.4:
                # Reference related memories
                related_mems = [
                    m for m in all_memories[-20:] 
                    if any(t in topics for t in memory_network.memory_graph.nodes[m]['data'].topics)
                ]
                if related_mems:
                    references = np.random.choice(
                        related_mems, 
                        min(3, len(related_mems)), 
                        replace=False
                    ).tolist()
            
            mem_id = memory_network.add_memory_with_references(
                query=query,
                response=response,
                memory_type=memory_type,
                topics=topics,
                potential_references=references,
                auto_save=False
            )
            
            # Set timestamp and access patterns
            if mem_id in memory_network.memory_graph.nodes:
                node = memory_network.memory_graph.nodes[mem_id]['data']
                node.timestamp = current_date
                
                # Store context as graph node attribute (not on MemoryNode)
                memory_network.memory_graph.nodes[mem_id]['context'] = context
                
                # Simulate access patterns
                if is_problem and "solution" in topics:
                    # Solutions get accessed more
                    node.access_count = np.random.poisson(5)
                else:
                    node.access_count = np.random.poisson(2)
                
                if node.access_count > 0:
                    # Last accessed within a few days
                    node.last_accessed = current_date + timedelta(
                        days=np.random.randint(0, min(7, days-day))
                    )
            
            all_memories.append(mem_id)
    
    return all_memories


def test_memory_network_analytics():
    """Test comprehensive memory network analytics"""
    print("=== Memory Network Analytics Test ===\n")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize systems
        conv_system = ConversationSystem(temp_dir)
        mock_memory = AnalyticsMockMemory()
        memory_network = MemoryNetworkSystem(mock_memory, conv_system)
        
        # Generate realistic data
        print("1. Generating Realistic Memory Network")
        print("-" * 40)
        
        memories = generate_realistic_memory_network(memory_network, days=60, users=5)
        print(f"Generated {len(memories)} memories over 60 days")
        print()
        
        # Test 1: Basic network statistics
        print("2. Basic Network Statistics")
        print("-" * 40)
        
        stats = {
            'total_memories': memory_network.memory_graph.number_of_nodes(),
            'total_references': memory_network.memory_graph.number_of_edges(),
            'unique_topics': len(memory_network.topic_to_memories),
            'conversation_threads': len(memory_network.threads)
        }
        
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Degree distribution
        degrees = [d for n, d in memory_network.memory_graph.degree()]
        print(f"\nDegree distribution:")
        print(f"  Mean: {np.mean(degrees):.2f}")
        print(f"  Median: {np.median(degrees):.1f}")
        print(f"  Max: {max(degrees) if degrees else 0}")
        print()
        
        # Test 2: Topic analysis
        print("3. Topic Analysis")
        print("-" * 40)
        
        topic_stats = []
        for topic, memory_ids in memory_network.topic_to_memories.items():
            memories_data = []
            for mem_id in memory_ids:
                if mem_id in memory_network.memory_graph.nodes:
                    node = memory_network.memory_graph.nodes[mem_id]['data']
                    memories_data.append({
                        'access_count': node.access_count,
                        'age_days': (datetime.now() - node.timestamp).days,
                        'reference_count': len(node.references) + len(node.referenced_by)
                    })
            
            if memories_data:
                topic_stats.append({
                    'topic': topic,
                    'memory_count': len(memory_ids),
                    'avg_access': np.mean([m['access_count'] for m in memories_data]),
                    'avg_age': np.mean([m['age_days'] for m in memories_data]),
                    'avg_references': np.mean([m['reference_count'] for m in memories_data])
                })
        
        # Sort by memory count
        topic_stats.sort(key=lambda x: x['memory_count'], reverse=True)
        
        print("Top 10 Topics by Memory Count:")
        for stat in topic_stats[:10]:
            print(f"  {stat['topic']:15} | Memories: {stat['memory_count']:3} | "
                  f"Avg Access: {stat['avg_access']:.1f} | "
                  f"Avg Age: {stat['avg_age']:.0f}d")
        print()
        
        # Test 3: Memory type distribution
        print("4. Memory Type Distribution")
        print("-" * 40)
        
        type_counts = Counter()
        for node_id in memory_network.memory_graph.nodes():
            node = memory_network.memory_graph.nodes[node_id]['data']
            type_counts[node.memory_type] += 1
        
        total = sum(type_counts.values())
        for mem_type, count in type_counts.most_common():
            percentage = (count / total) * 100
            print(f"  {mem_type:20} {count:4} ({percentage:.1f}%)")
        print()
        
        # Test 4: Temporal patterns
        print("5. Temporal Activity Patterns")
        print("-" * 40)
        
        # Activity by day of week
        day_activity = defaultdict(int)
        hour_activity = defaultdict(int)
        
        for node_id in memory_network.memory_graph.nodes():
            node = memory_network.memory_graph.nodes[node_id]['data']
            # Get context from graph node attributes
            context = memory_network.memory_graph.nodes[node_id].get('context', {})
            if context:
                if 'day_of_week' in context:
                    day_activity[context['day_of_week']] += 1
                if 'time_of_day' in context:
                    hour_activity[context['time_of_day']] += 1
        
        print("Activity by Day of Week:")
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in days_order:
            if day in day_activity:
                print(f"  {day:10} {'█' * (day_activity[day] // 5)}")
        print()
        
        # Test 5: User behavior analysis
        print("6. User Behavior Analysis")
        print("-" * 40)
        
        user_stats = defaultdict(lambda: {'problems': 0, 'learning': 0, 'topics': set()})
        
        for node_id in memory_network.memory_graph.nodes():
            node = memory_network.memory_graph.nodes[node_id]['data']
            # Get context from graph node attributes
            context = memory_network.memory_graph.nodes[node_id].get('context', {})
            if context and 'user_type' in context:
                user = context['user_type']
                interaction = context.get('interaction_type', 'unknown')
                
                if interaction == 'problem':
                    user_stats[user]['problems'] += 1
                elif interaction == 'learning':
                    user_stats[user]['learning'] += 1
                
                user_stats[user]['topics'].update(node.topics)
        
        print("User Activity Profiles:")
        for user, stats in user_stats.items():
            total = stats['problems'] + stats['learning']
            if total > 0:
                problem_ratio = stats['problems'] / total
                print(f"\n  {user}:")
                print(f"    Total interactions: {total}")
                print(f"    Problem-solving: {stats['problems']} ({problem_ratio*100:.1f}%)")
                print(f"    Learning: {stats['learning']} ({(1-problem_ratio)*100:.1f}%)")
                print(f"    Unique topics: {len(stats['topics'])}")
        print()
        
        # Test 6: Knowledge evolution
        print("7. Knowledge Evolution Analysis")
        print("-" * 40)
        
        # Group memories by week
        week_stats = defaultdict(lambda: {
            'count': 0, 'new_topics': set(), 
            'references': 0, 'solutions': 0
        })
        
        all_topics_seen = set()
        
        for node_id in memory_network.memory_graph.nodes():
            node = memory_network.memory_graph.nodes[node_id]['data']
            week = node.timestamp.isocalendar()[1]
            
            week_stats[week]['count'] += 1
            week_stats[week]['references'] += len(node.references)
            
            if node.memory_type == 'solution':
                week_stats[week]['solutions'] += 1
            
            # Track new topics
            for topic in node.topics:
                if topic not in all_topics_seen:
                    week_stats[week]['new_topics'].add(topic)
                    all_topics_seen.add(topic)
        
        print("Weekly Knowledge Growth:")
        sorted_weeks = sorted(week_stats.keys())[-4:]  # Last 4 weeks
        for week in sorted_weeks:
            stats = week_stats[week]
            print(f"\n  Week {week}:")
            print(f"    Memories added: {stats['count']}")
            print(f"    New topics: {len(stats['new_topics'])}")
            print(f"    References created: {stats['references']}")
            print(f"    Solutions provided: {stats['solutions']}")
        print()
        
        # Test 7: Generate analytics visualizations
        print("8. Generating Analytics Visualizations")
        print("-" * 40)
        
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Topic popularity over time
        ax1 = plt.subplot(3, 3, 1)
        top_topics = [stat['topic'] for stat in topic_stats[:5]]
        topic_timeline = defaultdict(lambda: defaultdict(int))
        
        for node_id in memory_network.memory_graph.nodes():
            node = memory_network.memory_graph.nodes[node_id]['data']
            week = node.timestamp.isocalendar()[1]
            for topic in node.topics:
                if topic in top_topics:
                    topic_timeline[topic][week] += 1
        
        for topic in top_topics:
            weeks = sorted(topic_timeline[topic].keys())
            counts = [topic_timeline[topic][w] for w in weeks]
            ax1.plot(weeks, counts, label=topic, marker='o')
        
        ax1.set_xlabel('Week')
        ax1.set_ylabel('Memory Count')
        ax1.set_title('Topic Popularity Over Time')
        ax1.legend()
        
        # 2. Memory type distribution pie chart
        ax2 = plt.subplot(3, 3, 2)
        type_labels = list(type_counts.keys())
        type_values = list(type_counts.values())
        ax2.pie(type_values, labels=type_labels, autopct='%1.1f%%')
        ax2.set_title('Memory Type Distribution')
        
        # 3. Access count distribution
        ax3 = plt.subplot(3, 3, 3)
        access_counts = []
        for node_id in memory_network.memory_graph.nodes():
            node = memory_network.memory_graph.nodes[node_id]['data']
            access_counts.append(node.access_count)
        
        ax3.hist(access_counts, bins=20, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Access Count')
        ax3.set_ylabel('Number of Memories')
        ax3.set_title('Memory Access Distribution')
        
        # 4. Reference network density
        ax4 = plt.subplot(3, 3, 4)
        in_degrees = [d for n, d in memory_network.memory_graph.in_degree()]
        out_degrees = [d for n, d in memory_network.memory_graph.out_degree()]
        
        ax4.hist([in_degrees, out_degrees], bins=15, label=['Incoming', 'Outgoing'], 
                 alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Number of References')
        ax4.set_ylabel('Number of Memories')
        ax4.set_title('Reference Distribution')
        ax4.legend()
        
        # 5. Activity heatmap
        ax5 = plt.subplot(3, 3, 5)
        # Create hour x day matrix
        hour_day_matrix = np.zeros((24, 7))
        day_map = {day: i for i, day in enumerate(days_order)}
        
        for node_id in memory_network.memory_graph.nodes():
            node = memory_network.memory_graph.nodes[node_id]['data']
            # Get context from graph node attributes
            context = memory_network.memory_graph.nodes[node_id].get('context', {})
            if context:
                hour = context.get('time_of_day', -1)
                day = context.get('day_of_week', '')
                if hour >= 0 and day in day_map:
                    hour_day_matrix[hour][day_map[day]] += 1
        
        im = ax5.imshow(hour_day_matrix, cmap='YlOrRd', aspect='auto')
        ax5.set_yticks(range(24))
        ax5.set_xticks(range(7))
        ax5.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax5.set_ylabel('Hour of Day')
        ax5.set_title('Activity Heatmap')
        plt.colorbar(im, ax=ax5)
        
        # 6. Memory age distribution
        ax6 = plt.subplot(3, 3, 6)
        ages = []
        for node_id in memory_network.memory_graph.nodes():
            node = memory_network.memory_graph.nodes[node_id]['data']
            age = (datetime.now() - node.timestamp).days
            ages.append(age)
        
        ax6.hist(ages, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax6.set_xlabel('Age (days)')
        ax6.set_ylabel('Number of Memories')
        ax6.set_title('Memory Age Distribution')
        
        # 7. Topic co-occurrence
        ax7 = plt.subplot(3, 3, 7)
        # Build co-occurrence matrix
        topic_pairs = defaultdict(int)
        for node_id in memory_network.memory_graph.nodes():
            node = memory_network.memory_graph.nodes[node_id]['data']
            topics = sorted(node.topics)
            for i in range(len(topics)):
                for j in range(i+1, len(topics)):
                    pair = (topics[i], topics[j])
                    topic_pairs[pair] += 1
        
        # Get top co-occurring pairs
        top_pairs = sorted(topic_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
        pair_labels = [f"{p[0][0]}-{p[0][1]}" for p in top_pairs]
        pair_counts = [p[1] for p in top_pairs]
        
        ax7.barh(pair_labels, pair_counts)
        ax7.set_xlabel('Co-occurrence Count')
        ax7.set_title('Top Topic Co-occurrences')
        
        # 8. User problem vs learning ratio
        ax8 = plt.subplot(3, 3, 8)
        users = []
        problem_ratios = []
        
        for user, stats in user_stats.items():
            total = stats['problems'] + stats['learning']
            if total > 10:  # Only include active users
                users.append(user)
                problem_ratios.append(stats['problems'] / total)
        
        ax8.bar(users, problem_ratios)
        ax8.set_ylabel('Problem-Solving Ratio')
        ax8.set_title('User Interaction Patterns')
        ax8.set_ylim(0, 1)
        
        # 9. Memory decay forecast
        ax9 = plt.subplot(3, 3, 9)
        decay_scores = []
        for node_id in memory_network.memory_graph.nodes():
            node = memory_network.memory_graph.nodes[node_id]['data']
            decay = memory_network._calculate_memory_decay(
                node.timestamp, node.access_count, node.last_accessed
            )
            decay_scores.append(decay)
        
        # Group by decay range
        decay_ranges = {
            'Active (>0.8)': sum(1 for d in decay_scores if d > 0.8),
            'Fading (0.5-0.8)': sum(1 for d in decay_scores if 0.5 < d <= 0.8),
            'Weak (0.1-0.5)': sum(1 for d in decay_scores if 0.1 < d <= 0.5),
            'To Remove (<0.1)': sum(1 for d in decay_scores if d <= 0.1)
        }
        
        ax9.bar(decay_ranges.keys(), decay_ranges.values(), 
                color=['green', 'yellow', 'orange', 'red'])
        ax9.set_ylabel('Number of Memories')
        ax9.set_title('Memory Decay Status')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('test_memory_analytics.png', dpi=150, bbox_inches='tight')
        print("✓ Saved comprehensive analytics to test_memory_analytics.png")
        
        # Generate summary report
        print("\n9. Analytics Summary Report")
        print("-" * 40)
        
        print("\n📊 ADAM Memory Network Analytics Report")
        print("=" * 50)
        
        # Recompute stats for summary
        total_memories = memory_network.memory_graph.number_of_nodes()
        total_references = memory_network.memory_graph.number_of_edges()
        unique_topics = len(memory_network.topic_to_memories)
        conversation_threads = len(memory_network.threads)
        
        print(f"\n📈 Network Overview:")
        print(f"  • Total Memories: {total_memories}")
        print(f"  • Total References: {total_references}")
        print(f"  • Reference Density: {total_references/total_memories:.2f} refs/memory")
        print(f"  • Unique Topics: {unique_topics}")
        print(f"  • Conversation Threads: {conversation_threads}")
        
        print(f"\n🏆 Top Performing Memories:")
        # Find most accessed memories
        top_accessed = sorted(
            [(n, memory_network.memory_graph.nodes[n]['data']) for n in memory_network.memory_graph.nodes()],
            key=lambda x: x[1].access_count,
            reverse=True
        )[:5]
        
        for mem_id, node in top_accessed:
            print(f"  • {node.query[:50]}: {node.access_count} accesses")
        
        print(f"\n📊 Knowledge Distribution:")
        type_total = sum(type_counts.values()) if 'type_counts' in locals() else total_memories
        print(f"  • Solutions: {type_counts.get('solution', 0)} ({type_counts.get('solution', 0)/type_total*100:.1f}%)")
        print(f"  • Explanations: {type_counts.get('explanation', 0)} ({type_counts.get('explanation', 0)/type_total*100:.1f}%)")
        print(f"  • Implementations: {type_counts.get('code_implementation', 0)}")
        
        print(f"\n⏰ Activity Patterns:")
        busiest_day = max(day_activity.items(), key=lambda x: x[1])[0] if day_activity else "N/A"
        print(f"  • Busiest Day: {busiest_day}")
        print(f"  • Peak Hours: {', '.join(str(h) for h, c in sorted(hour_activity.items(), key=lambda x: x[1], reverse=True)[:3])}")
        
        print(f"\n🔮 Predictions:")
        removal_count = sum(1 for d in decay_scores if d <= 0.1)
        print(f"  • Memories scheduled for removal: {removal_count}")
        print(f"  • Memory growth rate: ~{len(memories)/60:.1f} memories/day")
        print(f"  • Projected size in 30 days: ~{int(len(memories) * 1.5)} memories")
        
        print("\n✅ Analytics test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_memory_network_analytics()