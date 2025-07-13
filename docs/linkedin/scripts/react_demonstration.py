#!/usr/bin/env python3
"""
ReAct (Reasoning and Acting) Framework Demonstration
Shows ADAM's ability to think, act, and observe in solving complex problems
"""
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from src.adam.llm.client import UnifiedLLMClient
from src.adam.llm.config import LLMConfig

class ReActDemo:
    def __init__(self):
        self.memory = ADAMMemoryAdvanced()
        self.llm_config = LLMConfig()
        self.llm_client = UnifiedLLMClient(self.llm_config)
        self.thought_history = []
        
    async def think(self, problem: str, context: Dict, observations: List[str] = None) -> str:
        """Generate a thought about the current problem state"""
        observations_text = ""
        if observations:
            observations_text = "\nPrevious observations:\n" + "\n".join(f"- {obs}" for obs in observations[-3:])
        
        prompt = f"""Using the ReAct framework, generate a THOUGHT about this problem:

Problem: {problem}

Context:
{json.dumps(context, indent=2)}
{observations_text}

Generate a single, focused thought about what to investigate or try next. Be specific and actionable.
Start with "Thought:" and keep it under 2 sentences."""

        response = await self.llm_client.complete(prompt=prompt, model="grok-3-mini", stream=False)
        thought = response.content.strip()
        if thought.startswith("Thought:"):
            thought = thought[8:].strip()
        
        self.thought_history.append(thought)
        return thought
    
    async def act(self, thought: str, problem_context: Dict) -> Dict:
        """Take an action based on the thought"""
        # Search memory for relevant information
        memories = self.memory.recall_with_context(query=thought, n_results=3)
        
        # Simulate different types of actions based on the thought
        action_types = {
            "analyze": "Analyzing data patterns and metrics",
            "check": "Checking system configuration and settings",
            "test": "Testing hypothesis with sample data",
            "investigate": "Investigating root causes",
            "optimize": "Applying optimization techniques"
        }
        
        # Determine action type from thought
        action_type = "investigate"
        for key in action_types:
            if key in thought.lower():
                action_type = key
                break
        
        action = {
            "type": action_type,
            "description": action_types[action_type],
            "memory_support": len(memories) > 0,
            "relevant_memories": len(memories)
        }
        
        return action
    
    async def observe(self, action: Dict, problem_context: Dict) -> str:
        """Make an observation based on the action taken"""
        # Generate observation based on action type and context
        prompt = f"""Based on this action in a BigQuery optimization context:

Action: {action['description']}
Action Type: {action['type']}
Memory Support: {'Yes' if action['memory_support'] else 'No'} ({action['relevant_memories']} relevant memories)

Problem Context:
{json.dumps(problem_context, indent=2)}

Generate a realistic observation that would result from this action. 
The observation should reveal new information about the problem.
Keep it concise (1-2 sentences) and specific to BigQuery.
Start with "Observation:" """

        response = await self.llm_client.complete(prompt=prompt, model="grok-3-mini", stream=False)
        observation = response.content.strip()
        if observation.startswith("Observation:"):
            observation = observation[12:].strip()
        
        return observation
    
    async def solve_problem(self, scenario: Dict) -> Dict:
        """Solve a problem using the ReAct framework"""
        print(f"\n🧩 Solving: {scenario['title']}")
        print(f"📋 Problem: {scenario['problem']}")
        
        thoughts = []
        actions = []
        observations = []
        max_iterations = 4
        
        for i in range(max_iterations):
            print(f"\n🔄 Iteration {i+1}:")
            
            # Think
            thought = await self.think(scenario['problem'], scenario['context'], observations)
            thoughts.append(thought)
            print(f"💭 Thought: {thought}")
            
            # Act
            action = await self.act(thought, scenario['context'])
            actions.append(action)
            print(f"🎯 Action: {action['description']}")
            if action['memory_support']:
                print(f"   📚 Leveraging {action['relevant_memories']} relevant memories")
            
            # Observe
            observation = await self.observe(action, scenario['context'])
            observations.append(observation)
            print(f"👁️ Observation: {observation}")
            
            # Check if we have enough information to conclude
            if any(word in observation.lower() for word in ['identified', 'found', 'cause', 'solution']):
                print("\n✅ Problem-solving complete!")
                break
        
        # Generate final recommendation
        recommendation = await self.generate_recommendation(scenario, thoughts, actions, observations)
        
        return {
            "scenario": scenario['title'],
            "iterations": len(thoughts),
            "thoughts": thoughts,
            "actions": actions,
            "observations": observations,
            "recommendation": recommendation,
            "memory_utilized": sum(1 for a in actions if a['memory_support'])
        }
    
    async def generate_recommendation(self, scenario: Dict, thoughts: List[str], 
                                    actions: List[Dict], observations: List[str]) -> str:
        """Generate final recommendation based on ReAct process"""
        prompt = f"""Based on this ReAct problem-solving process for a BigQuery issue:

Problem: {scenario['problem']}

Thought-Action-Observation History:
{chr(10).join(f"{i+1}. Thought: {thoughts[i]}{chr(10)}   Action: {actions[i]['description']}{chr(10)}   Observation: {observations[i]}" for i in range(len(thoughts)))}

Generate a concise final recommendation (3-4 bullet points) that:
1. Addresses the root cause
2. Provides specific actions to take
3. Estimates expected improvement

Keep it practical and BigQuery-specific."""

        response = await self.llm_client.complete(prompt=prompt, model="grok-3-mini", stream=False)
        return response.content.strip()

async def main():
    """Run the ReAct demonstration"""
    print("="*60)
    print("🧠 ReAct Framework Demonstration")
    print("   Reasoning + Acting = Intelligent Problem Solving")
    print("="*60)
    
    demo = ReActDemo()
    
    # Load scenarios
    scenarios_path = Path(__file__).parent.parent / "data" / "react_scenarios.json"
    with open(scenarios_path, 'r') as f:
        data = json.load(f)
    
    # Demonstrate ReAct on first scenario
    scenario = data['scenarios'][0]
    result = await demo.solve_problem(scenario)
    
    # Save results
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    results_path = output_dir / f"react_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Create summary for LinkedIn
    summary = f"""# ReAct Framework Demonstration Results

## Problem Solved: {result['scenario']}

### Process Overview
- **Iterations**: {result['iterations']}
- **Memory Utilization**: {result['memory_utilized']} actions supported by past experience

### Thought-Action-Observation Cycle

"""
    
    for i in range(len(result['thoughts'])):
        summary += f"**Iteration {i+1}:**\n"
        summary += f"- 💭 **Thought**: {result['thoughts'][i]}\n"
        summary += f"- 🎯 **Action**: {result['actions'][i]['description']}\n"
        summary += f"- 👁️ **Observation**: {result['observations'][i]}\n\n"
    
    summary += f"""### Final Recommendation

{result['recommendation']}

### Key Insights
1. ReAct framework enables systematic problem decomposition
2. Each thought leads to specific, actionable investigations  
3. Observations inform the next iteration of thinking
4. Memory integration provides context from past experiences

---
*This demonstration shows how ADAM uses the ReAct framework to solve complex data engineering problems through iterative reasoning and action.*
"""
    
    summary_path = output_dir / f"react_demo_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"\n📄 Results saved to:")
    print(f"   - {results_path}")
    print(f"   - {summary_path}")
    
    print("\n🎯 ReAct Demo Complete!")
    print("\nKey Takeaways:")
    print("1. Structured thinking leads to better problem solving")
    print("2. Actions are informed by both reasoning and memory")
    print("3. Observations guide the next steps")
    print("4. The cycle continues until a solution emerges")

if __name__ == "__main__":
    asyncio.run(main())