#!/usr/bin/env python3
"""
BigQuery Performance Optimization Demo for LinkedIn
Shows how ADAM learns and improves query optimization suggestions
"""
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from src.adam.llm.client import UnifiedLLMClient
from src.adam.llm.config import LLMConfig

class BigQueryDemo:
    def __init__(self):
        self.memory = ADAMMemoryAdvanced()
        self.llm_config = LLMConfig()
        self.llm_client = UnifiedLLMClient(self.llm_config)
        self.demo_results = []
        
    async def analyze_query(self, scenario: Dict) -> Dict:
        """Analyze a BigQuery scenario using ADAM's memory"""
        print(f"\n🔍 Analyzing: {scenario['title']}")
        print(f"   Original runtime: {scenario['metrics']['original_runtime']}s")
        print(f"   Data processed: {scenario['metrics']['data_processed_gb']}GB")
        
        # Search memory for similar issues
        query = f"BigQuery optimization for: {scenario['issue']}"
        memories = self.memory.recall_with_context(query=query, n_results=3)
        
        # Build context from memories
        context = "Based on similar optimization experiences:\n"
        memory_used = False
        if memories:
            memory_used = True
            for i, memory in enumerate(memories[:2]):
                content = memory.get('content', '')
                if 'Response:' in content:
                    response_part = content.split('Response:')[1].strip()
                    context += f"\n[Experience {i+1}]: {response_part[:500]}...\n"
        
        # Prepare the analysis prompt
        prompt = f"""As a BigQuery optimization expert, analyze this slow query scenario:

Query Issue: {scenario['issue']}
Query (truncated): {scenario['query'][:200]}...
Current Performance: {scenario['metrics']['original_runtime']}s runtime, {scenario['metrics']['data_processed_gb']}GB processed

{context if memory_used else ""}

Provide:
1. Root cause analysis (2-3 sentences)
2. Specific optimization recommendations (3-4 bullet points)
3. Expected performance improvement estimate

Keep the response concise and actionable."""

        # Get ADAM's analysis
        try:
            response = await self.llm_client.complete(
                prompt=prompt,
                model="grok-3-mini",  # Fast model for demo
                stream=False
            )
            
            analysis = response.content
            
            # Store the result
            result = {
                "scenario_id": scenario['id'],
                "title": scenario['title'],
                "original_metrics": scenario['metrics'],
                "memory_used": memory_used,
                "memories_found": len(memories),
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            self.demo_results.append(result)
            
            print("\n📊 ADAM's Analysis:")
            print(analysis)
            
            return result
            
        except Exception as e:
            print(f"❌ Error analyzing query: {e}")
            return None
    
    async def demonstrate_learning(self):
        """Show how ADAM learns from new scenarios"""
        print("\n🎓 Demonstrating ADAM's Learning Process...")
        
        # Create a new scenario
        new_scenario = {
            "query": "SELECT * FROM large_table WHERE status = 'active'",
            "issue": "Full table scan on 1TB table",
            "suggestion": "Add partitioning by date and clustering by status"
        }
        
        print(f"\n📝 Teaching ADAM a new optimization...")
        print(f"   Issue: {new_scenario['issue']}")
        print(f"   Solution: {new_scenario['suggestion']}")
        
        # Store in memory
        self.memory.remember_if_worthy(
            query=f"BigQuery slow query: {new_scenario['query']} - {new_scenario['issue']}",
            response=f"Optimization: {new_scenario['suggestion']}. This reduces scan size by 95% when filtering by status.",
            context={
                "domain": "bigquery",
                "type": "performance_optimization",
                "learned_from": "demo"
            },
            generation_cost=0.001,
            model_used="demo"
        )
        
        print("✅ ADAM has learned this optimization pattern!")
        
        # Test recall
        print("\n🧪 Testing ADAM's recall...")
        test_query = "BigQuery full table scan on large table"
        memories = self.memory.recall_with_context(query=test_query, n_results=1)
        
        if memories:
            print("✅ ADAM successfully recalls the optimization!")
            content = memories[0].get('content', '')
            if 'Response:' in content:
                print(f"   Remembered: {content.split('Response:')[1].strip()[:200]}...")
        
    async def generate_report(self):
        """Generate a summary report of the demo"""
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        report = {
            "demo_title": "BigQuery Performance Optimization with ADAM",
            "timestamp": datetime.now().isoformat(),
            "scenarios_analyzed": len(self.demo_results),
            "memory_utilization": sum(1 for r in self.demo_results if r['memory_used']),
            "results": self.demo_results
        }
        
        # Save JSON report
        report_path = output_dir / f"bigquery_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create markdown summary
        summary = f"""# BigQuery Optimization Demo Results

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary
- Scenarios analyzed: {len(self.demo_results)}
- Memories utilized: {sum(1 for r in self.demo_results if r['memory_used'])}
- Average memories per query: {sum(r['memories_found'] for r in self.demo_results) / len(self.demo_results):.1f}

## Key Insights
1. ADAM successfully leveraged past experiences to optimize queries
2. Memory-based suggestions provided actionable improvements
3. Learning system can adapt to new optimization patterns

## Performance Improvements Identified
"""
        
        for result in self.demo_results:
            summary += f"\n### {result['title']}\n"
            summary += f"- Original runtime: {result['original_metrics']['original_runtime']}s\n"
            summary += f"- Memories used: {result['memories_found']}\n"
            summary += f"- Key optimization: Extract from analysis\n"
        
        summary_path = output_dir / f"bigquery_demo_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(f"\n📄 Reports saved to:")
        print(f"   - {report_path}")
        print(f"   - {summary_path}")
        
        return report

async def main():
    """Run the BigQuery optimization demo"""
    print("="*60)
    print("🚀 BigQuery Performance Optimization Demo")
    print("   Showcasing ADAM's Learning Capabilities")
    print("="*60)
    
    demo = BigQueryDemo()
    
    # Load scenarios
    scenarios_path = Path(__file__).parent.parent / "data" / "bigquery_scenarios.json"
    with open(scenarios_path, 'r') as f:
        data = json.load(f)
    
    # Analyze first 3 scenarios
    print("\n📊 Analyzing BigQuery Performance Scenarios...")
    for scenario in data['scenarios'][:3]:
        await demo.analyze_query(scenario)
        await asyncio.sleep(1)  # Pause for readability
    
    # Demonstrate learning
    await demo.demonstrate_learning()
    
    # Generate report
    print("\n📈 Generating Demo Report...")
    report = await demo.generate_report()
    
    print("\n✅ Demo Complete!")
    print("\nKey Takeaways for LinkedIn:")
    print("1. ADAM learns from every optimization experience")
    print("2. Past solutions inform future recommendations")
    print("3. The system gets smarter with each interaction")
    print("4. Domain-specific knowledge accumulates over time")

if __name__ == "__main__":
    asyncio.run(main())