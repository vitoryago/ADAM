#!/usr/bin/env python3
"""
Diagnostic script to debug memory retrieval issues
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from src.adam.memory_search_enhanced import MemorySearchEnhancer
from rich.console import Console
from rich.table import Table
import json

console = Console()

def diagnose_memory_system():
    """Diagnose what's happening with memory storage and retrieval"""
    
    console.print("\n[bold blue]🔍 ADAM Memory System Diagnostic[/bold blue]\n")
    
    # Initialize memory system
    memory = ADAMMemoryAdvanced()
    enhancer = MemorySearchEnhancer()
    
    # 1. Check total memories
    stats = memory.get_memory_analytics()
    console.print(f"[green]Total memories in system:[/green] {stats.get('total_memories', 0)}")
    console.print(f"[green]Memory hit rate:[/green] {stats.get('memory_hit_rate', 0):.1%}")
    
    # 2. Search for DAG-related memories
    console.print("\n[yellow]Searching for DAG-related memories...[/yellow]")
    
    test_queries = [
        "dag",
        "dbt model",
        "new_fee_repricing_user", 
        "airflow dag dbt",
        "create a new dag to run a private DBT model",
        "bring the code again"
    ]
    
    all_results = []
    for query in test_queries:
        results = memory.recall_with_context(query=query, n_results=5)
        console.print(f"\nQuery: '[cyan]{query}[/cyan]' - Found {len(results)} results")
        
        for i, result in enumerate(results[:2]):  # Show top 2
            content = result.get('content', '')
            similarity = result.get('similarity', 0)
            console.print(f"  [{i+1}] Similarity: {similarity:.3f}")
            console.print(f"      Content preview: {content[:150]}...")
            
            # Check if this is the DAG conversation
            if 'new_fee_repricing_user' in content or 'dbt_private_model_runner' in content:
                console.print(f"      [green]✓ This looks like the DAG conversation![/green]")
                all_results.append((query, result))
    
    # 3. Test intent detection
    console.print("\n[yellow]Testing intent detection...[/yellow]")
    
    test_intents = [
        "Hi we were talking about some dag, do you remember?",
        "can you bring the code again?",
        "show me the last DAG we created together"
    ]
    
    for phrase in test_intents:
        intent = enhancer.analyze_user_intent(phrase)
        console.print(f"Phrase: '[cyan]{phrase}[/cyan]'")
        console.print(f"Detected intent: [magenta]{intent}[/magenta]")
    
    # 4. Check memory storage logs
    console.print("\n[yellow]Checking recent memory operations...[/yellow]")
    
    # Look for the specific content in all memories
    all_memories = memory.collection.get()
    if all_memories and 'documents' in all_memories:
        dag_memories = []
        for i, doc in enumerate(all_memories['documents']):
            if any(term in doc.lower() for term in ['new_fee_repricing', 'dbt_private', 'instacash']):
                dag_memories.append({
                    'id': all_memories['ids'][i],
                    'content': doc[:200],
                    'metadata': all_memories['metadatas'][i]
                })
        
        if dag_memories:
            console.print(f"\n[green]Found {len(dag_memories)} DAG-related memories![/green]")
            for mem in dag_memories:
                console.print(f"\nMemory ID: {mem['id']}")
                console.print(f"Content: {mem['content']}...")
                console.print(f"Metadata: {json.dumps(mem['metadata'], indent=2)}")
        else:
            console.print("\n[red]No DAG-related memories found in storage![/red]")
            console.print("This suggests the conversation was never stored in memory.")
    
    # 5. Test enhanced search
    console.print("\n[yellow]Testing enhanced memory search...[/yellow]")
    
    # Simulate conversation history
    mock_conversation = [
        {"role": "user", "content": "I need to create a new dag to run a private DBT model"},
        {"role": "assistant", "content": "I'll help you create a DAG for your private DBT model..."},
        {"role": "user", "content": "can you bring the code again?"}
    ]
    
    # Get raw memories
    raw_memories = memory.recall_with_context(
        query="bring the code again dag dbt",
        n_results=10
    )
    
    if raw_memories:
        enhanced_memories, context = enhancer.enhance_memory_search(
            query="can you bring the code again?",
            conversation_history=mock_conversation,
            raw_memories=raw_memories
        )
        
        console.print(f"\nEnhanced search context:")
        console.print(f"  Intent: {context.user_intent}")
        console.print(f"  Technical terms: {context.technical_terms}")
        console.print(f"  Enhanced memories: {len(enhanced_memories)}")
    
    # 6. Recommendations
    console.print("\n[bold red]Diagnostic Summary:[/bold red]")
    
    if not all_results and not dag_memories:
        console.print("❌ The DAG conversation was likely never stored in memory")
        console.print("   Possible reasons:")
        console.print("   - Memory worthiness evaluator rejected it")
        console.print("   - Error during storage")
        console.print("   - Session ended before memory storage")
    else:
        console.print("✅ DAG memories found but retrieval is failing")
        console.print("   Possible reasons:")
        console.print("   - Query enhancement not working properly")
        console.print("   - Relevance scoring too strict")
        console.print("   - Memory context not being used by LLM")

if __name__ == "__main__":
    diagnose_memory_system()