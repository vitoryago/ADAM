#!/usr/bin/env python3
"""
Test the web interface's ability to retrieve the DAG with enhanced memory search
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from src.adam.memory import ADAMMemoryAdvanced
from src.adam.conversation_system import ConversationSystem
from src.adam.llm.client import UnifiedLLMClient
from src.adam.llm.config import LLMConfig
from src.adam.memory_search_enhanced import MemorySearchEnhancer, format_memory_for_prompt
from rich.console import Console

console = Console()

async def test_web_memory_retrieval():
    """Test what the web interface would retrieve"""
    
    console.print("\n[yellow]Testing Web Interface Memory Retrieval[/yellow]")
    console.print("=" * 60)
    
    # Initialize components like the web interface does
    memory = ADAMMemoryAdvanced()
    conversation = ConversationSystem()
    llm_config = LLMConfig()
    llm_client = UnifiedLLMClient(llm_config)
    memory_enhancer = MemorySearchEnhancer()
    
    # Test query
    prompt = "bring back the last dag we created together for new_fee_repricing_user"
    
    # Mock conversation history
    messages = [
        {"role": "user", "content": "I need to create a dag for fee repricing"},
        {"role": "assistant", "content": "I'll help you create that DAG"},
        {"role": "user", "content": prompt}
    ]
    
    console.print(f"\n[cyan]User query: '{prompt}'[/cyan]")
    
    # Build enhanced query like the web interface does
    enhanced_query = prompt
    conversation_context = ""
    if messages:
        recent_messages = messages[-6:]
        if len(recent_messages) > 0:
            conversation_context = "Current conversation:\n"
            for msg in recent_messages:
                role = "Human" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content'][:200]}...\n"
            conversation_context += "\n"
            enhanced_query = f"{prompt} {conversation_context[:500]}"
    
    # Get initial memories
    raw_memories = memory.recall_with_context(
        query=enhanced_query,
        n_results=10
    )
    
    console.print(f"\nFound {len(raw_memories)} raw memories")
    
    # Enhance and filter memories
    memory_context = ""
    if raw_memories:
        enhanced_memories, search_context = memory_enhancer.enhance_memory_search(
            query=prompt,
            conversation_history=messages,
            raw_memories=raw_memories
        )
        
        console.print(f"Intent: {search_context.user_intent}")
        console.print(f"Technical terms: {search_context.technical_terms}")
        console.print(f"Enhanced to {len(enhanced_memories)} memories")
        
        # Build memory context like the web interface
        if enhanced_memories:
            memory_context = "\n📚 Relevant from your memory:\n"
            for i, memory in enumerate(enhanced_memories[:3]):
                console.print(f"\n[green]Memory {i+1}:[/green]")
                # Memory ID might be in metadata
                memory_id = memory.get('memory_id') or memory.get('metadata', {}).get('memory_id', 'unknown')
                console.print(f"ID: {memory_id}")
                console.print(f"Score: {memory.get('relevance_score', 0):.3f}")
                
                # Check content
                content = memory.get('content', '')
                has_new_fee = 'new_fee_repricing_user' in content
                has_marketing = 'MARKETING_ANALYTICS' in content
                has_dbt = 'DbtOperator' in content
                
                console.print(f"Has new_fee_repricing_user: {'✓' if has_new_fee else '✗'}")
                console.print(f"Has MARKETING_ANALYTICS: {'✓' if has_marketing else '✗'}")
                console.print(f"Has DbtOperator: {'✓' if has_dbt else '✗'}")
                
                if has_new_fee and has_marketing and has_dbt:
                    console.print("[green]✓ THIS IS THE CORRECT DAG![/green]")
                
                formatted = format_memory_for_prompt(memory, search_context)
                memory_context += f"\n{formatted}\n"
                memory_context += "-" * 50 + "\n"
    
    # Show what would be sent to the LLM
    console.print("\n" + "=" * 60)
    console.print("[yellow]Memory context that would be sent to LLM:[/yellow]")
    if memory_context:
        console.print(memory_context[:1000] + "..." if len(memory_context) > 1000 else memory_context)
    else:
        console.print("[red]No memory context would be sent![/red]")
    
    # Check if the context contains the right DAG
    if 'MARKETING_ANALYTICS' in memory_context and 'new_fee_repricing_user' in memory_context:
        console.print("\n[green]✓ SUCCESS: The correct DAG would be in the LLM context![/green]")
    else:
        console.print("\n[red]✗ FAILURE: The correct DAG would NOT be in the LLM context![/red]")

if __name__ == "__main__":
    asyncio.run(test_web_memory_retrieval())