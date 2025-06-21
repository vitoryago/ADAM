#!/usr/bin/env python3
"""
ADAM - Advanced Data Analytics Model with Integrated Memory System
Your AI-powered analytics engineering assistant that learns and remembers
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from langchain.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pyttsx3
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
from dotenv import load_dotenv

# Import our advanced memory system
from src.adam.memory import ADAMMemoryAdvanced, QueryComplexity

# Load environment variables
load_dotenv()

console = Console()

class IntegratedADAM:
    """ADAM with full memory integration and intelligent response handling"""
    
    def __init__(self):
        console.print("[yellow]🚀 Initializing ADAM with Advanced Memory System...[/yellow]")
        
        # Initialize the brain (LLM)
        self.llm = Ollama(
            model="mistral",
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0.7
        )
        
        # Initialize voice
        self.voice_engine = pyttsx3.init()
        self.voice_engine.setProperty('rate', int(os.getenv('ADAM_VOICE_SPEED', 180)))
        
        # Initialize advanced memory system
        self.memory = ADAMMemoryAdvanced()
        
        # Track current conversation for context
        self.current_problem_id: Optional[str] = None
        self.conversation_history: list = []
        
        # Set personality
        self.personality = """You are ADAM (Advanced Data Analytics Model), an AI assistant 
        specifically designed to help analytics engineers at MoneyLion. You're knowledgeable about SQL, 
        dbt, data modeling, and analytics best practices. You understand financial data, user analytics,
        and product metrics. You have a helpful and encouraging tone, always explaining concepts in a way
        that's easy to understand. You don't just answer questions; you anticipate potential issues and offer
        proactive solutions. Whenever possible, you explain *why* a solution works, not just *how* to implement
        it. You speak clearly in English, Portuguese, and Spanish, always aiming to teach and improve the user's
        skills while making data analysis enjoyable. You occasionally use data-related puns, but only when appropriate.
        If a request is unclear, you ask clarifying questions before attempting to provide a solution. If you are unable
        to answer a question, you politely explain why and suggest alternative resources."""
        
        # Show memory stats on startup
        self._display_startup_stats()
        
        console.print("[green]✅ ADAM is ready with full memory integration![/green]")
    
    def _display_startup_stats(self):
        """Show memory system statistics on startup"""
        analytics = self.memory.get_memory_analytics()
        
        if analytics.get('total_memories', 0) > 0:
            console.print(f"\n[cyan]📊 Memory Stats:[/cyan]")
            console.print(f"  Total memories: {analytics['total_memories']}")
            console.print(f"  Success rate: {analytics['average_success_rate']:.1%}")
            console.print(f"  Total savings: ${analytics['total_savings']:.2f}")
            console.print(f"  ROI: {analytics['roi_percentage']:.1f}%\n")
    
    def speak(self, text):
        """Convert text to speech"""
        # Remove markdown and special formatting for speech
        clean_text = text.replace('*', '').replace('`', '').replace('#', '')
        self.voice_engine.say(clean_text)
        self.voice_engine.runAndWait()
    
    def think_with_memory(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> tuple[str, bool, float]:
        """
        Process a thought with memory integration
        Returns: (response, from_memory, estimated_cost)
        """
        # First, check if we have relevant memories
        memories = self.memory.recall_with_context(prompt, n_results=3)
        
        # Check if we have a high-confidence memory match
        if memories and memories[0]['similarity'] > 0.85:
            best_memory = memories[0]
            similarity = best_memory['similarity']
            
            # Extract the response part from the memory content
            content = best_memory['content']
            if 'Response: ' in content:
                response = content.split('Response: ', 1)[1]
            else:
                response = content
            
            # Check success rate
            success_rate = best_memory['metadata'].get('success_rate', 1.0)
            
            if similarity > 0.95 and success_rate > 0.8:
                # Very high confidence - use memory directly
                console.print(f"\n[green]💭 Retrieved from memory (similarity: {similarity:.2f}, success rate: {success_rate:.1%})[/green]")
                saved_cost = best_memory['metadata'].get('generation_cost', 0.01)
                console.print(f"[green]💰 Saved: ${saved_cost:.3f}[/green]\n")
                return response, True, 0.0
            
            elif similarity > 0.85:
                # Good match but maybe need to verify or enhance
                console.print(f"\n[yellow]💭 Found similar memory (similarity: {similarity:.2f})[/yellow]")
                console.print("[yellow]🔄 Generating fresh response to ensure accuracy...[/yellow]\n")
        
        # No good memory match - generate new response
        console.print("\n[cyan]🧠 Thinking with Mistral...[/cyan]\n")
        
        # Build context with any relevant memories
        memory_context = ""
        if memories:
            memory_context = "\n\nPotentially relevant past interactions:\n"
            for mem in memories[:2]:  # Include top 2 memories as context
                if mem['similarity'] > 0.7:
                    memory_context += f"- {mem['content'][:200]}...\n"
        
        full_prompt = f"{self.personality}\n{memory_context}\nUser: {prompt}\n\nADAM:"
        
        # Generate response
        response = self.llm.invoke(full_prompt)
        
        # Estimate generation cost (rough estimate for Mistral)
        estimated_cost = len(prompt + response) * 0.000001  # Very rough estimate
        
        # Store in memory if worthy
        memory_id = self.memory.remember_if_worthy(
            query=prompt,
            response=response,
            context=context,
            generation_cost=estimated_cost,
            model_used="mistral"
        )
        
        if memory_id:
            console.print(f"\n[green]💾 Stored in memory for future use (ID: {memory_id})[/green]")
        
        return response, False, estimated_cost
    
    def handle_feedback(self, feedback: str):
        """Process user feedback about whether a solution worked"""
        result = self.memory.handle_solution_feedback(feedback)
        
        if result['status'] == 'success':
            console.print(f"\n[green]✅ {result['message']}[/green]")
            self.current_problem_id = None
            
        elif result['status'] == 'failure':
            console.print(f"\n[yellow]🔄 {result['message']}[/yellow]")
            # Continue with the same problem ID for follow-up
            
        elif result['status'] == 'unclear':
            console.print(f"\n[cyan]❓ {result['message']}[/cyan]")
    
    def process_command(self, command: str) -> bool:
        """
        Process special commands
        Returns True if command was processed, False otherwise
        """
        command_lower = command.lower().strip()
        
        if command_lower == '/stats':
            self.memory.display_analytics_dashboard()
            return True
            
        elif command_lower == '/memories':
            self.show_recent_memories()
            return True
            
        elif command_lower == '/export':
            filename = f"adam_knowledge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.memory.export_knowledge_base(filename)
            return True
            
        elif command_lower == '/help':
            self.show_help()
            return True
            
        elif command_lower.startswith('/forget '):
            # Future: implement memory deletion
            console.print("[yellow]Memory deletion not yet implemented[/yellow]")
            return True
        
        return False
    
    def show_recent_memories(self):
        """Display recent memories"""
        # For now, show analytics - could be enhanced to show actual memories
        self.memory.display_analytics_dashboard()
    
    def show_help(self):
        """Show available commands"""
        help_table = Table(title="🛠️ ADAM Commands")
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="white")
        
        help_table.add_row("/stats", "Show memory statistics and ROI")
        help_table.add_row("/memories", "View stored memories")
        help_table.add_row("/export", "Export knowledge base to JSON")
        help_table.add_row("/help", "Show this help message")
        help_table.add_row("exit/quit/bye", "End the session")
        
        console.print(help_table)
        console.print("\n[dim]Tip: Tell me 'that didn't work' after a failed solution to help me learn![/dim]")
    
    def chat(self):
        """Interactive chat session with memory integration"""
        console.print("\n[green]Let's chat! I remember our past conversations and learn from them.[/green]")
        console.print("[dim]Commands: /help for available commands[/dim]\n")
        
        session_cost = 0.0
        queries_count = 0
        memory_hits = 0
        
        while True:
            # Get user input
            user_input = Prompt.ask("\n[blue]You[/blue]")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                # Show session stats
                console.print(f"\n[cyan]📊 Session Summary:[/cyan]")
                console.print(f"  Queries: {queries_count}")
                console.print(f"  Memory hits: {memory_hits} ({memory_hits/max(queries_count,1)*100:.1f}%)")
                console.print(f"  Estimated cost: ${session_cost:.4f}")
                
                farewell = "Great chatting with you! I've learned from our conversation and will remember the important parts."
                console.print(f"\n[green]ADAM:[/green] {farewell}")
                self.speak(farewell)
                break
            
            # Check for commands
            if user_input.startswith('/'):
                if self.process_command(user_input):
                    continue
            
            # Check for feedback about previous solutions
            feedback_keywords = ['didn\'t work', 'doesn\'t work', 'failed', 'error', 'still broken', 
                               'works', 'fixed', 'solved', 'perfect', 'thank']
            if any(keyword in user_input.lower() for keyword in feedback_keywords):
                self.handle_feedback(user_input)
                continue
            
            # Track query
            queries_count += 1
            
            # Check if this might be starting a new problem
            if any(word in user_input.lower() for word in ['error', 'bug', 'issue', 'problem', 'help with']):
                if not self.current_problem_id:
                    self.current_problem_id = self.memory.start_problem_solving(user_input)
            
            # Process with memory integration
            console.print("\n[green]ADAM:[/green] ", end="")
            response, from_memory, cost = self.think_with_memory(user_input)
            
            # Track stats
            if from_memory:
                memory_hits += 1
            session_cost += cost
            
            # Add to conversation history
            self.conversation_history.append({
                'query': user_input,
                'response': response,
                'timestamp': datetime.now(),
                'from_memory': from_memory
            })
            
            # If we generated a solution for a tracked problem, record it
            if self.current_problem_id and not from_memory:
                self.memory.add_solution_attempt(response)
            
            print()  # New line after streaming
    
    def introduce(self):
        """ADAM's introduction with memory awareness"""
        intro = f"""
# 🤖 ADAM - Advanced Data Analytics Model

Hello! I'm ADAM, your AI-powered analytics assistant with **perfect memory**.

## What Makes Me Special:
- **🧠 I Learn**: I remember successful solutions and learn from failures
- **💰 I Save Money**: I track costs and show ROI from cached responses  
- **🚀 I'm Fast**: Common questions answered from memory in milliseconds
- **📈 I Improve**: My success rate increases as we work together

## I Can Help You With:
- **SQL Queries** - From simple SELECT to complex window functions
- **dbt Models** - Best practices, macros, and testing strategies
- **Debugging** - I remember what worked (and what didn't)
- **Architecture** - Data pipeline design and optimization

## Available Commands:
- `/stats` - See memory statistics and ROI
- `/memories` - Browse stored knowledge
- `/export` - Export my knowledge base
- `/help` - Show all commands

Let's build amazing data solutions together! 🚀
        """
        
        console.print(Panel(Markdown(intro), title="Welcome", border_style="green"))
        
        # Spoken introduction
        self.speak(
            "Hello! I'm ADAM, your Advanced Data Analytics Model with perfect memory. "
            "I learn from our conversations and get smarter over time. "
            "Ask me anything about SQL, data modeling, or analytics!"
        )


def main():
    """Main entry point"""
    adam = IntegratedADAM()
    adam.introduce()
    adam.chat()


if __name__ == "__main__":
    main()