#!/usr/bin/env python3
"""
ADAM Fresh Start - Clean interactive interface
"""
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Setup
load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adam.llm.client import UnifiedLLMClient
from adam.tools.sql_tools import SQLAnalyzer, SQLOptimizer

# For colored output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


class ADAMFresh:
    """Fresh ADAM without complex memory systems"""
    
    def __init__(self):
        self._print_header()
        
        # Initialize core components
        self._print_status("Initializing LLM client...")
        self.llm_client = UnifiedLLMClient()
        
        # Show available models
        self._show_available_models()
        
        # Initialize tools
        self._print_status("Initializing SQL tools...")
        self.sql_analyzer = SQLAnalyzer("snowflake")
        self.sql_optimizer = SQLOptimizer("snowflake")
        
        # Track usage
        self.total_cost = 0.0
        self.model_usage = {}
        self.conversation_history = []
        
        self._print_success("\n✅ ADAM is ready! Type 'help' for commands.\n")
    
    def _print_header(self):
        """Print welcome header"""
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold blue]🧠 ADAM - Analytics Data Assistant[/bold blue]\n"
                "[dim]Your AI companion for SQL and Analytics Engineering[/dim]",
                border_style="blue"
            ))
        else:
            print("""
╔════════════════════════════════════════════════════════════╗
║            🧠 ADAM - Analytics Data Assistant              ║
║      Your AI companion for SQL and Analytics Engineering   ║
╚════════════════════════════════════════════════════════════╝
            """)
    
    def _print_status(self, message: str):
        if RICH_AVAILABLE:
            console.print(f"[yellow]⏳[/yellow] {message}")
        else:
            print(f"⏳ {message}")
    
    def _print_success(self, message: str):
        if RICH_AVAILABLE:
            console.print(f"[green]{message}[/green]")
        else:
            print(message)
    
    def _print_error(self, message: str):
        if RICH_AVAILABLE:
            console.print(f"[red]❌ {message}[/red]")
        else:
            print(f"❌ {message}")
    
    def _show_available_models(self):
        """Show available LLM models"""
        models = self.llm_client.config.get_available_models()
        
        if RICH_AVAILABLE:
            table = Table(title="Available LLM Models")
            table.add_column("Model", style="cyan")
            table.add_column("Provider", style="magenta")
            table.add_column("Best For", style="green")
            
            model_uses = {
                "grok-4": "SQL & Analytics",
                "grok-3-mini": "Fast Reasoning",
                "gpt-4": "Complex Analysis",
                "gpt-3.5-turbo": "Quick Responses",
                "o4-mini-high": "Deep Reasoning"
            }
            
            for model_name in models:
                config = self.llm_client.config.get_model_config(model_name)
                use_case = model_uses.get(model_name, "General")
                table.add_row(model_name, config.provider.value, use_case)
            
            console.print(table)
        else:
            print("\n📊 Available Models:")
            for model in models:
                print(f"  - {model}")
    
    async def process_input(self, user_input: str):
        """Process user input and generate response"""
        
        # Check commands
        if user_input.lower() == 'help':
            return self._get_help()
        elif user_input.lower() == 'stats':
            return self._get_stats()
        elif user_input.lower() == 'clear':
            os.system('clear' if os.name == 'posix' else 'cls')
            return "Screen cleared!"
        elif user_input.lower() in ['exit', 'quit']:
            return None
        
        # Check if SQL
        if self._is_sql(user_input):
            return await self._handle_sql(user_input)
        
        # Regular chat
        return await self._handle_chat(user_input)
    
    def _is_sql(self, text: str) -> bool:
        """Check if text is SQL"""
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'WITH', 'FROM']
        text_upper = text.upper().strip()
        return any(text_upper.startswith(kw) for kw in sql_keywords) or text_upper.count('SELECT') + text_upper.count('FROM') >= 2
    
    async def _handle_sql(self, query: str):
        """Handle SQL query analysis"""
        print("\n🔍 Analyzing SQL query...")
        
        # Analyze
        issues, metrics = self.sql_analyzer.analyze_query(query)
        
        # Format response
        response = "📊 **SQL Analysis:**\n\n"
        response += f"Complexity: {metrics.complexity_score}/10\n"
        response += f"Issues found: {len(issues)}\n"
        
        if issues:
            response += "\n**Issues:**\n"
            for i, issue in enumerate(issues[:5], 1):
                response += f"{i}. {issue.message}\n"
                if issue.suggestion:
                    response += f"   💡 {issue.suggestion}\n"
        
        if len(issues) > 2:
            response += "\nType 'optimize' to see an optimized version."
            self._last_query = query
        
        return response
    
    async def _handle_chat(self, user_input: str):
        """Handle regular chat"""
        
        # Check for optimize command
        if user_input.lower() == 'optimize' and hasattr(self, '_last_query'):
            print("\n🚀 Optimizing query...")
            result = await self.sql_optimizer.optimize_query(self._last_query)
            return f"**Optimized Query:**\n```sql\n{result['optimized_query']}\n```\n\nEstimated improvement: {result['estimated_improvement']}"
        
        # Build context from recent conversation
        context = ""
        if self.conversation_history:
            context = "Recent conversation:\n"
            for q, a in self.conversation_history[-2:]:
                context += f"User: {q[:50]}...\n"
                context += f"Assistant: {a[:100]}...\n\n"
        
        # Create prompt
        prompt = f"""You are ADAM, an expert Analytics Engineering AI assistant.
You help with SQL, dbt, Snowflake, BigQuery, data warehouses, and analytics.
Be concise, technical, and helpful.

{context}

User: {user_input}
Assistant:"""

        # Show model selection
        print(f"\n🤖 Selecting model...")
        
        # Detect query type
        query_lower = user_input.lower()
        
        if any(word in query_lower for word in ['sql', 'query', 'snowflake', 'optimize', 'index']):
            selected_reason = "SQL/Analytics content → Using grok-4"
        elif any(word in query_lower for word in ['explain', 'why', 'debug', 'how does']):
            selected_reason = "Reasoning task → Preferring reasoning model"
        else:
            selected_reason = "General query → Using balanced model"
        
        print(f"   {selected_reason}")
        
        # Get response
        print(f"\n💭 Thinking...")
        
        try:
            response = await self.llm_client.complete(
                prompt=prompt,
                model=None,  # Auto-select
                temperature=0.7,
                max_tokens=1000
            )
            
            # Track usage
            self.total_cost += response.cost
            self.model_usage[response.model] = self.model_usage.get(response.model, 0) + 1
            
            # Add to history
            self.conversation_history.append((user_input, response.content))
            if len(self.conversation_history) > 10:
                self.conversation_history.pop(0)
            
            # Format response
            result = response.content
            result += f"\n\n[Model: {response.model} | Cost: ${response.cost:.4f} | Tokens: {response.total_tokens}]"
            
            return result
            
        except Exception as e:
            return f"Error: {str(e)}\n\nPlease check your API keys are set correctly."
    
    def _get_help(self):
        """Get help message"""
        return """
🤖 **ADAM Commands:**

**Commands:**
- `help` - Show this message
- `stats` - Show usage statistics
- `clear` - Clear the screen
- `exit/quit` - End session

**Usage:**
- Ask questions about SQL, dbt, analytics
- Paste SQL queries for automatic analysis
- Type 'optimize' after SQL analysis for improvements

**Multi-line Input:**
- SQL queries automatically trigger multi-line mode
- End any line with \\ to continue on next line
- Press Enter on empty line to finish input

**Example Questions:**
- "How do I create an incremental dbt model?"
- "What's the difference between RANK() and ROW_NUMBER()?"
- "Explain Snowflake clustering keys"
- "Debug this error: [paste error message]"

**SQL Analysis:**
Just paste any SQL query and I'll analyze it!
"""
    
    def _get_stats(self):
        """Get session statistics"""
        stats = f"""
📊 **Session Statistics:**

**Model Usage:**"""
        
        for model, count in self.model_usage.items():
            stats += f"\n- {model}: {count} calls"
        
        stats += f"\n\n**Total Cost:** ${self.total_cost:.4f}"
        stats += f"\n**Conversations:** {len(self.conversation_history)}"
        
        return stats
    
    async def run(self):
        """Run interactive session"""
        
        while True:
            try:
                # Get input
                if RICH_AVAILABLE:
                    user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
                else:
                    user_input = input("\nYou: ").strip()
                
                # Check for multi-line input (SQL or large queries)
                if user_input and (self._is_sql(user_input) or user_input.endswith('\\')):
                    # Multi-line mode
                    lines = [user_input.rstrip('\\')]
                    if RICH_AVAILABLE:
                        console.print("[dim]... (Enter empty line to finish)[/dim]")
                    else:
                        print("... (Enter empty line to finish)")
                    
                    while True:
                        try:
                            if RICH_AVAILABLE:
                                line = console.input("[dim]... [/dim]")
                            else:
                                line = input("... ")
                            
                            if not line.strip():  # Empty line ends input
                                break
                            lines.append(line)
                        except EOFError:
                            break
                    
                    user_input = '\n'.join(lines)
                
                if not user_input:
                    continue
                
                # Process
                response = await self.process_input(user_input)
                
                if response is None:  # Exit
                    print("\n📊 Final Statistics:")
                    print(self._get_stats())
                    print("\n👋 Thanks for using ADAM!")
                    break
                
                # Display response
                if RICH_AVAILABLE:
                    console.print(f"\n[bold green]ADAM:[/bold green] {response}")
                else:
                    print(f"\nADAM: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                self._print_error(f"Error: {e}")


async def main():
    """Main entry point"""
    print("Starting ADAM...\n")
    
    # Check for rich
    if not RICH_AVAILABLE:
        print("💡 Tip: Install 'rich' for better formatting: pip install rich\n")
    
    adam = ADAMFresh()
    await adam.run()


if __name__ == "__main__":
    asyncio.run(main())