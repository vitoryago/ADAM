#!/usr/bin/env python3
"""
ADAM Chat - Real-world conversational interface for testing and using ADAM
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.adam.integrated_conversation_system import IntegratedADAMSystem
from src.adam.conversation_system import ConversationSystem

# For colored output (optional)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Install 'rich' for better formatting: pip install rich")


class ADAMChat:
    """Interactive chat interface for ADAM"""
    
    def __init__(self):
        print("Initializing ADAM...")
        self.adam = IntegratedADAMSystem()
        self.conversation = ConversationSystem()
        self.session_id = self.conversation.start_session("Real World Testing")
        self.last_model = None
        self.last_cost = 0.0
        print("ADAM is ready for real-world conversations!\n")
    
    async def process_message(self, query: str) -> str:
        """Process a message through ADAM's integrated system"""
        try:
            # Process through the LangGraph system
            result = await self.adam.process_query_with_langgraph(
                query=query,
                session_id=self.session_id
            )
            
            # Extract response and metadata
            response = result.get("response", "I couldn't process that query.")
            self.last_model = result.get("selected_model", "unknown")
            self.last_cost = result.get("cost", 0.0)
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"\nDebug info: {error_msg}")
            return "I encountered an error processing your request. Please try again."
    
    def display_response(self, response: str):
        """Display response with optional rich formatting"""
        if RICH_AVAILABLE:
            console.print(Panel(
                Markdown(response),
                title="[bold green]ADAM[/bold green]",
                border_style="green"
            ))
            console.print(
                f"[dim]Model: {self.last_model} | Cost: ${self.last_cost:.4f}[/dim]\n"
            )
        else:
            print(f"\nADAM: {response}")
            print(f"[Model: {self.last_model}, Cost: ${self.last_cost:.4f}]\n")
    
    def get_cost_summary(self):
        """Get cost summary from ADAM"""
        try:
            return self.adam.get_cost_report()
        except:
            return {"total": self.last_cost}
    
    async def run(self):
        """Main chat loop"""
        print("="*60)
        print("ADAM - Advanced Data Analytics Model")
        print("Your AI-powered assistant with perfect memory")
        print("="*60)
        print("\nCommands:")
        print("  'exit' or 'quit' - End conversation")
        print("  'cost' - Show cost breakdown")
        print("  'help' - Show available features")
        print("\n")
        
        while True:
            try:
                # Get user input
                if RICH_AVAILABLE:
                    query = console.input("[bold blue]You:[/bold blue] ")
                else:
                    query = input("You: ")
                
                # Handle special commands
                if query.lower() in ['exit', 'quit']:
                    print("\nGoodbye! Your conversation has been saved.")
                    break
                
                elif query.lower() == 'cost':
                    costs = self.get_cost_summary()
                    print("\nCost Summary:")
                    for model, cost in costs.items():
                        print(f"  {model}: ${cost:.4f}")
                    print(f"  Total: ${sum(costs.values()):.4f}\n")
                    continue
                
                elif query.lower() == 'help':
                    help_text = """
ADAM can help you with:
- SQL query optimization and debugging
- dbt model development and troubleshooting  
- Data analysis and insights
- Code generation and refactoring
- Technical documentation
- Learning new concepts

ADAM remembers all your conversations and learns from them!
                    """
                    print(help_text)
                    continue
                
                # Process the message
                response = await self.process_message(query)
                
                # Display the response
                self.display_response(response)
                
            except KeyboardInterrupt:
                print("\n\nUse 'exit' to quit properly.")
                continue
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                print("Please try again.\n")


async def main():
    """Main entry point"""
    chat = ADAMChat()
    await chat.run()


if __name__ == "__main__":
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        sys.exit(1)
    
    # Run the chat interface
    asyncio.run(main())