#!/usr/bin/env python3
"""
ADAM Chat - Real-world conversational interface for testing and using ADAM
"""
import asyncio
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Set tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv(override=True)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import ADAM components
from src.adam.memory import ADAMMemoryAdvanced
from src.adam.conversation_system import ConversationSystem
from src.adam.llm.client import UnifiedLLMClient
from src.adam.llm.config import LLMConfig

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
    """Interactive chat interface for ADAM with real LLM support"""
    
    def __init__(self, stream_output=True):
        print("Initializing ADAM...")
        
        # Initialize memory system
        self.memory = ADAMMemoryAdvanced()
        
        # Initialize conversation system
        self.conversation = ConversationSystem()
        self.session_id = self.conversation.start_session("Real World Testing")
        
        # Initialize LLM client with available models
        self.llm_config = LLMConfig()
        self.llm_client = UnifiedLLMClient(self.llm_config)
        
        # Check available models
        available_models = self.llm_config.get_available_models()
        if not available_models:
            print("\n⚠️  No API keys found! Please set XAI_API_KEY or OPENAI_API_KEY")
            sys.exit(1)
        
        print(f"Available models: {', '.join(available_models)}")
        
        # Track conversation
        self.last_model = None
        self.last_cost = 0.0
        self.total_cost = 0.0
        self.stream_output = stream_output
        self.first_interaction = True  # Track if this is the first interaction
        
        print("ADAM is ready for real-world conversations!\n")
    
    async def process_message(self, query: str) -> str:
        """Process a message through ADAM"""
        try:
            # Search memory for relevant context
            # For specific queries about past conversations, search more broadly
            if any(word in query.lower() for word in ['last', 'previous', 'query', 'sql', 'apple']):
                # Search for more results to find the specific memory
                memories = self.memory.recall_with_context(
                    query=query,
                    n_results=10  # Get more results
                )
                # Filter for SQL-related memories if asking about queries
                if 'query' in query.lower() or 'sql' in query.lower():
                    sql_memories = [m for m in memories if 'select' in m.get('content', '').lower() or 'sql' in m.get('content', '').lower()]
                    if sql_memories:
                        memories = sql_memories[:3]  # Use top 3 SQL-related memories
            else:
                memories = self.memory.recall_with_context(
                    query=query,
                    n_results=3
                )
            
            # Build context from memories
            context = ""
            if memories:
                context = "Here's what I found in my memory:\n"
                for i, memory in enumerate(memories[:3]):  # Limit to top 3
                    # Each memory has 'content' field with the full response
                    content = memory.get('content', '')
                    metadata = memory.get('metadata', {})
                    query_text = metadata.get('query_text', '')
                    
                    if query_text:
                        context += f"\n[Previous Conversation {i+1}]\n"
                        context += f"User asked: '{query_text}'\n"
                        context += f"ADAM responded: {content[:500]}\n"
                        if len(content) > 500:
                            context += "...(truncated)\n"
                    else:
                        context += f"\n[Memory {i+1}]: {content[:300]}...\n"
            
            # Use intelligent model routing
            analysis = self.llm_client.analyze_query(query)
            
            # Log model selection for transparency
            if RICH_AVAILABLE:
                console.print(f"[dim]Analyzing query... Complexity: {analysis['complexity']} | Selected: {analysis['recommended_model']}[/dim]")
            else:
                print(f"[Query complexity: {analysis['complexity']}, Using: {analysis['recommended_model']}]")
            
            # Model will be auto-selected by the client
            model = None  # Let the client auto-select
            self.last_model = analysis['recommended_model']
            
            # Check if user is asking about previous conversations
            is_memory_query = any(keyword in query.lower() for keyword in [
                'last conversation', 'previous conversation', 'what we talked',
                'last chat', 'previous chat', 'remember', 'recall', 'memory'
            ])
            
            # Build prompt with context
            # Add instruction to not introduce yourself if not first interaction
            intro_instruction = "" if self.first_interaction else "\nIMPORTANT: Do not introduce yourself. The user already knows who you are. Just answer their question directly."
            
            if context and is_memory_query:
                # Special prompt for memory queries
                prompt = f"""You are ADAM, an AI assistant with perfect memory. The user is asking about previous conversations.{intro_instruction}

{context}

User's current question: {query}

Based on the memory context above, provide a helpful summary of what was discussed in the previous conversation(s). Be specific about the details you recall."""
            elif context:
                # Regular prompt with context
                prompt = f"""You are ADAM, an AI assistant with memory of previous conversations.{intro_instruction}

Relevant context from memory:
{context}

Current question: {query}

Please answer the current question, using the context if relevant."""
            else:
                if self.first_interaction:
                    prompt = query
                else:
                    prompt = f"You are ADAM. Do not introduce yourself, just answer directly.\n\nUser's question: {query}"
            
            # Generate response with streaming
            if self.stream_output:
                # Stream the response
                response_chunks = []
                if RICH_AVAILABLE:
                    console.print(f"[bold green]ADAM ({self.last_model}):[/bold green] ", end="")
                else:
                    print(f"\nADAM ({self.last_model}): ", end="", flush=True)
                
                async for chunk in await self.llm_client.complete(
                    prompt=prompt,
                    model=model,
                    stream=True
                    # reasoning_effort will be auto-determined
                ):
                    response_chunks.append(chunk)
                    if RICH_AVAILABLE:
                        console.print(chunk, end="")
                    else:
                        print(chunk, end="", flush=True)
                
                print()  # New line after response
                response = ''.join(response_chunks)
                # Estimate cost for streaming (actual cost not available in chunks)
                self.last_cost = len(prompt + response) / 1000 * 0.001
                self.total_cost += self.last_cost
            else:
                # Non-streaming response
                llm_response = await self.llm_client.complete(
                    prompt=prompt,
                    model=model,
                    stream=False
                    # reasoning_effort will be auto-determined
                )
                response = llm_response.content
                self.last_cost = llm_response.cost
                self.total_cost += self.last_cost
            
            # Extract topics from query (simple implementation)
            topics = []
            if 'query' in query.lower() or 'sql' in query.lower():
                topics.append('sql')
            if 'debug' in query.lower():
                topics.append('debugging')
            if not topics:
                topics = ['general']
            
            # Record the exchange in conversation history
            self.conversation.record_exchange(
                query=query,
                response=response,
                topics=topics,
                context={
                    "model": model,
                    "cost": self.last_cost,
                    "memory_count": len(memories)
                }
            )
            
            # Store in memory if valuable
            if len(response) > 50:  # Simple heuristic for valuable responses
                self.memory.remember_if_worthy(
                    query=query,
                    response=response,
                    context={"session_id": self.session_id, "model": model},
                    generation_cost=0.001  # Estimate
                )
            
            # Mark that we've had at least one interaction
            self.first_interaction = False
            
            return response
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\nDebug info: {error_msg}")
            
            # Try a simpler model or provide helpful error
            if "rate limit" in str(e).lower():
                return "I'm experiencing rate limits. Please wait a moment and try again."
            elif "api key" in str(e).lower():
                return f"API key issue detected. Please check your {model} API key in .env file."
            else:
                return "I encountered an error. Please try rephrasing your question."
    
    def display_response(self, response: str):
        """Display response with optional rich formatting (only for non-streaming)"""
        if not self.stream_output:
            if RICH_AVAILABLE:
                console.print(Panel(
                    Markdown(response),
                    title=f"[bold green]ADAM[/bold green] ({self.last_model})",
                    border_style="green"
                ))
            else:
                print(f"\nADAM ({self.last_model}): {response}")
        
        # Always show cost info
        if RICH_AVAILABLE:
            console.print(
                f"[dim]Model: {self.last_model} | Cost: ${self.last_cost:.4f} | Total: ${self.total_cost:.4f}[/dim]\n"
            )
        else:
            print(f"[Cost: ${self.last_cost:.4f}, Total: ${self.total_cost:.4f}]\n")
    
    def show_memory_stats(self):
        """Show memory system statistics"""
        stats = self.memory.get_statistics()
        print("\nMemory Statistics:")
        print(f"  Total memories: {stats.get('total_memories', 0)}")
        print(f"  Sessions: {stats.get('total_sessions', 0)}")
        print(f"  Topics: {len(stats.get('topics', []))}")
        if stats.get('topics'):
            print(f"  Top topics: {', '.join(stats['topics'][:5])}")
        print()
    
    async def run(self):
        """Main chat loop"""
        print("="*60)
        print("ADAM - Advanced Data Analytics Model")
        print("Your AI assistant with perfect memory")
        print("="*60)
        print("\nCommands:")
        print("  'exit' or 'quit' - End conversation")
        print("  'cost' - Show cost breakdown")
        print("  'memory' - Show memory statistics")
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
                    # End the conversation session
                    self.conversation.end_session()
                    break
                
                elif query.lower() == 'cost':
                    print(f"\nTotal cost this session: ${self.total_cost:.4f}")
                    continue
                
                elif query.lower() == 'memory':
                    self.show_memory_stats()
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

Intelligent Model Selection:
- grok-4-reasoning: Complex tasks (code generation, deep analysis)
- grok-4: Medium complexity queries
- grok-3-mini-high: Simple questions, memory recaps
ADAM automatically selects the best model for your query!

ADAM remembers all your conversations and learns from them!
                    """
                    print(help_text)
                    continue
                
                # Process the message
                response = await self.process_message(query)
                
                # Display the response (only if not already streamed)
                if self.stream_output:
                    # Response was already displayed during streaming
                    # Just show cost info
                    if RICH_AVAILABLE:
                        console.print(
                            f"[dim]Model: {self.last_model} | Cost: ${self.last_cost:.4f} | Total: ${self.total_cost:.4f}[/dim]\n"
                        )
                    else:
                        print(f"[Cost: ${self.last_cost:.4f}, Total: ${self.total_cost:.4f}]\n")
                else:
                    # Display the full response
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
    # Check for at least one API key
    has_grok = bool(os.getenv("XAI_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    if not (has_grok or has_openai):
        print("Error: No API keys found!")
        print("\nPlease set at least one of these in your .env file:")
        print("  XAI_API_KEY=your_xai_api_key_here")
        print("  OPENAI_API_KEY=your_openai_api_key_here")
        sys.exit(1)
    
    # Run the chat interface
    asyncio.run(main())