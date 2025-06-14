#!/usr/bin/env python3
"""
ADAM - Advanced Data Analytics Model
Your AI-powered analytics engineering assistant
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from langchain.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pyttsx3
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

console = Console()

class ADAM:
    """ADAM - Your Advanced Data Analytics Model assistant"""
    
    def __init__(self):
        console.print("[yellow]Initializing ADAM's neural pathways...[/yellow]")
        
        # Initialize the brain (LLM)
        self.llm = Ollama(
            model="mistral",
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0.7
        )
        
        # Initialize voice
        self.voice_engine = pyttsx3.init()
        self.voice_engine.setProperty('rate', int(os.getenv('ADAM_VOICE_SPEED', 180)))
        
        # Set personality
        self.personality = """You are ADAM (Advanced Data Analytics Model), an AI assistant 
        specifically designed to help analytics engineers. You're knowledgeable about SQL, 
        dbt, data modeling, and analytics best practices. You speak clearly and helpfully,
        always aiming to teach and improve the user's skills."""
        
        console.print("[green]✅ ADAM is ready![/green]")
    
    def speak(self, text):
        """Convert text to speech"""
        self.voice_engine.say(text)
        self.voice_engine.runAndWait()
    
    def think(self, prompt):
        """Process a thought with full context"""
        full_prompt = f"{self.personality}\n\nUser: {prompt}\n\nADAM:"
        return self.llm.invoke(full_prompt)
    
    def introduce(self):
        """ADAM's introduction"""
        intro = """
# 🤖 ADAM - Advanced Data Analytics Model

Hello! I'm ADAM, your AI-powered analytics assistant.

I'm here to help you:
- **Write better SQL** - From simple queries to complex analytics
- **Master dbt** - Model development, testing, and best practices  
- **Debug faster** - Understand errors and find solutions
- **Learn continuously** - Improve your skills with every interaction

I can communicate in English, Portuguese, and Spanish. Just speak naturally!

Let's build amazing data solutions together! 🚀
        """
        
        console.print(Panel(Markdown(intro), title="Welcome", border_style="green"))
        
        # Spoken introduction
        self.speak(
            "Hello! I'm ADAM, your Advanced Data Analytics Model. "
            "I'm excited to help you become a more effective analytics engineer. "
            "Ask me anything about SQL, data modeling, or analytics!"
        )
    
    def chat(self):
        """Interactive chat session"""
        console.print("\n[green]Let's chat! Type 'exit' to end our session.[/green]\n")
        
        while True:
            # Get user input
            user_input = Prompt.ask("\n[blue]You[/blue]")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                farewell = "Great chatting with you! Remember, every query you write makes you a better engineer. See you soon!"
                console.print(f"\n[green]ADAM:[/green] {farewell}")
                self.speak(farewell)
                break
            
            # Process and respond
            console.print("\n[green]ADAM:[/green] ", end="")
            self.think(user_input)
            print()  # New line after streaming
    
    def demo_sql_help(self):
        """Demonstrate SQL assistance capabilities"""
        console.print("\n[yellow]SQL Assistance Demo[/yellow]")
        demo_prompt = """
        The user needs help writing a SQL query to find customers who made 
        purchases in the last 30 days but not in the previous 30 days (new reactivated customers).
        Provide the SQL and explain the logic.
        """
        
        console.print("\n[green]ADAM:[/green] ", end="")
        self.think(demo_prompt)
        print()

def main():
    """Main entry point"""
    adam = ADAM()
    adam.introduce()
    
    # Optional: Show a demo
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        adam.demo_sql_help()
    
    adam.chat()

if __name__ == "__main__":
    main()
