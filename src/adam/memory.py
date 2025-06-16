#!/usr/bin/env python3
"""
ADAM - Advanced Data Analytics Model
Now with mathematical vector memory!
"""

import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the standard libraries
from langchain.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pyttsx3
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from dotenv import load_dotenv

# Import ADAM's memory system
try:
    from adam.memory import ADAMMemory
except ImportError:
    console = Console()
    console.print("[red]Memory system not found. Please ensure adam/memory.py exists.[/red]")
    sys.exit(1)

load_dotenv()
console = Console()


class ADAM:
    """ADAM with integrated vector memory system"""
    
    def __init__(self):
        console.print("[yellow]Initializing ADAM with vector memory system...[/yellow]")
        
        # Initialize the brain (LLM)
        self.llm = Ollama(
            model="mistral",
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0.7
        )
        
        # Initialize voice
        self.voice_engine = pyttsx3.init()
        self.voice_engine.setProperty('rate', int(os.getenv('ADAM_VOICE_SPEED', 180)))
        
        # Initialize memory system
        try:
            self.memory = ADAMMemory()
            self.memory_available = True
            console.print("[green]✅ Vector memory system online![/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Memory system initialization failed: {e}[/yellow]")
            self.memory = None
            self.memory_available = False
        
        # Track conversation context
        self.conversation_history = []
        
        # Set personality with mathematical awareness
        self.personality = """You are ADAM (Advanced Data Analytics Model), an AI assistant 
        specifically designed to help analytics engineers. You have a vector-based memory system
        that allows you to remember and learn from past conversations. You understand the mathematics
        of embeddings and can explain technical concepts clearly. You're knowledgeable about SQL, 
        dbt, data modeling, and analytics best practices."""
        
        console.print("[green]✅ ADAM ready with full mathematical memory![/green]")
    
    def speak(self, text):
        """Convert text to speech"""
        if len(text) < 200:  # Only speak short responses
            self.voice_engine.say(text)
            self.voice_engine.runAndWait()
    
    def think(self, prompt, use_memory=True):
        """
        Process a thought, checking memory first for efficiency.
        This demonstrates the cost-saving aspect of RAG!
        """
        # Step 1: Check memory if available
        context = ""
        memory_used = False
        
        if use_memory and self.memory_available:
            # Search for relevant memories
            memories = self.memory.recall(prompt, n_results=2)
            
            if memories and memories[0]['similarity'] > 0.75:
                # Found highly relevant memory!
                context = f"\n\nRelevant information from my memory:\n{memories[0]['content']}\n"
                memory_used = True
                
                # Track cost savings
                self.memory.cost_savings['total_saved'] += 0.02  # Saved a cloud API call
                self.memory.cost_savings['queries_from_memory'] += 1
                self.memory._save_cost_savings()
                
                console.print(f"\n[dim]💭 Using memory (similarity: {memories[0]['similarity']:.3f})[/dim]")
        
        # Step 2: Construct prompt with context
        full_prompt = f"{self.personality}\n{context}\nUser: {prompt}\n\nADAM:"
        
        # Step 3: Generate response
        response = self.llm.invoke(full_prompt)
        
        # Step 4: Store this interaction in memory for future use
        if self.memory_available and not memory_used:
            # Only store if we didn't use memory (to avoid duplicates)
            self.memory.remember(
                content=f"Q: {prompt}\nA: {response}",
                metadata={
                    "type": "conversation",
                    "timestamp": datetime.now().isoformat()
                },
                source="chat"
            )
        
        return response
    
    def explain_memory_search(self, query: str):
        """
        Educational function: Show how vector search works mathematically
        """
        if not self.memory_available:
            console.print("[yellow]Memory system not available[/yellow]")
            return
            
        console.print(f"\n[bold cyan]🔬 Mathematical Vector Search Demo[/bold cyan]")
        console.print(f"Query: '{query}'")
        
        # Show the query vector
        query_vector = self.memory.embedding_model.encode(query)
        console.print(f"\n[green]Query vector shape:[/green] {query_vector.shape}")
        console.print(f"[green]L2 norm:[/green] {np.linalg.norm(query_vector):.3f}")
        console.print(f"[green]First 5 dimensions:[/green] {query_vector[:5].round(3)}")
        
        # Search and show mathematical similarity
        memories = self.memory.recall(query, n_results=3)
        
        if memories:
            console.print("\n[bold]Cosine Similarities:[/bold]")
            for i, mem in enumerate(memories):
                # Get the stored vector for comparison
                stored_text = mem['content'][:50] + "..."
                similarity = mem['similarity']
                
                # Calculate the angle in degrees for intuition
                angle_radians = np.arccos(np.clip(similarity, -1, 1))
                angle_degrees = np.degrees(angle_radians)
                
                console.print(f"\n{i+1}. '{stored_text}'")
                console.print(f"   Cosine similarity: {similarity:.4f}")
                console.print(f"   Angle between vectors: {angle_degrees:.1f}°")
                console.print(f"   Distance in vector space: {1 - similarity:.4f}")
    
    def introduce(self):
        """ADAM's introduction with memory statistics"""
        intro = f"""
# 🤖 ADAM - Advanced Data Analytics Model

Hello! I'm ADAM, your AI-powered analytics assistant with vector-based memory.

## Current Status:
- **Vector Memory**: {"✅ Online" if self.memory_available else "❌ Offline"}
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Vector Database**: ChromaDB with cosine similarity search
- **Cost Savings**: ${self.memory.cost_savings['total_saved']:.2f} saved so far

## Mathematical Capabilities:
- I understand embeddings: f: Text → ℝ³⁸⁴
- I use cosine similarity: cos(θ) = (A·B) / (||A|| × ||B||)
- I can explain the geometry of meaning in high-dimensional space

## How I Can Help:
- **Write SQL** with remembered patterns
- **Debug Code** using cached solutions  
- **Learn Together** about vectors and AI
- **Save Money** by remembering expensive analyses

Let's explore data and mathematics together! 🚀
        """
        
        console.print(Panel(Markdown(intro), title="Welcome", border_style="green"))
        self.speak("Hello! I'm ADAM with mathematical vector memory. Let's learn together!")
    
    def chat(self):
        """Interactive chat with memory integration"""
        console.print("\n[green]Chat with ADAM! Special commands:[/green]")
        console.print("[dim]- 'memory status' - See memory statistics[/dim]")
        console.print("[dim]- 'explain search: [query]' - See vector search mathematics[/dim]")
        console.print("[dim]- 'exit' - End conversation[/dim]\n")
        
        while True:
            user_input = Prompt.ask("\n[blue]You[/blue]")
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                farewell = "Thanks for exploring vectors and AI with me! Keep learning!"
                console.print(f"\n[green]ADAM:[/green] {farewell}")
                self.speak(farewell)
                
                # Show cost savings on exit
                if self.memory_available:
                    self.memory.show_cost_savings()
                break
                
            elif user_input.lower() == 'memory status':
                if self.memory_available:
                    self.memory.show_cost_savings()
                    total_memories = len(self.memory.collection.get()['ids'])
                    console.print(f"\n[cyan]Total memories stored:[/cyan] {total_memories}")
                continue
                
            elif user_input.lower().startswith('explain search:'):
                query = user_input[14:].strip()
                self.explain_memory_search(query)
                continue
            
            # Normal conversation
            console.print("\n[green]ADAM:[/green] ", end="")
            self.think(user_input)
            print()  # New line after streaming response
    
    def learn_about_vectors(self):
        """Interactive lesson about vectors and embeddings"""
        console.print("\n[bold cyan]📚 Vector Mathematics Lesson[/bold cyan]\n")
        
        # Demonstrate with actual examples
        examples = [
            ("SELECT * FROM users", "SQL query"),
            ("df.groupby('user_id').sum()", "Pandas code"),
            ("I love data analysis", "Natural language"),
            ("ERROR: relation does not exist", "Error message")
        ]
        
        console.print("Let me show you how different texts become vectors:\n")
        
        vectors = []
        for text, desc in examples:
            vec = self.memory.embedding_model.encode(text)
            vectors.append(vec)
            console.print(f"[green]{desc}:[/green] '{text}'")
            console.print(f"  → Vector norm: {np.linalg.norm(vec):.3f}")
            console.print(f"  → First 3 dimensions: {vec[:3].round(3)}")
        
        # Show similarity matrix
        console.print("\n[bold]Cosine Similarity Matrix:[/bold]")
        console.print("(How similar are these texts to each other?)\n")
        
        # Calculate all pairwise similarities
        for i, (text1, desc1) in enumerate(examples):
            similarities = []
            for j, (text2, desc2) in enumerate(examples):
                sim = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
                similarities.append(f"{sim:.2f}")
            console.print(f"{desc1:15} : {' '.join(similarities)}")
        
        console.print("\n[dim]Notice how SQL and Pandas code are more similar to each other than to natural language![/dim]")


def main():
    """Main entry point"""
    adam = ADAM()
    adam.introduce()
    
    while True:
        console.print("\n[bold]What would you like to do?[/bold]")
        console.print("  1. Chat with ADAM")
        console.print("  2. Learn about vectors and embeddings")
        console.print("  3. View memory statistics")
        console.print("  4. Exit")
        
        choice = Prompt.ask("Choose", choices=["1", "2", "3", "4"], default="1")
        
        if choice == "1":
            adam.chat()
        elif choice == "2":
            adam.learn_about_vectors()
        elif choice == "3":
            if adam.memory_available:
                adam.memory.show_cost_savings()
            else:
                console.print("[yellow]Memory system not available[/yellow]")
        elif choice == "4":
            console.print("\n[green]Keep exploring the mathematics of AI! 🎓[/green]")
            break


if __name__ == "__main__":
    main()