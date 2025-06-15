#!/usr/bin/env python3
"""
ADAM's Memory System - Your Gateway to Understanding Vectors and RAG!

This module teaches you:
1. How text becomes vectors (embeddings)
2. How vector similarity works
3. How RAG retrieves relevant information
4. How to build cost-effective AI systems
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import hashlib

# Vector database and embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# For visualizing what we're learning
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()


class ADAMMemory:
    """
    ADAM's memory system using vectors and RAG.
    This is exactly what ChatGPT and Claude use internally!
    """
    
    def __init__(self, persist_directory: str = "./adam_memory"):
        """Initialize ADAM's vector-based memory system."""
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        console.print("[yellow]🧠 Initializing ADAM's Vector Memory System...[/yellow]")
        
        # Initialize the embedding model - this converts text to vectors!
        # Using sentence-transformers because it's free and runs locally
        console.print("[dim]Loading embedding model (this creates vectors from text)...[/dim]")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB - our vector database
        console.print("[dim]Initializing vector database (stores and searches embeddings)...[/dim]")
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection (like a table in traditional DB)
        self.collection = self.chroma_client.get_or_create_collection(
            name="adam_knowledge",
            metadata={"description": "ADAM's learned knowledge"}
        )
        
        # Track costs saved
        self.cost_savings_log = self.persist_directory / "cost_savings.json"
        self.cost_savings = self._load_cost_savings()
        
        console.print("[green]✅ Memory system ready! Let's learn about vectors![/green]\n")
    
    def _load_cost_savings(self) -> Dict[str, float]:
        """Load our cost savings tracker."""
        if self.cost_savings_log.exists():
            with open(self.cost_savings_log, 'r') as f:
                return json.load(f)
        return {"total_saved": 0.0, "queries_from_memory": 0}
    
    def _save_cost_savings(self):
        """Save our cost savings data."""
        with open(self.cost_savings_log, 'w') as f:
            json.dump(self.cost_savings, f, indent=2)
    
    def demonstrate_embeddings(self, text: str) -> np.ndarray:
        """
        Show how text becomes vectors - this is the foundation of modern AI!
        
        This is what happens inside ChatGPT when you type something.
        """
        console.print(f"\n[bold]🔍 Converting text to vector:[/bold] '{text}'")
        
        # Create embedding (vector)
        vector = self.embedding_model.encode(text)
        
        # Show the vector (first 10 dimensions)
        console.print(f"[cyan]Vector shape:[/cyan] {vector.shape} dimensions")
        console.print(f"[cyan]First 10 values:[/cyan] {vector[:10].round(3)}")
        console.print(f"[dim]Each number represents a semantic feature of your text![/dim]\n")
        
        return vector
    
    def show_similarity(self, text1: str, text2: str) -> float:
        """
        Demonstrate how vector similarity works - the core of RAG!
        
        This is how AI finds relevant information.
        """
        console.print(f"\n[bold]📊 Comparing similarity:[/bold]")
        console.print(f"Text 1: '{text1}'")
        console.print(f"Text 2: '{text2}'")
        
        # Get vectors
        vec1 = self.embedding_model.encode(text1)
        vec2 = self.embedding_model.encode(text2)
        
        # Calculate cosine similarity (most common similarity metric)
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        console.print(f"[green]Similarity score:[/green] {similarity:.3f} (0 = different, 1 = identical)")
        console.print(f"[dim]This is how AI knows 'SQL' is related to 'database' but not 'cooking'![/dim]\n")
        
        return similarity
    
    def remember(self, content: str, metadata: Optional[Dict] = None, 
                 source: str = "conversation") -> str:
        """
        Store information in vector memory - this is the 'storage' part of RAG!
        
        This is like creating a searchable memory that understands meaning.
        """
        # Create unique ID
        memory_id = hashlib.md5(f"{content}{datetime.now()}".encode()).hexdigest()[:8]
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "length": len(content)
        })
        
        # Store in vector database
        # ChromaDB automatically creates embeddings!
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[memory_id]
        )
        
        console.print(f"[green]💾 Memorized![/green] ID: {memory_id}")
        console.print(f"[dim]This content is now searchable by meaning, not just keywords![/dim]")
        
        return memory_id
    
    def recall(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search memory using vector similarity - this is the 'retrieval' part of RAG!
        
        This is exactly how ChatGPT finds relevant context for your questions.
        """
        console.print(f"\n[bold]🔎 Searching memory for:[/bold] '{query}'")
        
        # Search vector database
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Process results
        memories = []
        if results['documents'][0]:  # If we found something
            console.print(f"[green]Found {len(results['documents'][0])} relevant memories![/green]")
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                similarity = 1 - distance  # Convert distance to similarity
                memories.append({
                    'content': doc,
                    'metadata': metadata,
                    'similarity': similarity
                })
                
                console.print(f"\n[cyan]Memory {i+1}:[/cyan] (similarity: {similarity:.3f})")
                console.print(f"[dim]{doc[:100]}...[/dim]")
        else:
            console.print("[yellow]No relevant memories found.[/yellow]")
        
        return memories
    
    def smart_query(self, query: str, always_use_cloud: bool = False) -> Dict[str, Any]:
        """
        Demonstrate cost-effective querying - check memory first, then cloud.
        
        This is how production AI systems save money!
        """
        console.print(f"\n[bold]💡 Smart Query:[/bold] '{query}'")
        
        # Step 1: Check memory first (FREE!)
        memories = self.recall(query, n_results=1)
        
        if memories and memories[0]['similarity'] > 0.8 and not always_use_cloud:
            # Found good match in memory!
            console.print("[green]✅ Found in memory! Saving $0.03[/green]")
            self.cost_savings['total_saved'] += 0.03
            self.cost_savings['queries_from_memory'] += 1
            self._save_cost_savings()
            
            return {
                'response': memories[0]['content'],
                'source': 'memory',
                'cost': 0.0,
                'similarity': memories[0]['similarity']
            }
        else:
            # Need to use cloud model
            console.print("[yellow]📤 Not in memory, would use cloud model (costs $0.03)[/yellow]")
            console.print("[dim]In production, this would call GPT-4/Claude here[/dim]")
            
            return {
                'response': f"[Would call cloud API for: {query}]",
                'source': 'cloud',
                'cost': 0.03,
                'similarity': 0.0
            }
    
    def show_cost_savings(self):
        """Display how much money the memory system has saved."""
        table = Table(title="💰 ADAM's Cost Savings Report")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Saved", f"${self.cost_savings['total_saved']:.2f}")
        table.add_row("Queries from Memory", str(self.cost_savings['queries_from_memory']))
        table.add_row("Cloud Calls Avoided", str(self.cost_savings['queries_from_memory']))
        
        if self.cost_savings['queries_from_memory'] > 0:
            avg_saving = self.cost_savings['total_saved'] / self.cost_savings['queries_from_memory']
            table.add_row("Average Saving per Query", f"${avg_saving:.3f}")
        
        console.print(table)
    
    def learn_from_screen(self, screen_content: str, analysis: str, cost: float = 0.03):
        """
        Store expensive screen analysis results for future use.
        
        This is the key to making screen-based AI affordable!
        """
        # Create a searchable summary
        summary = f"Screen content: {screen_content[:200]}... Analysis: {analysis}"
        
        metadata = {
            "type": "screen_analysis",
            "original_cost": cost,
            "screen_preview": screen_content[:100]
        }
        
        memory_id = self.remember(analysis, metadata, source="screen_analysis")
        
        console.print(f"[green]📸 Cached screen analysis![/green] Future queries about this will be FREE!")
        console.print(f"[dim]Original cost: ${cost} - now cached forever[/dim]")
        
        return memory_id


def interactive_demo():
    """
    Interactive demonstration of vectors, embeddings, and RAG.
    This will teach you everything that manager was talking about!
    """
    console.print(Panel.fit(
        "[bold cyan]Welcome to ADAM's Memory System Tutorial![/bold cyan]\n\n"
        "You're about to learn:\n"
        "• How text becomes vectors (embeddings)\n"
        "• How vector similarity works\n"
        "• How RAG retrieves information\n"
        "• How to build cost-effective AI systems\n\n"
        "Let's dive in! 🚀",
        title="🧠 Vector & RAG Learning Lab"
    ))
    
    # Initialize memory
    memory = ADAMMemory()
    
    # Demo 1: Embeddings
    console.print("\n[bold magenta]Demo 1: Understanding Embeddings[/bold magenta]")
    memory.demonstrate_embeddings("SELECT * FROM users WHERE age > 18")
    
    # Demo 2: Similarity
    console.print("\n[bold magenta]Demo 2: Vector Similarity (The Magic of RAG)[/bold magenta]")
    memory.show_similarity(
        "How do I filter users by age in SQL?",
        "SELECT * FROM users WHERE age > 18"
    )
    memory.show_similarity(
        "How do I filter users by age in SQL?",
        "Python pandas DataFrame operations"
    )
    
    # Demo 3: Memory Storage
    console.print("\n[bold magenta]Demo 3: Storing Knowledge (Building Your RAG)[/bold magenta]")
    memory.remember(
        "To find users who made purchases last month but not this month, use: "
        "SELECT * FROM users WHERE user_id IN (SELECT user_id FROM purchases "
        "WHERE DATE_TRUNC('month', purchase_date) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')) "
        "AND user_id NOT IN (SELECT user_id FROM purchases WHERE DATE_TRUNC('month', purchase_date) = DATE_TRUNC('month', CURRENT_DATE))",
        metadata={"pattern": "lapsed_users", "difficulty": "intermediate"}
    )
    
    # Demo 4: Retrieval
    console.print("\n[bold magenta]Demo 4: Intelligent Retrieval (RAG in Action)[/bold magenta]")
    memory.recall("find users who stopped purchasing")
    
    # Demo 5: Cost Savings
    console.print("\n[bold magenta]Demo 5: Cost-Effective AI (The Business Case)[/bold magenta]")
    
    # Simulate some queries
    queries = [
        "find lapsed users",
        "users who haven't bought recently",
        "SELECT active users",  # This won't match
        "customers who stopped buying"
    ]
    
    for q in queries:
        result = memory.smart_query(q)
        console.print(f"Query result: {result['source']} (cost: ${result['cost']})\n")
    
    memory.show_cost_savings()
    
    console.print(Panel.fit(
        "[bold green]Congratulations! You now understand:[/bold green]\n\n"
        "✅ Embeddings: How AI converts text to meaningful numbers\n"
        "✅ Vector Similarity: How AI finds related information\n"
        "✅ RAG: Retrieval-Augmented Generation for smarter AI\n"
        "✅ Cost Optimization: Using memory to reduce API calls\n\n"
        "You can now talk intelligently about vector databases and RAG! 🎉",
        title="🏆 You're Now a RAG Expert!"
    ))


if __name__ == "__main__":
    # Run the interactive demo
    interactive_demo()