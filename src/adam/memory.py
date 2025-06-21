#!/usr/bin/env python3
"""
ADAM's Advanced Memory System - Complete Rewrite
Incorporates intelligent storage decisions, memory updates, and context awareness

This system implements:
1. Selective memory storage based on value assessment
2. Memory correction and versioning when solutions fail  
3. Context-aware retrieval that considers screen state
4. Automatic caching of expensive model outputs
5. Conversation state tracking for multi-turn problem solving
"""

import os
import json
import hashlib
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

# Vector and storage components
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# For conversation state management
from collections import deque

# Rich output for better visibility
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class MemoryType(Enum):
    """Types of memories for different handling strategies"""
    ERROR_SOLUTION = "error_solution"
    CODE_PATTERN = "code_pattern"
    CONCEPT_EXPLANATION = "concept_explanation"
    SCREEN_ANALYSIS = "screen_analysis"
    EXPENSIVE_RESPONSE = "expensive_response"
    CONVERSATION = "conversation"
    FACTUAL = "factual"  # Simple facts that might not need storage


class QueryComplexity(Enum):
    """Assess query complexity to decide storage and model routing"""
    TRIVIAL = 1      # "What day is it?"
    SIMPLE = 2       # "How do I print in Python?"
    MODERATE = 3     # "Explain list comprehensions"
    COMPLEX = 4      # "Debug this error in my code"
    EXPERT = 5       # "Design a distributed system for..."


@dataclass
class Memory:
    """Structured memory with rich metadata"""
    id: str
    content: str
    memory_type: MemoryType
    query: str
    response: str
    context: Dict[str, Any]
    timestamp: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    success_rate: float = 1.0  # Tracks if solutions worked
    version: int = 1  # For updates when solutions fail
    parent_id: Optional[str] = None  # Links to previous versions
    generation_cost: float = 0.0  # Cost to generate this response
    model_used: str = "unknown"
    confidence_score: float = 0.8


@dataclass
class ConversationState:
    """Tracks the current problem-solving context"""
    problem_id: str
    original_query: str
    screen_context: Optional[str]
    attempted_solutions: List[Dict[str, Any]]
    current_status: str  # "solving", "testing", "failed", "succeeded"
    started_at: datetime


class MemoryWorthinessEvaluator:
    """Decides whether information is worth storing in memory"""
    
    def __init__(self):
        # Patterns that indicate trivial queries
        self.trivial_patterns = [
            "what day", "what time", "what date",
            "how old", "when was", "where is",
            "capital of", "population of"
        ]
        
        # Patterns indicating valuable technical knowledge
        self.valuable_patterns = [
            "error", "bug", "implement", "optimize",
            "debug", "architect", "design pattern",
            "best practice", "how to", "explain"
        ]
    
    def assess_query_complexity(self, query: str) -> QueryComplexity:
        """Determine the complexity level of a query"""
        query_lower = query.lower()
        
        # Check for trivial patterns
        if any(pattern in query_lower for pattern in self.trivial_patterns):
            if len(query.split()) < 10:  # Short trivial questions
                return QueryComplexity.TRIVIAL
        
        # Check word count and structure
        word_count = len(query.split())
        question_marks = query.count('?')
        
        # Check for code-like content
        has_code = any(marker in query for marker in ['()', '[]', '{}', '->', '=>', 'def', 'class'])
        
        # Check for valuable patterns
        has_valuable = any(pattern in query_lower for pattern in self.valuable_patterns)
        
        # Scoring logic
        if word_count < 5 and not has_code:
            return QueryComplexity.SIMPLE
        elif has_valuable or has_code:
            if word_count > 20 or 'implement' in query_lower or 'design' in query_lower:
                return QueryComplexity.EXPERT
            else:
                return QueryComplexity.COMPLEX
        elif word_count > 15:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def should_store_memory(self, query: str, response: str, generation_cost: float,
                          complexity: QueryComplexity) -> Tuple[bool, str]:
        """
        Decide if a memory is worth storing
        Returns: (should_store, reason)
        """
        # Always store expensive responses
        if generation_cost > 0.01:  # More than 1 cent
            return True, f"Expensive response (${generation_cost:.3f})"
        
        # Always store complex and expert queries
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
            return True, f"High complexity: {complexity.name}"
        
        # Check response characteristics
        response_length = len(response)
        has_code = '```' in response or 'def ' in response or 'class ' in response
        has_structured_answer = any(marker in response for marker in ['1.', '2.', '```', '- '])
        
        # Long, structured responses are valuable
        if response_length > 500 and has_structured_answer:
            return True, "Long structured response"
        
        # Code solutions are always valuable
        if has_code:
            return True, "Contains code solution"
        
        # Skip trivial responses
        if complexity == QueryComplexity.TRIVIAL and response_length < 100:
            return False, "Trivial query with short response"
        
        # Skip simple factual answers
        if complexity == QueryComplexity.SIMPLE and not has_structured_answer:
            return False, "Simple factual response"
        
        # Default: store moderate complexity
        return True, "Default storage for moderate complexity"


class ADAMMemoryAdvanced:
    """
    Advanced memory system with intelligent storage, updates, and retrieval
    """
    
    def __init__(self, persist_directory: str = "./adam_memory_advanced"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        console.print("[yellow]🧠 Initializing Advanced Memory System...[/yellow]")
        
        # Core components
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.worthiness_evaluator = MemoryWorthinessEvaluator()
        
        # Vector database with advanced settings
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Main memory collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="adam_advanced_memory",
            metadata={"description": "ADAM's intelligent memory with versioning"}
        )
        
        # Conversation state tracking
        self.conversation_states: Dict[str, ConversationState] = {}
        self.current_conversation_id: Optional[str] = None
        
        # Memory access patterns for optimization
        self.access_log_path = self.persist_directory / "access_patterns.json"
        self.access_patterns = self._load_access_patterns()
        
        # Cost tracking
        self.cost_log_path = self.persist_directory / "cost_savings.json"
        self.cost_savings = self._load_cost_savings()
        
        console.print("[green]✅ Advanced Memory System Ready![/green]")
    
    def _load_access_patterns(self) -> Dict[str, Any]:
        """Load memory access patterns for optimization"""
        if self.access_log_path.exists():
            with open(self.access_log_path, 'r') as f:
                return json.load(f)
        return {"memory_hits": 0, "memory_misses": 0, "total_queries": 0}
    
    def _load_cost_savings(self) -> Dict[str, float]:
        """Load cost tracking data"""
        if self.cost_log_path.exists():
            with open(self.cost_log_path, 'r') as f:
                return json.load(f)
        return {
            "total_saved": 0.0,
            "storage_cost": 0.0,
            "retrieval_savings": 0.0,
            "expensive_queries_cached": 0
        }
    
    def _save_metadata(self):
        """Persist metadata to disk"""
        with open(self.access_log_path, 'w') as f:
            json.dump(self.access_patterns, f, indent=2)
        
        with open(self.cost_log_path, 'w') as f:
            json.dump(self.cost_savings, f, indent=2)
    
    def remember_if_worthy(self, query: str, response: str, 
                          context: Optional[Dict[str, Any]] = None,
                          generation_cost: float = 0.0,
                          model_used: str = "unknown") -> Optional[str]:
        """
        Intelligently decide whether to store a memory based on its value
        """
        # Assess the query complexity
        complexity = self.worthiness_evaluator.assess_query_complexity(query)
        
        # Decide if worth storing
        should_store, reason = self.worthiness_evaluator.should_store_memory(
            query, response, generation_cost, complexity
        )
        
        if not should_store:
            console.print(f"[dim]💭 Not storing: {reason}[/dim]")
            return None
        
        # Determine memory type
        memory_type = self._classify_memory_type(query, response, context)
        
        # Create memory object
        memory = Memory(
            id=hashlib.md5(f"{query}{datetime.now()}".encode()).hexdigest()[:12],
            content=f"Query: {query}\n\nResponse: {response}",
            memory_type=memory_type,
            query=query,
            response=response,
            context=context or {},
            timestamp=datetime.now(),
            generation_cost=generation_cost,
            model_used=model_used,
            confidence_score=self._calculate_confidence(response, complexity)
        )
        
        # Store in vector database
        self._store_memory(memory)
        
        # Update cost tracking
        self.cost_savings["storage_cost"] += 0.00001  # Rough storage cost estimate
        self._save_metadata()
        
        console.print(f"[green]💾 Stored {memory_type.value} memory: {reason}[/green]")
        console.print(f"[dim]   ID: {memory.id} | Cost: ${generation_cost:.3f}[/dim]")
        
        return memory.id
    
    def _classify_memory_type(self, query: str, response: str, 
                             context: Optional[Dict[str, Any]]) -> MemoryType:
        """Classify the type of memory for better organization"""
        query_lower = query.lower()
        
        if context and "screen_content" in context:
            return MemoryType.SCREEN_ANALYSIS
        elif "error" in query_lower or "bug" in query_lower:
            return MemoryType.ERROR_SOLUTION
        elif any(term in response.lower() for term in ["```", "def ", "class ", "function"]):
            return MemoryType.CODE_PATTERN
        elif "explain" in query_lower or "what is" in query_lower:
            return MemoryType.CONCEPT_EXPLANATION
        elif context and context.get("generation_cost", 0) > 0.02:
            return MemoryType.EXPENSIVE_RESPONSE
        else:
            return MemoryType.CONVERSATION
    
    def _calculate_confidence(self, response: str, complexity: QueryComplexity) -> float:
        """Calculate confidence score for a response"""
        # Base confidence on response characteristics
        confidence = 0.5
        
        # Longer responses often more complete
        if len(response) > 200:
            confidence += 0.2
        
        # Structured responses indicate thoughtfulness
        if any(marker in response for marker in ['1.', '2.', '```', '- ']):
            confidence += 0.2
        
        # Complex queries might have less certain answers
        if complexity == QueryComplexity.EXPERT:
            confidence -= 0.1
        
        return min(1.0, max(0.1, confidence))
    
    def _store_memory(self, memory: Memory):
        """Store memory in the vector database"""
        # Prepare metadata for ChromaDB
        metadata = {
            "memory_type": memory.memory_type.value,
            "timestamp": memory.timestamp.isoformat(),
            "generation_cost": memory.generation_cost,
            "model_used": memory.model_used,
            "confidence_score": memory.confidence_score,
            "version": memory.version,
            "success_rate": memory.success_rate,
            "query_text": memory.query  # For exact matching
        }
        
        if memory.parent_id:
            metadata["parent_id"] = memory.parent_id
        
        # Store in ChromaDB
        self.collection.add(
            documents=[memory.content],
            metadatas=[metadata],
            ids=[memory.id]
        )
    
    def recall_with_context(self, query: str, 
                           screen_context: Optional[str] = None,
                           n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories considering both query similarity and context relevance
        """
        console.print(f"[cyan]🔍 Searching memories for: '{query[:50]}...'[/cyan]")
        
        # Update access patterns
        self.access_patterns["total_queries"] += 1
        
        # Search vector database
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results * 2  # Get more candidates for filtering
        )
        
        if not results['documents'] or not results['documents'][0]:
            self.access_patterns["memory_misses"] += 1
            self._save_metadata()
            return []
        
        # Process and score results considering context
        memories = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0]
        )):
            similarity = 1 - distance
            
            # Boost score for exact query matches
            if metadata.get('query_text', '').lower() == query.lower():
                similarity = min(1.0, similarity + 0.2)
            
            # Adjust score based on memory type relevance
            if screen_context and metadata.get('memory_type') == 'screen_analysis':
                similarity = min(1.0, similarity + 0.1)
            
            # Penalize failed solutions
            success_rate = metadata.get('success_rate', 1.0)
            similarity *= success_rate
            
            memory_dict = {
                'id': results['ids'][0][i],
                'content': doc,
                'metadata': metadata,
                'similarity': similarity,
                'distance': distance
            }
            
            memories.append(memory_dict)
        
        # Sort by adjusted similarity and return top results
        memories.sort(key=lambda x: x['similarity'], reverse=True)
        top_memories = memories[:n_results]
        
        if top_memories:
            self.access_patterns["memory_hits"] += 1
            # Track cost savings
            avg_cost = np.mean([m['metadata'].get('generation_cost', 0.01) 
                              for m in top_memories])
            self.cost_savings["retrieval_savings"] += avg_cost
            self._save_metadata()
        
        return top_memories
    
    def update_memory_success(self, memory_id: str, success: bool, 
                            new_solution: Optional[str] = None):
        """
        Update a memory based on whether its solution worked
        This is called when user reports success or failure
        """
        # Get the original memory
        result = self.collection.get(ids=[memory_id])
        if not result['ids']:
            console.print(f"[red]Memory {memory_id} not found[/red]")
            return
        
        metadata = result['metadatas'][0]
        document = result['documents'][0]
        
        if success:
            # Increase success rate
            new_success_rate = min(1.0, metadata.get('success_rate', 1.0) * 1.1)
            metadata['success_rate'] = new_success_rate
            metadata['last_confirmed'] = datetime.now().isoformat()
            
            # Update in place
            self.collection.update(
                ids=[memory_id],
                metadatas=[metadata]
            )
            
            console.print(f"[green]✅ Memory {memory_id} marked as successful[/green]")
        
        else:
            # Decrease success rate of old solution
            metadata['success_rate'] = max(0.1, metadata.get('success_rate', 1.0) * 0.7)
            
            # If we have a new solution, create a new version
            if new_solution:
                # Parse the original memory
                parts = document.split('\n\nResponse: ')
                original_query = parts[0].replace('Query: ', '')
                
                # Create new memory with updated solution
                new_memory = Memory(
                    id=hashlib.md5(f"{original_query}{datetime.now()}".encode()).hexdigest()[:12],
                    content=f"Query: {original_query}\n\nResponse: {new_solution}",
                    memory_type=MemoryType(metadata.get('memory_type', 'conversation')),
                    query=original_query,
                    response=new_solution,
                    context={"updated_from_failure": True, "previous_attempts": 1},
                    timestamp=datetime.now(),
                    version=metadata.get('version', 1) + 1,
                    parent_id=memory_id,
                    generation_cost=metadata.get('generation_cost', 0.0),
                    model_used=metadata.get('model_used', 'unknown')
                )
                
                # Store the new version
                self._store_memory(new_memory)
                
                console.print(f"[yellow]📝 Created new version of memory: {new_memory.id}[/yellow]")
                console.print(f"[dim]   Previous version ({memory_id}) marked as less reliable[/dim]")
                
                # Update the old memory to reflect failure
                self.collection.update(
                    ids=[memory_id],
                    metadatas=[metadata]
                )
    
    def start_problem_solving(self, query: str, screen_context: Optional[str] = None) -> str:
        """
        Begin tracking a problem-solving conversation
        """
        problem_id = hashlib.md5(f"{query}{datetime.now()}".encode()).hexdigest()[:8]
        
        self.conversation_states[problem_id] = ConversationState(
            problem_id=problem_id,
            original_query=query,
            screen_context=screen_context,
            attempted_solutions=[],
            current_status="solving",
            started_at=datetime.now()
        )
        
        self.current_conversation_id = problem_id
        console.print(f"[cyan]🎯 Started tracking problem: {problem_id}[/cyan]")
        
        return problem_id
    
    def add_solution_attempt(self, solution: str, memory_id: Optional[str] = None):
        """
        Track a solution attempt in the current conversation
        """
        if not self.current_conversation_id:
            return
        
        state = self.conversation_states.get(self.current_conversation_id)
        if state:
            state.attempted_solutions.append({
                "solution": solution,
                "memory_id": memory_id,
                "timestamp": datetime.now().isoformat()
            })
            state.current_status = "testing"
    
    def handle_solution_feedback(self, feedback: str) -> Dict[str, Any]:
        """
        Process user feedback about whether a solution worked
        Returns context for generating next response
        """
        if not self.current_conversation_id:
            return {"error": "No active problem-solving session"}
        
        state = self.conversation_states.get(self.current_conversation_id)
        if not state:
            return {"error": "Conversation state not found"}
        
        feedback_lower = feedback.lower()
        
        # Determine if the solution worked
        success_indicators = ["works", "fixed", "solved", "thank", "perfect", "great"]
        failure_indicators = ["doesn't work", "didn't work", "still broken", "error", "failed"]
        
        if any(indicator in feedback_lower for indicator in success_indicators):
            state.current_status = "succeeded"
            
            # Mark the last attempted solution as successful
            if state.attempted_solutions:
                last_attempt = state.attempted_solutions[-1]
                if last_attempt.get("memory_id"):
                    self.update_memory_success(last_attempt["memory_id"], True)
            
            return {
                "status": "success",
                "message": "Great! The solution worked.",
                "problem_id": state.problem_id
            }
        
        elif any(indicator in feedback_lower for indicator in failure_indicators):
            state.current_status = "failed"
            
            # Mark the last attempted solution as failed
            if state.attempted_solutions:
                last_attempt = state.attempted_solutions[-1]
                if last_attempt.get("memory_id"):
                    self.update_memory_success(last_attempt["memory_id"], False)
            
            return {
                "status": "failure",
                "message": "I understand the solution didn't work. Let me try a different approach.",
                "problem_id": state.problem_id,
                "original_query": state.original_query,
                "screen_context": state.screen_context,
                "previous_attempts": len(state.attempted_solutions),
                "attempted_solutions": state.attempted_solutions
            }
        
        else:
            return {
                "status": "unclear",
                "message": "I'm not sure if the solution worked. Could you clarify?",
                "problem_id": state.problem_id
            }
    
    def get_memory_analytics(self) -> Dict[str, Any]:
        """
        Provide analytics about memory usage and effectiveness
        """
        # Get all memories
        all_data = self.collection.get()
        
        if not all_data['ids']:
            return {"status": "No memories stored yet"}
        
        total_memories = len(all_data['ids'])
        
        # Analyze by type
        type_counts = {}
        total_cost = 0.0
        success_rates = []
        
        for metadata in all_data['metadatas']:
            # Count by type
            mem_type = metadata.get('memory_type', 'unknown')
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
            
            # Sum costs
            total_cost += metadata.get('generation_cost', 0)
            
            # Collect success rates
            success_rates.append(metadata.get('success_rate', 1.0))
        
        # Calculate hit rate
        hit_rate = 0.0
        if self.access_patterns['total_queries'] > 0:
            hit_rate = self.access_patterns['memory_hits'] / self.access_patterns['total_queries']
        
        analytics = {
            "total_memories": total_memories,
            "memory_types": type_counts,
            "total_generation_cost": total_cost,
            "total_savings": self.cost_savings['retrieval_savings'],
            "net_savings": self.cost_savings['retrieval_savings'] - total_cost,
            "average_success_rate": np.mean(success_rates) if success_rates else 0.0,
            "memory_hit_rate": hit_rate,
            "total_queries": self.access_patterns['total_queries'],
            "roi_percentage": ((self.cost_savings['retrieval_savings'] / max(total_cost, 0.01)) - 1) * 100
        }
        
        return analytics
    
    def display_analytics_dashboard(self):
        """
        Show a beautiful analytics dashboard
        """
        analytics = self.get_memory_analytics()
        
        # Create main stats table
        stats_table = Table(title="📊 ADAM Memory Analytics Dashboard")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Memories", str(analytics.get('total_memories', 0)))
        stats_table.add_row("Memory Hit Rate", f"{analytics.get('memory_hit_rate', 0):.1%}")
        stats_table.add_row("Average Success Rate", f"{analytics.get('average_success_rate', 0):.1%}")
        stats_table.add_row("Total Queries", str(analytics.get('total_queries', 0)))
        
        console.print(stats_table)
        
        # Create cost analysis table
        cost_table = Table(title="💰 Cost Analysis")
        cost_table.add_column("Category", style="cyan")
        cost_table.add_column("Amount", style="green")
        
        cost_table.add_row("Generation Costs", f"${analytics.get('total_generation_cost', 0):.3f}")
        cost_table.add_row("Retrieval Savings", f"${analytics.get('total_savings', 0):.3f}")
        cost_table.add_row("Net Savings", f"${analytics.get('net_savings', 0):.3f}")
        cost_table.add_row("ROI", f"{analytics.get('roi_percentage', 0):.1f}%")
        
        console.print(cost_table)
        
        # Show memory type distribution
        if analytics.get('memory_types'):
            type_table = Table(title="🗂️ Memory Type Distribution")
            type_table.add_column("Type", style="cyan")
            type_table.add_column("Count", style="green")
            
            for mem_type, count in analytics['memory_types'].items():
                type_table.add_row(mem_type, str(count))
            
            console.print(type_table)
    
    def export_knowledge_base(self, output_path: str = "adam_knowledge.json"):
        """
        Export the entire knowledge base for backup or analysis
        """
        all_data = self.collection.get()
        
        export_data = {
            "export_date": datetime.now().isoformat(),
            "total_memories": len(all_data['ids']),
            "memories": []
        }
        
        for i, memory_id in enumerate(all_data['ids']):
            export_data["memories"].append({
                "id": memory_id,
                "content": all_data['documents'][i],
                "metadata": all_data['metadatas'][i]
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        console.print(f"[green]📤 Exported {len(all_data['ids'])} memories to {output_path}[/green]")


def demonstrate_advanced_features():
    """
    Interactive demonstration of the advanced memory system
    """
    console.print(Panel.fit(
        "[bold cyan]ADAM Advanced Memory System Demo[/bold cyan]\n\n"
        "This demonstration shows:\n"
        "• Intelligent storage decisions\n"
        "• Memory versioning and updates\n"
        "• Cost tracking and ROI\n"
        "• Conversation state management",
        title="🧠 Advanced Memory Features"
    ))
    
    # Initialize system
    memory = ADAMMemoryAdvanced()
    
    # Demo 1: Selective Storage
    console.print("\n[bold magenta]Demo 1: Intelligent Storage Decisions[/bold magenta]\n")
    
    test_queries = [
        ("What day is June 19th?", "June 19th is Juneteenth, a federal holiday.", 0.001),
        ("How do I implement a binary search tree in Python?", "Here's a complete implementation:\n```python\nclass Node:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None\n\nclass BinarySearchTree:\n    def __init__(self):\n        self.root = None\n    \n    def insert(self, value):\n        if not self.root:\n            self.root = Node(value)\n        else:\n            self._insert_recursive(self.root, value)\n```", 0.02),
        ("What's 2+2?", "2+2 equals 4", 0.0001)
    ]
    
    for query, response, cost in test_queries:
        console.print(f"\n[yellow]Query:[/yellow] {query}")
        memory_id = memory.remember_if_worthy(query, response, generation_cost=cost, model_used="demo")
        
    # Demo 2: Problem Solving with Feedback
    console.print("\n\n[bold magenta]Demo 2: Problem Solving with Feedback[/bold magenta]\n")
    
    # Start a problem
    problem_id = memory.start_problem_solving(
        "NameError: name 'df' is not defined",
        screen_context="import pandas as pd\n# Some code here\nprint(df.head())"
    )
    
    # First solution attempt
    first_solution = "You need to create the DataFrame first. Add: df = pd.DataFrame()"
    memory_id = memory.remember_if_worthy(
        "NameError: name 'df' is not defined",
        first_solution,
        context={"error_type": "NameError", "variable": "df"},
        generation_cost=0.01
    )
    memory.add_solution_attempt(first_solution, memory_id)
    
    # Simulate failure feedback
    console.print("\n[red]User: That didn't work, still getting the error[/red]")
    feedback_context = memory.handle_solution_feedback("That didn't work")
    console.print(f"[cyan]Feedback context:[/cyan] {feedback_context['message']}")
    
    # Try a different solution
    second_solution = "I see the issue. You need to actually read the data: df = pd.read_csv('your_file.csv')"
    memory.update_memory_success(memory_id, False, second_solution)
    
    # Demo 3: Memory Analytics
    console.print("\n\n[bold magenta]Demo 3: Memory Analytics Dashboard[/bold magenta]\n")
    memory.display_analytics_dashboard()
    
    # Demo 4: Context-Aware Retrieval
    console.print("\n\n[bold magenta]Demo 4: Context-Aware Retrieval[/bold magenta]\n")
    
    test_query = "NameError with pandas"
    memories = memory.recall_with_context(test_query, screen_context="working with dataframes")
    
    if memories:
        console.print(f"\n[green]Found {len(memories)} relevant memories:[/green]")
        for i, mem in enumerate(memories[:2], 1):
            console.print(f"\n[cyan]Memory {i}:[/cyan]")
            console.print(f"Similarity: {mem['similarity']:.3f}")
            console.print(f"Success Rate: {mem['metadata'].get('success_rate', 1.0):.1%}")
            console.print(f"Content: {mem['content'][:150]}...")


if __name__ == "__main__":
    # Run the demonstration
    demonstrate_advanced_features()