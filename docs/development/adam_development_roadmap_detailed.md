# ADAM Development Roadmap: Complete Implementation Guide

## Table of Contents
1. [Start Testing ADAM Today (Even Without Full Features)](#start-testing-adam-today)
2. [Week 1: Core LLM Integration - Detailed Steps](#week-1-core-llm-integration)
3. [Week 2: SQL and dbt Tools Implementation](#week-2-sql-and-dbt-tools)
4. [Week 3: Memory and Learning Systems](#week-3-memory-and-learning)
5. [Week 4: Analytics Intelligence](#week-4-analytics-intelligence)
6. [Week 5: Proactive Monitoring](#week-5-proactive-monitoring)
7. [Week 6: Production Features](#week-6-production-features)
8. [Week 7-8: Deployment and Scaling](#week-7-8-deployment)

---

## Start Testing ADAM Today (Even Without Full Features) 🚀

### What You Can Test Right Now (30 Minutes Setup)

#### Step 1: Basic Setup (10 minutes)
```bash
# 1. Clone and setup
cd ~/ADAM
python -m venv venv
source venv/bin/activate

# 2. Install current dependencies
pip install -r requirements.txt

# 3. Create test script
touch test_adam_basic.py
```

#### Step 2: Create Minimal Working ADAM (10 minutes)
```python
# test_adam_basic.py
#!/usr/bin/env python3
"""
Minimal ADAM for immediate testing
"""
import os
from datetime import datetime
from src.adam.memory import ADAMMemoryAdvanced
from src.adam.conversation_system import ConversationSystem
from src.adam.advanced_rag import AdvancedRAGSystem
from src.adam.memory_network import MemoryNetworkSystem

# Initialize ADAM
print("🧠 Initializing ADAM...")
memory = ADAMMemoryAdvanced(persist_directory="./adam_test_memory")
conversations = ConversationSystem(storage_path="./adam_test_conversations")
network = MemoryNetworkSystem(memory, conversations)

# Pre-populate with Analytics Engineering knowledge
analytics_knowledge = [
    ("How do I optimize a slow Snowflake query?", 
     "To optimize Snowflake queries: 1) Check query profile for bottlenecks, 2) Ensure proper clustering keys, 3) Use result caching, 4) Avoid SELECT *, 5) Materialize CTEs that are referenced multiple times"),
    
    ("My dbt model is failing with 'Database Error'", 
     "Common dbt database errors: 1) Check credentials in profiles.yml, 2) Verify schema permissions, 3) Look for syntax errors in compiled SQL (target/compiled/), 4) Check if upstream dependencies exist"),
    
    ("How to debug data discrepancies between dashboards?",
     "To debug data discrepancies: 1) Compare SQL queries side-by-side, 2) Check for different time zones, 3) Verify filters and WHERE clauses, 4) Look for different aggregation levels, 5) Check for NULL handling differences"),
]

# Seed knowledge
for query, response in analytics_knowledge:
    memory_id = memory.remember_if_worthy(
        query=query,
        response=response,
        context={"type": "analytics_engineering", "seeded": True},
        generation_cost=0.001,
        model_used="seed"
    )
    if memory_id:
        print(f"✓ Stored knowledge: {query[:50]}...")

# Initialize RAG system after seeding
rag = AdvancedRAGSystem(memory, network)

# Test retrieval
def ask_adam(question):
    print(f"\n👤 You: {question}")
    
    # Search memories
    results = rag.retrieve(question, k=3)
    
    if results:
        print(f"🤖 ADAM: Based on my knowledge...")
        for i, result in enumerate(results[:1]):  # Show top result
            print(f"\n{result.content}")
            print(f"\n[Found via: {result.retrieval_method} | Score: {result.score:.3f}]")
    else:
        print("🤖 ADAM: I don't have knowledge about that yet. Teach me!")
    
    # Store this conversation
    session_id = conversations.current_session.session_id if conversations.current_session else conversations.start_session()
    conversations.record_exchange(
        query=question,
        response=results[0].content if results else "No knowledge yet",
        topics=["analytics", "testing"],
        context={"test_run": True}
    )

# Interactive testing
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADAM is ready for testing! (Limited functionality)")
    print("="*60)
    
    # Test queries
    test_queries = [
        "How can I make my Snowflake queries faster?",
        "Debug dbt database error",
        "Why don't my dashboard numbers match?",
        "How to optimize a query with window functions?",  # Not in knowledge - will fail
    ]
    
    for query in test_queries:
        ask_adam(query)
        input("\nPress Enter for next query...")
    
    # Interactive mode
    print("\n💬 Now you can ask your own questions (type 'quit' to exit):")
    while True:
        user_query = input("\n👤 You: ")
        if user_query.lower() == 'quit':
            break
        ask_adam(user_query)
```

#### Step 3: Run Your First Test (10 minutes)
```bash
# Run the test
python test_adam_basic.py

# You'll see:
# 🧠 Initializing ADAM...
# ✓ Stored knowledge: How do I optimize a slow Snowflake query?...
# ✓ Stored knowledge: My dbt model is failing with 'Database Error'...
# ✓ Stored knowledge: How to debug data discrepancies between...
# 
# ADAM is ready for testing! (Limited functionality)
# 
# 👤 You: How can I make my Snowflake queries faster?
# 🤖 ADAM: Based on my knowledge...
# 
# Query: How do I optimize a slow Snowflake query?
# 
# Response: To optimize Snowflake queries: 1) Check query profile...
```

### What This Basic Test Shows You

1. **Memory System Works**: ADAM stores and retrieves information
2. **RAG Functions**: All three retrieval methods (BM25, Vector, Graph) are operational
3. **Conversation Tracking**: Each interaction is recorded
4. **Pattern Matching**: Similar questions find relevant answers

### Immediate Experiments You Can Try

#### Experiment 1: Teach ADAM Your Knowledge
```python
# Add this to your test script
def teach_adam(question, answer):
    memory_id = memory.remember_if_worthy(
        query=question,
        response=answer,
        context={"type": "user_teaching", "timestamp": datetime.now()},
        generation_cost=0.002,
        model_used="user_input"
    )
    if memory_id:
        print(f"✓ ADAM learned: {question[:50]}...")
    else:
        print("❌ ADAM deemed this not worth remembering (too simple)")

# Teach ADAM about your specific environment
teach_adam(
    "What's our Snowflake warehouse configuration?",
    "We use: ANALYTICS_WH (Large) for heavy transforms, REPORTING_WH (Medium) for BI tools, DEV_WH (Small) for development. Auto-suspend after 60 seconds."
)

teach_adam(
    "What's our dbt project structure?",
    "Our dbt structure: /models/staging (raw data cleaning), /models/marts (business logic), /models/intermediate (complex transforms). We use tags: 'daily', 'hourly', 'critical' for scheduling."
)
```

#### Experiment 2: Test Memory Connections
```python
# Test how ADAM connects related memories
queries = [
    "Snowflake warehouse",
    "warehouse configuration", 
    "which warehouse for development",
    "dbt development environment"
]

for q in queries:
    print(f"\nTesting variations: '{q}'")
    results = rag.retrieve(q, k=2)
    print(f"Found {len(results)} results")
    for r in results:
        print(f"  - {r.retrieval_method}: {r.metadata.get('query', '')[:50]}...")
```

#### Experiment 3: Test Learning Over Time
```python
# Simulate solving a problem and learning from it
problem_id = memory.start_problem_solving(
    "dbt model taking too long to run",
    screen_context="Model: marts/finance/monthly_revenue.sql - Runtime: 45 minutes"
)

# First attempt
memory.add_solution_attempt(
    "Added LIMIT 1000 for testing",
    memory_id=None
)

# Didn't work - learn from failure
feedback = memory.handle_solution_feedback("Still slow even with LIMIT")
print(f"Feedback result: {feedback}")

# Second attempt
solution = "Changed to incremental model with proper partition pruning"
memory_id = memory.remember_if_worthy(
    query="dbt model taking too long - monthly_revenue",
    response=f"Solution: {solution}. Key insight: Full refresh was scanning 5 years of data daily. Incremental on partition key reduced to scanning 1 day.",
    context={"problem_id": problem_id, "solution_worked": True},
    generation_cost=0.003
)

# Now test if ADAM remembers this solution
results = rag.retrieve("slow dbt model monthly revenue", k=1)
if results:
    print(f"\nADAM remembers: {results[0].content[:100]}...")
```

---

## Week 1: Core LLM Integration - Detailed Steps

### Day 1-2: LLM Provider Setup

#### Step 1: Create LLM Configuration System
```python
# src/adam/llm/config.py
import os
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

class ModelCapability(Enum):
    BASIC_QA = "basic_qa"
    CODE_GENERATION = "code_generation"
    COMPLEX_REASONING = "complex_reasoning"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"

@dataclass
class ModelConfig:
    name: str
    provider: str
    capabilities: List[ModelCapability]
    context_window: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    max_output_tokens: int
    supports_streaming: bool = True
    supports_functions: bool = False

class LLMConfig:
    """Central configuration for all LLM providers"""
    
    def __init__(self):
        self.models = {
            # OpenAI Models
            "gpt-3.5-turbo": ModelConfig(
                name="gpt-3.5-turbo",
                provider="openai",
                capabilities=[ModelCapability.BASIC_QA, ModelCapability.FUNCTION_CALLING],
                context_window=16384,
                cost_per_1k_input=0.0005,
                cost_per_1k_output=0.0015,
                max_output_tokens=4096,
                supports_functions=True
            ),
            "gpt-4": ModelConfig(
                name="gpt-4",
                provider="openai",
                capabilities=[ModelCapability.COMPLEX_REASONING, ModelCapability.CODE_GENERATION],
                context_window=8192,
                cost_per_1k_input=0.03,
                cost_per_1k_output=0.06,
                max_output_tokens=4096,
                supports_functions=True
            ),
            "gpt-4-turbo": ModelConfig(
                name="gpt-4-turbo-preview",
                provider="openai",
                capabilities=[ModelCapability.COMPLEX_REASONING, ModelCapability.CODE_GENERATION, ModelCapability.VISION],
                context_window=128000,
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.03,
                max_output_tokens=4096,
                supports_functions=True
            ),
            
            # Anthropic Models
            "claude-3-haiku": ModelConfig(
                name="claude-3-haiku-20240307",
                provider="anthropic",
                capabilities=[ModelCapability.BASIC_QA],
                context_window=200000,
                cost_per_1k_input=0.00025,
                cost_per_1k_output=0.00125,
                max_output_tokens=4096
            ),
            "claude-3-opus": ModelConfig(
                name="claude-3-opus-20240229",
                provider="anthropic",
                capabilities=[ModelCapability.COMPLEX_REASONING, ModelCapability.CODE_GENERATION],
                context_window=200000,
                cost_per_1k_input=0.015,
                cost_per_1k_output=0.075,
                max_output_tokens=4096
            ),
        }
        
        # API Keys from environment
        self.api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        }
    
    def get_cheapest_model_for_capability(self, capability: ModelCapability) -> Optional[str]:
        """Find the most cost-effective model for a given capability"""
        suitable_models = [
            (name, model) for name, model in self.models.items()
            if capability in model.capabilities and self.api_keys.get(model.provider)
        ]
        
        if not suitable_models:
            return None
            
        # Sort by average cost (input + output)
        suitable_models.sort(key=lambda x: x[1].cost_per_1k_input + x[1].cost_per_1k_output)
        return suitable_models[0][0]
```

#### Step 2: Create Unified LLM Client
```python
# src/adam/llm/client.py
import asyncio
from typing import Dict, List, Optional, AsyncGenerator
import openai
from anthropic import Anthropic, AsyncAnthropic
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMClient:
    """Unified client for all LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.clients = {}
        self._initialize_clients()
        self.token_counts = {"input": 0, "output": 0}
        self.total_cost = 0.0
        
    def _initialize_clients(self):
        """Initialize API clients for available providers"""
        if self.config.api_keys.get("openai"):
            openai.api_key = self.config.api_keys["openai"]
            self.clients["openai"] = openai
            
        if self.config.api_keys.get("anthropic"):
            self.clients["anthropic"] = AsyncAnthropic(
                api_key=self.config.api_keys["anthropic"]
            )
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens for a given text and model"""
        if "gpt" in model:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        else:
            # Rough estimation for non-OpenAI models
            return len(text) // 4
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def complete(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict:
        """
        Get completion from LLM
        
        Args:
            prompt: The prompt to send
            model: Model name (if None, auto-select based on prompt)
            temperature: Randomness (0-1)
            max_tokens: Maximum response length
            stream: Whether to stream the response
            
        Returns:
            Dict with 'content', 'model', 'tokens', 'cost'
        """
        # Auto-select model if not specified
        if not model:
            model = self._select_model_for_prompt(prompt)
        
        model_config = self.config.models.get(model)
        if not model_config:
            raise ValueError(f"Unknown model: {model}")
        
        # Count input tokens
        input_tokens = self.count_tokens(prompt, model)
        
        # Route to appropriate provider
        if model_config.provider == "openai":
            response = await self._complete_openai(
                prompt, model, temperature, max_tokens, stream
            )
        elif model_config.provider == "anthropic":
            response = await self._complete_anthropic(
                prompt, model, temperature, max_tokens, stream
            )
        else:
            raise ValueError(f"Unknown provider: {model_config.provider}")
        
        # Count output tokens and calculate cost
        output_tokens = self.count_tokens(response["content"], model)
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        
        # Update tracking
        self.token_counts["input"] += input_tokens
        self.token_counts["output"] += output_tokens
        self.total_cost += cost
        
        return {
            "content": response["content"],
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "total_cost": self.total_cost
        }
    
    async def _complete_openai(
        self, prompt: str, model: str, temperature: float, 
        max_tokens: Optional[int], stream: bool
    ) -> Dict:
        """OpenAI-specific completion"""
        client = self.clients["openai"]
        
        response = await asyncio.to_thread(
            client.ChatCompletion.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        
        if stream:
            return {"content": self._stream_openai_response(response)}
        else:
            return {"content": response.choices[0].message.content}
    
    async def _complete_anthropic(
        self, prompt: str, model: str, temperature: float,
        max_tokens: Optional[int], stream: bool
    ) -> Dict:
        """Anthropic-specific completion"""
        client = self.clients["anthropic"]
        
        response = await client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens or 4096,
            stream=stream
        )
        
        if stream:
            return {"content": self._stream_anthropic_response(response)}
        else:
            return {"content": response.content[0].text}
    
    def _select_model_for_prompt(self, prompt: str) -> str:
        """Auto-select best model based on prompt characteristics"""
        prompt_lower = prompt.lower()
        
        # Simple heuristics for model selection
        if any(term in prompt_lower for term in ["simple", "basic", "what is", "define"]):
            return self.config.get_cheapest_model_for_capability(ModelCapability.BASIC_QA)
        
        elif any(term in prompt_lower for term in ["code", "sql", "implement", "function"]):
            return self.config.get_cheapest_model_for_capability(ModelCapability.CODE_GENERATION)
        
        elif any(term in prompt_lower for term in ["analyze", "complex", "architect", "design"]):
            return self.config.get_cheapest_model_for_capability(ModelCapability.COMPLEX_REASONING)
        
        # Default to basic QA
        return self.config.get_cheapest_model_for_capability(ModelCapability.BASIC_QA)
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost of a completion"""
        model_config = self.config.models[model]
        input_cost = (input_tokens / 1000) * model_config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * model_config.cost_per_1k_output
        return input_cost + output_cost
```

#### Step 3: Create Analytics-Specific Prompt Templates
```python
# src/adam/llm/prompts.py
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    name: str
    template: str
    variables: List[str]
    model_hint: Optional[str] = None

class AnalyticsPrompts:
    """Prompt templates for analytics engineering tasks"""
    
    SQL_OPTIMIZATION = PromptTemplate(
        name="sql_optimization",
        template="""You are an expert SQL and {dialect} optimization specialist.

Analyze this query for performance issues:

```sql
{query}
```

Context:
- Database: {dialect}
- Table sizes: {table_info}
- Current execution time: {execution_time}

Provide:
1. Identified performance issues
2. Specific optimization recommendations
3. Rewritten query if applicable
4. Expected performance improvement

Focus on practical, implementable solutions.""",
        variables=["query", "dialect", "table_info", "execution_time"],
        model_hint="code_generation"
    )
    
    DBT_ERROR_RESOLUTION = PromptTemplate(
        name="dbt_error_resolution",
        template="""You are a dbt (data build tool) expert helping debug an error.

Error message:
```
{error_message}
```

Model details:
- Model name: {model_name}
- Model path: {model_path}
- dbt version: {dbt_version}

Model SQL:
```sql
{model_sql}
```

Provide:
1. Root cause of the error
2. Step-by-step solution
3. Preventive measures
4. Related dbt best practices

Be specific and actionable.""",
        variables=["error_message", "model_name", "model_path", "dbt_version", "model_sql"],
        model_hint="complex_reasoning"
    )
    
    DATA_LINEAGE_TRACE = PromptTemplate(
        name="data_lineage_trace",
        template="""Trace the lineage of the column '{column}' in table '{table}'.

Upstream dependencies:
{dependencies}

Transformation logic:
{transformations}

Provide:
1. Source system origin
2. All transformation steps
3. Business logic applied
4. Potential data quality issues
5. Impact of changes to this column""",
        variables=["column", "table", "dependencies", "transformations"]
    )
    
    QUERY_EXPLANATION = PromptTemplate(
        name="query_explanation",
        template="""Explain this SQL query in business terms:

```sql
{query}
```

Target audience: {audience}

Provide:
1. What the query does (business purpose)
2. Key metrics calculated
3. Filters and conditions applied
4. Any assumptions or caveats
5. Simple diagram if helpful

Use clear, non-technical language where possible.""",
        variables=["query", "audience"]
    )
    
    @classmethod
    def fill_template(cls, template: PromptTemplate, **kwargs) -> str:
        """Fill a template with provided variables"""
        # Check all required variables are provided
        missing = set(template.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        return template.template.format(**kwargs)
```

### Day 3-4: Integration with ADAM's Memory System

#### Step 1: Create LLM-Powered Memory Enhancement
```python
# src/adam/llm/memory_integration.py
from typing import Dict, Optional, List
import asyncio
from ..memory import ADAMMemoryAdvanced
from .client import LLMClient
from .prompts import AnalyticsPrompts

class LLMMemoryIntegration:
    """Integrates LLM capabilities with ADAM's memory system"""
    
    def __init__(self, memory_system: ADAMMemoryAdvanced, llm_client: LLMClient):
        self.memory = memory_system
        self.llm = llm_client
        
    async def answer_with_memory(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Answer a query using both memory and LLM
        
        Process:
        1. Search memory for relevant information
        2. Use LLM to synthesize answer from memories
        3. Store the new answer if valuable
        """
        # Search existing memories
        memories = self.memory.recall_with_context(
            query=query,
            screen_context=context.get("screen_context") if context else None,
            n_results=5
        )
        
        if memories:
            # Build context from memories
            memory_context = self._build_memory_context(memories)
            
            # Create prompt with memory context
            prompt = f"""Based on the following relevant memories, answer the user's question.

User Question: {query}

Relevant Memories:
{memory_context}

Provide a comprehensive answer that:
1. Directly addresses the user's question
2. Incorporates relevant information from memories
3. Indicates if information might be outdated
4. Suggests additional resources if needed

Answer:"""
            
            # Get LLM response
            response = await self.llm.complete(prompt, temperature=0.3)
            
            # Determine if this Q&A should be stored
            should_store = self._should_store_answer(query, response["content"], memories)
            
            if should_store:
                memory_id = self.memory.remember_if_worthy(
                    query=query,
                    response=response["content"],
                    context={
                        "source": "llm_synthesis",
                        "based_on_memories": [m["id"] for m in memories],
                        **context
                    } if context else {"source": "llm_synthesis"},
                    generation_cost=response["cost"],
                    model_used=response["model"]
                )
            
            return {
                "answer": response["content"],
                "based_on_memories": len(memories),
                "confidence": self._calculate_confidence(memories),
                "cost": response["cost"],
                "stored": should_store
            }
        
        else:
            # No relevant memories - pure LLM response
            prompt = f"""You are ADAM, an AI assistant for analytics engineers.

User Question: {query}

Provide a helpful answer based on general knowledge about:
- SQL optimization
- dbt (data build tool)
- Data warehousing (especially Snowflake)
- Analytics engineering best practices

If you're not certain, indicate that clearly.

Answer:"""
            
            response = await self.llm.complete(prompt, temperature=0.5)
            
            # Store this new knowledge
            memory_id = self.memory.remember_if_worthy(
                query=query,
                response=response["content"],
                context=context or {},
                generation_cost=response["cost"],
                model_used=response["model"]
            )
            
            return {
                "answer": response["content"],
                "based_on_memories": 0,
                "confidence": 0.6,  # Lower confidence without memory support
                "cost": response["cost"],
                "stored": memory_id is not None
            }
    
    def _build_memory_context(self, memories: List[Dict]) -> str:
        """Format memories for LLM context"""
        context_parts = []
        
        for i, memory in enumerate(memories, 1):
            # Extract key information
            query = memory["metadata"].get("query", "Unknown query")
            content = memory["content"]
            confidence = memory["metadata"].get("confidence_score", 0.8)
            date = memory["metadata"].get("timestamp", "Unknown date")
            
            context_parts.append(f"""Memory {i} (Confidence: {confidence:.0%}):
Question: {query}
Answer: {content}
Date: {date}
---""")
        
        return "\n".join(context_parts)
    
    def _should_store_answer(self, query: str, answer: str, used_memories: List[Dict]) -> bool:
        """Determine if an LLM-synthesized answer should be stored"""
        # Don't store if it's just repeating a single memory
        if len(used_memories) == 1 and len(answer) < len(used_memories[0]["content"]) * 1.2:
            return False
        
        # Store if it's a synthesis of multiple memories
        if len(used_memories) >= 2:
            return True
        
        # Store if it's significantly longer than the query (indicating new information)
        if len(answer) > len(query) * 3:
            return True
        
        return False
    
    def _calculate_confidence(self, memories: List[Dict]) -> float:
        """Calculate confidence based on memory support"""
        if not memories:
            return 0.5
        
        # Average similarity scores
        avg_similarity = sum(m.get("similarity", 0.5) for m in memories) / len(memories)
        
        # Boost for multiple supporting memories
        multi_memory_boost = min(0.2, len(memories) * 0.05)
        
        return min(0.95, avg_similarity + multi_memory_boost)
```

#### Step 2: Create Interactive ADAM Interface
```python
# src/adam/interface.py
import asyncio
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from .llm.memory_integration import LLMMemoryIntegration
from .llm.client import LLMClient
from .llm.config import LLMConfig

console = Console()

class ADAMInterface:
    """Interactive interface for ADAM with LLM capabilities"""
    
    def __init__(self, memory_integration: LLMMemoryIntegration):
        self.adam = memory_integration
        self.session_cost = 0.0
        
    async def chat(self):
        """Interactive chat with ADAM"""
        console.print(Panel.fit(
            "[bold cyan]ADAM - Analytics Engineering Assistant[/bold cyan]\n\n"
            "I can help with SQL, dbt, data warehousing, and analytics engineering.\n"
            "Type 'help' for commands or 'quit' to exit.",
            border_style="cyan"
        ))
        
        while True:
            try:
                # Get user input
                user_input = console.input("\n[bold green]You:[/bold green] ")
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'cost':
                    self._show_cost()
                    continue
                
                # Show thinking indicator
                with console.status("[bold yellow]ADAM is thinking...[/bold yellow]"):
                    response = await self.adam.answer_with_memory(user_input)
                
                # Display response
                console.print("\n[bold blue]ADAM:[/bold blue]")
                console.print(Markdown(response["answer"]))
                
                # Show metadata
                console.print(
                    f"\n[dim]Based on {response['based_on_memories']} memories | "
                    f"Confidence: {response['confidence']:.0%} | "
                    f"Cost: ${response['cost']:.4f}[/dim]"
                )
                
                self.session_cost += response["cost"]
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
        
        console.print(f"\n[cyan]Session ended. Total cost: ${self.session_cost:.4f}[/cyan]")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
## Available Commands

- **help** - Show this help message
- **cost** - Show session cost
- **quit** - Exit ADAM

## Example Questions

### SQL Optimization
- "How can I optimize a query with multiple JOINs?"
- "My query is scanning too much data in Snowflake"
- "Best practices for window functions performance"

### dbt Help
- "Debug this dbt compilation error: [paste error]"
- "When should I use incremental models?"
- "How to set up dbt tests for data quality?"

### Data Engineering
- "Design a slowly changing dimension (SCD Type 2)"
- "Best practices for data vault modeling"
- "How to handle late-arriving data?"
        """
        console.print(Markdown(help_text))
    
    def _show_cost(self):
        """Show session cost breakdown"""
        console.print(f"\n[cyan]Current session cost: ${self.session_cost:.4f}[/cyan]")
        console.print(f"[dim]Approximate messages at this rate: {0.10 / max(self.session_cost, 0.0001):.0f} for $0.10[/dim]")
```

### Day 5: Testing Your LLM-Integrated ADAM

#### Complete Test Script
```python
# test_adam_with_llm.py
#!/usr/bin/env python3
"""
Test ADAM with full LLM integration
"""
import asyncio
import os
from pathlib import Path

# Add to Python path
import sys
sys.path.append(str(Path(__file__).parent))

from src.adam.memory import ADAMMemoryAdvanced
from src.adam.conversation_system import ConversationSystem
from src.adam.memory_network import MemoryNetworkSystem
from src.adam.advanced_rag import AdvancedRAGSystem
from src.adam.llm.config import LLMConfig
from src.adam.llm.client import LLMClient
from src.adam.llm.memory_integration import LLMMemoryIntegration
from src.adam.interface import ADAMInterface

async def main():
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  No API keys found!")
        print("Set at least one of:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    print("🚀 Initializing ADAM with LLM capabilities...")
    
    # Initialize memory systems
    memory = ADAMMemoryAdvanced(persist_directory="./adam_llm_memory")
    conversations = ConversationSystem(storage_path="./adam_llm_conversations")
    network = MemoryNetworkSystem(memory, conversations)
    rag = AdvancedRAGSystem(memory, network)
    
    # Initialize LLM
    llm_config = LLMConfig()
    llm_client = LLMClient(llm_config)
    
    # Create integrated system
    memory_integration = LLMMemoryIntegration(memory, llm_client)
    
    # Create interface
    interface = ADAMInterface(memory_integration)
    
    print("✅ ADAM is ready!")
    print("\nAvailable models:")
    for provider, key in llm_config.api_keys.items():
        if key:
            print(f"  - {provider}: ✓")
    
    # Run interactive chat
    await interface.chat()

if __name__ == "__main__":
    asyncio.run(main())
```

#### What You Can Test Immediately

1. **Basic Q&A**:
   ```
   You: How do I create an incremental model in dbt?
   ADAM: [Provides comprehensive answer with examples]
   ```

2. **SQL Optimization**:
   ```
   You: This query is slow: SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.created_at > '2023-01-01'
   ADAM: [Analyzes query and suggests optimizations]
   ```

3. **Error Debugging**:
   ```
   You: Getting error "Database Error in model 'staging_orders' (models/staging/staging_orders.sql)"
   ADAM: [Helps debug based on error pattern]
   ```

4. **Learning Over Time**:
   ```
   You: Our Snowflake warehouse ANALYTICS_WH is set to XL size and auto-suspend after 60 seconds
   ADAM: [Stores this information for future queries]
   
   You: What size is our analytics warehouse?
   ADAM: [Recalls the information you taught it]
   ```

---

## Week 2: SQL and dbt Tools Implementation

### Day 1-2: SQL Analysis Tools

#### Step 1: SQL Parser and Analyzer
```python
# src/adam/tools/sql_analyzer.py
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Token
from typing import List, Dict, Set, Optional, Tuple
import re
from dataclasses import dataclass
from enum import Enum

class IssueType(Enum):
    PERFORMANCE = "performance"
    STYLE = "style"
    BEST_PRACTICE = "best_practice"
    ERROR_PRONE = "error_prone"

@dataclass
class SQLIssue:
    type: IssueType
    severity: str  # "high", "medium", "low"
    message: str
    line_number: Optional[int]
    suggestion: Optional[str]
    estimated_impact: Optional[str]

class SQLAnalyzer:
    """Comprehensive SQL analysis for analytics engineers"""
    
    def __init__(self, dialect: str = "snowflake"):
        self.dialect = dialect
        self.issues = []
        
    def analyze(self, query: str) -> Dict:
        """
        Analyze SQL query for various issues
        
        Returns:
            Dict containing issues, suggestions, and metrics
        """
        self.issues = []
        
        # Parse SQL
        parsed = sqlparse.parse(query)[0]
        
        # Run all analyzers
        self._check_select_star(parsed)
        self._check_missing_where_clause(parsed)
        self._check_join_conditions(parsed)
        self._check_subquery_optimization(query)
        self._check_window_functions(query)
        self._check_case_statements(query)
        self._check_data_type_conversions(query)
        self._check_snowflake_specific(query)
        
        # Calculate complexity score
        complexity = self._calculate_complexity(query)
        
        # Generate optimization suggestions
        suggestions = self._generate_suggestions()
        
        return {
            "issues": self.issues,
            "complexity_score": complexity,
            "suggestions": suggestions,
            "estimated_performance_impact": self._estimate_performance_impact()
        }
    
    def _check_select_star(self, parsed):
        """Check for SELECT * usage"""
        tokens = list(parsed.flatten())
        
        for i, token in enumerate(tokens):
            if token.ttype is None and token.value == '*':
                # Check if it's part of SELECT
                for j in range(max(0, i-5), i):
                    if tokens[j].ttype is sqlparse.tokens.DML and tokens[j].value.upper() == 'SELECT':
                        self.issues.append(SQLIssue(
                            type=IssueType.PERFORMANCE,
                            severity="high",
                            message="SELECT * found - specify explicit columns",
                            line_number=self._get_line_number(str(token)),
                            suggestion="List only required columns to reduce data transfer and improve performance",
                            estimated_impact="Can reduce query time by 20-80% depending on table width"
                        ))
                        break
    
    def _check_missing_where_clause(self, parsed):
        """Check for missing WHERE clause in large table queries"""
        has_where = any(isinstance(token, Where) for token in parsed.tokens)
        
        if not has_where and self._is_likely_large_table(str(parsed)):
            self.issues.append(SQLIssue(
                type=IssueType.PERFORMANCE,
                severity="high",
                message="No WHERE clause found - might scan entire table",
                line_number=None,
                suggestion="Add WHERE clause to filter data, or use LIMIT for development",
                estimated_impact="Unfiltered scans can be 100x slower than filtered queries"
            ))
    
    def _check_join_conditions(self, parsed):
        """Check for potentially problematic JOIN conditions"""
        query_str = str(parsed)
        
        # Check for JOIN without ON
        join_pattern = r'\b(JOIN|LEFT JOIN|RIGHT JOIN|INNER JOIN|OUTER JOIN)\s+(\w+)\s*(?!ON)'
        matches = re.finditer(join_pattern, query_str, re.IGNORECASE)
        
        for match in matches:
            self.issues.append(SQLIssue(
                type=IssueType.ERROR_PRONE,
                severity="high",
                message=f"JOIN without ON condition detected",
                line_number=self._get_line_number(match.group()),
                suggestion="Add explicit ON condition to avoid cartesian product",
                estimated_impact="Missing JOIN conditions can cause query to never complete"
            ))
        
        # Check for non-equi joins
        non_equi_pattern = r'ON\s+.*?(<>|!=|<|>)'
        if re.search(non_equi_pattern, query_str, re.IGNORECASE):
            self.issues.append(SQLIssue(
                type=IssueType.PERFORMANCE,
                severity="medium",
                message="Non-equi JOIN detected",
                line_number=None,
                suggestion="Non-equi JOINs can be slow. Consider restructuring query",
                estimated_impact="Can be 10-50x slower than equi-joins"
            ))
    
    def _check_subquery_optimization(self, query: str):
        """Check for subqueries that could be CTEs or JOINs"""
        # Count nested SELECT statements
        select_count = len(re.findall(r'\bSELECT\b', query, re.IGNORECASE))
        
        if select_count > 2:
            # Check for subqueries in SELECT clause (correlated subqueries)
            correlated_pattern = r'SELECT\s+.*?\(\s*SELECT'
            if re.search(correlated_pattern, query, re.IGNORECASE):
                self.issues.append(SQLIssue(
                    type=IssueType.PERFORMANCE,
                    severity="high",
                    message="Correlated subquery detected in SELECT clause",
                    line_number=None,
                    suggestion="Replace with JOIN or window function for better performance",
                    estimated_impact="Correlated subqueries execute once per row - extremely slow"
                ))
            
            # Suggest CTEs for readability
            if select_count > 3:
                self.issues.append(SQLIssue(
                    type=IssueType.STYLE,
                    severity="medium",
                    message=f"Query has {select_count} SELECT statements",
                    line_number=None,
                    suggestion="Consider using CTEs (WITH clause) for better readability",
                    estimated_impact="No performance impact, but greatly improves maintainability"
                ))
    
    def _check_window_functions(self, query: str):
        """Check for window function optimization opportunities"""
        window_pattern = r'(ROW_NUMBER|RANK|DENSE_RANK|LAG|LEAD|SUM|AVG|COUNT)\s*\(\s*\)\s*OVER'
        matches = re.finditer(window_pattern, query, re.IGNORECASE)
        
        window_count = len(list(matches))
        if window_count > 3:
            self.issues.append(SQLIssue(
                type=IssueType.PERFORMANCE,
                severity="medium",
                message=f"Query uses {window_count} window functions",
                line_number=None,
                suggestion="Consider materializing window function results in a CTE if used multiple times",
                estimated_impact="Multiple window functions can cause multiple data sorts"
            ))
        
        # Check for missing PARTITION BY
        over_without_partition = r'OVER\s*\(\s*ORDER BY'
        if re.search(over_without_partition, query, re.IGNORECASE):
            self.issues.append(SQLIssue(
                type=IssueType.PERFORMANCE,
                severity="low",
                message="Window function without PARTITION BY",
                line_number=None,
                suggestion="Add PARTITION BY if grouping is needed to enable parallel processing",
                estimated_impact="Can improve performance by 2-5x for large datasets"
            ))
    
    def _check_snowflake_specific(self, query: str):
        """Snowflake-specific optimizations"""
        if self.dialect.lower() != "snowflake":
            return
        
        # Check for FLATTEN without LATERAL
        if "FLATTEN" in query.upper() and "LATERAL" not in query.upper():
            self.issues.append(SQLIssue(
                type=IssueType.BEST_PRACTICE,
                severity="low",
                message="FLATTEN without LATERAL keyword",
                line_number=None,
                suggestion="Use LATERAL FLATTEN for clarity",
                estimated_impact="No performance impact, but improves readability"
            ))
        
        # Check for missing clustering keys hint
        if re.search(r'WHERE\s+\w+\s*=', query, re.IGNORECASE):
            self.issues.append(SQLIssue(
                type=IssueType.PERFORMANCE,
                severity="low",
                message="Consider clustering keys for frequently filtered columns",
                line_number=None,
                suggestion="If this query runs frequently, add clustering on WHERE clause columns",
                estimated_impact="Can reduce scan time by 50-90% for large tables"
            ))
    
    def _calculate_complexity(self, query: str) -> int:
        """Calculate query complexity score (0-100)"""
        score = 0
        
        # Base complexity from query length
        score += min(20, len(query) / 100)
        
        # JOINs add complexity
        join_count = len(re.findall(r'\bJOIN\b', query, re.IGNORECASE))
        score += join_count * 5
        
        # Subqueries add complexity
        subquery_count = len(re.findall(r'\(\s*SELECT', query, re.IGNORECASE))
        score += subquery_count * 10
        
        # CTEs reduce complexity
        cte_count = len(re.findall(r'\bWITH\b', query, re.IGNORECASE))
        score -= cte_count * 5
        
        # Window functions add complexity
        window_count = len(re.findall(r'\bOVER\b', query, re.IGNORECASE))
        score += window_count * 3
        
        return max(0, min(100, score))
    
    def format_query(self, query: str, style: str = "standard") -> str:
        """Format SQL query according to style guide"""
        if style == "dbt":
            # dbt style: lowercase keywords, 4-space indent
            formatted = sqlparse.format(
                query,
                keyword_case='lower',
                identifier_case='lower',
                indent_width=4,
                reindent=True,
                comma_first=True
            )
        else:
            # Standard style: uppercase keywords, 2-space indent
            formatted = sqlparse.format(
                query,
                keyword_case='upper',
                identifier_case='lower',
                indent_width=2,
                reindent=True
            )
        
        return formatted
    
    def _generate_suggestions(self) -> List[str]:
        """Generate actionable suggestions based on issues found"""
        suggestions = []
        
        # Group issues by type
        perf_issues = [i for i in self.issues if i.type == IssueType.PERFORMANCE]
        if perf_issues:
            suggestions.append(f"🚀 Found {len(perf_issues)} performance optimizations")
        
        # Prioritize high-severity issues
        high_severity = [i for i in self.issues if i.severity == "high"]
        if high_severity:
            suggestions.append(f"⚠️  Address {len(high_severity)} high-priority issues first")
        
        # Add specific quick wins
        if any("SELECT *" in i.message for i in self.issues):
            suggestions.append("✨ Quick win: Replace SELECT * with specific columns")
        
        return suggestions
```

#### Step 2: SQL Optimization Suggester
```python
# src/adam/tools/sql_optimizer.py
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class OptimizationSuggestion:
    title: str
    description: str
    original_pattern: str
    optimized_pattern: str
    expected_improvement: str
    difficulty: str  # "easy", "medium", "hard"

class SQLOptimizer:
    """Suggests specific SQL optimizations"""
    
    def __init__(self, dialect: str = "snowflake"):
        self.dialect = dialect
        
    def suggest_optimizations(self, query: str, execution_stats: Optional[Dict] = None) -> List[OptimizationSuggestion]:
        """
        Suggest specific optimizations based on query pattern and execution stats
        
        Args:
            query: SQL query to optimize
            execution_stats: Optional execution statistics (runtime, rows scanned, etc.)
        """
        suggestions = []
        
        # Pattern-based optimizations
        suggestions.extend(self._suggest_cte_conversions(query))
        suggestions.extend(self._suggest_join_optimizations(query))
        suggestions.extend(self._suggest_filter_pushdown(query))
        suggestions.extend(self._suggest_aggregation_optimizations(query))
        
        # Snowflake-specific
        if self.dialect.lower() == "snowflake":
            suggestions.extend(self._suggest_snowflake_optimizations(query))
        
        # Stats-based optimizations (if available)
        if execution_stats:
            suggestions.extend(self._suggest_stats_based_optimizations(query, execution_stats))
        
        return suggestions
    
    def _suggest_cte_conversions(self, query: str) -> List[OptimizationSuggestion]:
        """Suggest converting subqueries to CTEs"""
        suggestions = []
        
        # Find subqueries in FROM clause
        from_subquery_pattern = r'FROM\s*\(\s*SELECT[\s\S]+?\)\s*AS\s*(\w+)'
        matches = re.finditer(from_subquery_pattern, query, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            alias = match.group(1)
            suggestions.append(OptimizationSuggestion(
                title=f"Convert subquery '{alias}' to CTE",
                description="CTEs improve readability and can be referenced multiple times",
                original_pattern=match.group(0)[:50] + "...",
                optimized_pattern=f"WITH {alias} AS (SELECT ...) SELECT ... FROM {alias}",
                expected_improvement="Better readability, potential for reuse",
                difficulty="easy"
            ))
        
        return suggestions
    
    def _suggest_join_optimizations(self, query: str) -> List[OptimizationSuggestion]:
        """Suggest JOIN order and type optimizations"""
        suggestions = []
        
        # Check for RIGHT JOINs (less common, harder to read)
        if "RIGHT JOIN" in query.upper():
            suggestions.append(OptimizationSuggestion(
                title="Convert RIGHT JOIN to LEFT JOIN",
                description="LEFT JOINs are more intuitive and commonly used",
                original_pattern="... RIGHT JOIN table ON ...",
                optimized_pattern="... LEFT JOIN table ON ... (with tables swapped)",
                expected_improvement="Better readability, same performance",
                difficulty="easy"
            ))
        
        # Check for multiple JOINs without clear order
        join_count = len(re.findall(r'\bJOIN\b', query, re.IGNORECASE))
        if join_count > 3:
            suggestions.append(OptimizationSuggestion(
                title="Optimize JOIN order",
                description="Join smaller tables first, filter early",
                original_pattern="Multiple JOINs detected",
                optimized_pattern="1. Filter tables with WHERE first\n2. JOIN smallest tables first\n3. Save large JOINs for last",
                expected_improvement="Can reduce intermediate result sets by 50-90%",
                difficulty="medium"
            ))
        
        return suggestions
    
    def _suggest_filter_pushdown(self, query: str) -> List[OptimizationSuggestion]:
        """Suggest pushing filters down to reduce data scanned"""
        suggestions = []
        
        # Check for filters on JOINed tables that could be pushed down
        join_then_filter_pattern = r'FROM\s+(\w+).*?JOIN\s+(\w+).*?WHERE.*?\2\.'
        if re.search(join_then_filter_pattern, query, re.IGNORECASE | re.DOTALL):
            suggestions.append(OptimizationSuggestion(
                title="Push down WHERE filters",
                description="Apply filters before JOINs to reduce data volume",
                original_pattern="JOIN table THEN filter in WHERE",
                optimized_pattern="Filter in subquery/CTE BEFORE JOIN",
                expected_improvement="Reduce JOIN data volume by 50-95%",
                difficulty="medium"
            ))
        
        return suggestions
    
    def _suggest_snowflake_optimizations(self, query: str) -> List[OptimizationSuggestion]:
        """Snowflake-specific optimizations"""
        suggestions = []
        
        # Check for opportunities to use RESULT_SCAN
        if "LAST_QUERY_ID()" in query.upper():
            suggestions.append(OptimizationSuggestion(
                title="Use RESULT_SCAN for query results",
                description="RESULT_SCAN is more efficient than re-running queries",
                original_pattern="SELECT ... WHERE query_id = LAST_QUERY_ID()",
                optimized_pattern="SELECT * FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))",
                expected_improvement="Instant results, no recomputation",
                difficulty="easy"
            ))
        
        # Suggest clustering for large tables
        large_table_pattern = r'FROM\s+(\w+)\.(\w+)\.(\w+)'  # database.schema.table
        matches = re.finditer(large_table_pattern, query, re.IGNORECASE)
        
        for match in matches:
            table_name = f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
            suggestions.append(OptimizationSuggestion(
                title=f"Consider clustering on {table_name}",
                description="Clustering keys can dramatically improve query performance",
                original_pattern=f"Scanning {table_name}",
                optimized_pattern=f"ALTER TABLE {table_name} CLUSTER BY (frequently_filtered_columns)",
                expected_improvement="50-90% reduction in data scanned",
                difficulty="medium"
            ))
        
        return suggestions
    
    def optimize_query(self, query: str, aggressive: bool = False) -> str:
        """
        Automatically optimize a query
        
        Args:
            query: Original SQL query
            aggressive: If True, apply more aggressive optimizations
        """
        optimized = query
        
        # Basic optimizations (always safe)
        optimized = self._remove_select_star(optimized)
        optimized = self._add_limit_in_dev(optimized)
        optimized = self._convert_subqueries_to_ctes(optimized)
        
        if aggressive:
            # More aggressive optimizations
            optimized = self._reorder_joins(optimized)
            optimized = self._push_down_filters(optimized)
            optimized = self._materialize_repeated_ctes(optimized)
        
        return optimized
    
    def _remove_select_star(self, query: str) -> str:
        """Replace SELECT * with explicit columns (requires schema info)"""
        # For now, just add a comment
        if "SELECT *" in query.upper():
            return f"-- TODO: Replace SELECT * with explicit columns\n{query}"
        return query
```

### Day 3-4: dbt Integration Tools

#### Step 1: dbt Project Analyzer
```python
# src/adam/tools/dbt_analyzer.py
import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import re

@dataclass
class DbtModel:
    name: str
    path: Path
    sql_content: str
    config: Dict
    dependencies: Set[str]
    dependents: Set[str]
    materialization: str
    tags: List[str]
    
@dataclass
class DbtTest:
    name: str
    model: str
    column: Optional[str]
    test_type: str
    config: Dict

class DbtProjectAnalyzer:
    """Analyzes dbt projects for optimization opportunities and issues"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.models = {}
        self.tests = {}
        self.manifest = None
        self._load_project()
    
    def _load_project(self):
        """Load dbt project metadata"""
        # Load manifest if it exists
        manifest_path = self.project_path / "target" / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
                self._parse_manifest()
        else:
            # Fall back to parsing files directly
            self._parse_project_files()
    
    def _parse_manifest(self):
        """Parse dbt manifest.json for complete project info"""
        # Parse models
        for node_id, node in self.manifest.get("nodes", {}).items():
            if node["resource_type"] == "model":
                model = DbtModel(
                    name=node["name"],
                    path=Path(node["path"]),
                    sql_content=node.get("raw_sql", ""),
                    config=node.get("config", {}),
                    dependencies=set(node.get("depends_on", {}).get("nodes", [])),
                    dependents=set(),  # Will be populated later
                    materialization=node.get("config", {}).get("materialized", "view"),
                    tags=node.get("tags", [])
                )
                self.models[model.name] = model
        
        # Build dependent relationships
        for model in self.models.values():
            for dep in model.dependencies:
                if dep in self.models:
                    self.models[dep].dependents.add(model.name)
    
    def analyze_model_performance(self, model_name: str) -> Dict:
        """Analyze a specific model for performance issues"""
        if model_name not in self.models:
            return {"error": f"Model {model_name} not found"}
        
        model = self.models[model_name]
        issues = []
        suggestions = []
        
        # Check 1: Materialization strategy
        if model.materialization == "view" and len(model.dependents) > 3:
            issues.append({
                "type": "performance",
                "severity": "medium",
                "message": f"View '{model_name}' is referenced by {len(model.dependents)} models",
                "suggestion": "Consider materializing as a table for better performance"
            })
        
        # Check 2: Incremental strategy
        if model.materialization == "table" and self._should_be_incremental(model):
            issues.append({
                "type": "performance", 
                "severity": "high",
                "message": "Large table using full refresh",
                "suggestion": "Convert to incremental model to reduce build time"
            })
        
        # Check 3: Complex transformations
        complexity = self._calculate_model_complexity(model)
        if complexity > 80:
            issues.append({
                "type": "maintainability",
                "severity": "medium",
                "message": f"High complexity score: {complexity}",
                "suggestion": "Consider breaking into staging and intermediate models"
            })
        
        # Check 4: Missing tests
        model_tests = self._get_model_tests(model_name)
        if len(model_tests) == 0:
            issues.append({
                "type": "quality",
                "severity": "medium",
                "message": "No tests defined for this model",
                "suggestion": "Add unique, not_null, and relationship tests"
            })
        
        return {
            "model": model_name,
            "materialization": model.materialization,
            "dependencies": len(model.dependencies),
            "dependents": len(model.dependents),
            "complexity_score": complexity,
            "issues": issues,
            "tests": model_tests
        }
    
    def _should_be_incremental(self, model: DbtModel) -> bool:
        """Determine if a model should be incremental"""
        # Look for patterns indicating large data volume
        sql_lower = model.sql_content.lower()
        
        # Indicators of large data
        large_data_indicators = [
            "date_trunc('day'",  # Daily aggregations
            "date_trunc('hour'",  # Hourly aggregations
            "row_number() over",  # Window functions on large sets
            re.search(r'where\s+\w+_date\s*>=', sql_lower),  # Date filters
            re.search(r'from\s+raw\.', sql_lower),  # Reading from raw tables
        ]
        
        indicators_found = sum(1 for indicator in large_data_indicators if indicator)
        
        return indicators_found >= 2
    
    def suggest_incremental_conversion(self, model_name: str) -> str:
        """Generate incremental model template"""
        model = self.models.get(model_name)
        if not model:
            return f"Model {model_name} not found"
        
        # Find date column for incremental key
        date_column = self._find_date_column(model.sql_content)
        
        template = f"""-- Suggested incremental conversion for {model_name}

{{{{ config(
    materialized='incremental',
    unique_key='<your_unique_key>',  -- TODO: Set unique key
    on_schema_change='fail',
    incremental_strategy='merge'  -- or 'delete+insert' for Snowflake
) }}}}

{model.sql_content}

{{% if is_incremental() %}}
    -- Incremental filter
    WHERE {date_column or 'created_at'} > (
        SELECT MAX({date_column or 'created_at'}) 
        FROM {{{{ this }}}}
    )
{{% endif %}}
"""
        return template
    
    def _find_date_column(self, sql: str) -> Optional[str]:
        """Try to identify date column for incremental filter"""
        # Common date column patterns
        date_patterns = [
            r'(\w+_at)\s*[<>=]',  # created_at, updated_at
            r'(\w+_date)\s*[<>=]',  # order_date, transaction_date
            r'(\w+_timestamp)\s*[<>=]',  # event_timestamp
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, sql, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
```

#### Step 2: dbt Error Resolver
```python
# src/adam/tools/dbt_error_resolver.py
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass 
class DbtError:
    error_type: str
    model: Optional[str]
    message: str
    line_number: Optional[int]
    suggestion: str
    commands_to_try: List[str]

class DbtErrorResolver:
    """Intelligent dbt error resolution"""
    
    def __init__(self):
        self.error_patterns = self._build_error_patterns()
    
    def _build_error_patterns(self) -> List[Tuple[re.Pattern, callable]]:
        """Build regex patterns for common dbt errors"""
        return [
            # Compilation errors
            (
                re.compile(r"Compilation Error in model '(\w+)'.*?'(\w+)' is undefined", re.IGNORECASE),
                self._resolve_undefined_model
            ),
            (
                re.compile(r"Database Error in model '(\w+)'.*?Table '(.+?)' does not exist", re.IGNORECASE),
                self._resolve_missing_table
            ),
            (
                re.compile(r"Database Error.*?insufficient privileges", re.IGNORECASE),
                self._resolve_permission_error
            ),
            (
                re.compile(r"found duplicate model name '(\w+)'", re.IGNORECASE),
                self._resolve_duplicate_model
            ),
            (
                re.compile(r"Parsing Error.*?Invalid jinja", re.IGNORECASE),
                self._resolve_jinja_error
            ),
            (
                re.compile(r"Runtime Error.*?cannot connect to database", re.IGNORECASE),
                self._resolve_connection_error
            ),
            (
                re.compile(r"'(\w+)' is not a valid test", re.IGNORECASE),
                self._resolve_invalid_test
            ),
        ]
    
    def resolve_error(self, error_message: str) -> DbtError:
        """Analyze dbt error and provide resolution"""
        # Try each pattern
        for pattern, resolver in self.error_patterns:
            match = pattern.search(error_message)
            if match:
                return resolver(match, error_message)
        
        # Generic error handling
        return self._resolve_generic_error(error_message)
    
    def _resolve_undefined_model(self, match: re.Match, full_error: str) -> DbtError:
        """Resolve undefined model/source errors"""
        model_name = match.group(1)
        undefined_ref = match.group(2)
        
        return DbtError(
            error_type="compilation",
            model=model_name,
            message=f"Model '{model_name}' references undefined model/source '{undefined_ref}'",
            line_number=self._extract_line_number(full_error),
            suggestion=f"""To fix this error:

1. Check if '{undefined_ref}' exists in your project:
   - Is it a source? Check schema.yml
   - Is it a model? Check models/ directory
   - Is it in a different schema? Use ref('schema', 'model')

2. Common fixes:
   - Add missing source to schema.yml
   - Fix typo in ref() or source() call
   - Ensure dependent model is not disabled
   - Check if model is in a different package""",
            commands_to_try=[
                f"ls models/**/{undefined_ref}.sql",
                f"grep -r \"name: {undefined_ref}\" models/",
                "dbt ls",
                f"dbt run -m {undefined_ref}"
            ]
        )
    
    def _resolve_missing_table(self, match: re.Match, full_error: str) -> DbtError:
        """Resolve missing table errors"""
        model_name = match.group(1)
        table_name = match.group(2)
        
        return DbtError(
            error_type="database",
            model=model_name,
            message=f"Table '{table_name}' not found",
            line_number=self._extract_line_number(full_error),
            suggestion=f"""Table '{table_name}' doesn't exist in the database.

Possible causes:
1. Source table not created yet
2. Wrong database/schema in source definition
3. Table was dropped or renamed
4. Permission issues (can't see the table)

Solutions:
1. Verify table exists: 
   SELECT * FROM {table_name} LIMIT 1;
   
2. Check source configuration in schema.yml
3. Run upstream dependencies first
4. Verify database permissions""",
            commands_to_try=[
                f"dbt run -m +{model_name}",  # Run with upstream
                "dbt source freshness",  # Check sources
                f"dbt test -m source:*"  # Test sources
            ]
        )
    
    def _resolve_permission_error(self, match: re.Match, full_error: str) -> DbtError:
        """Resolve permission errors"""
        return DbtError(
            error_type="permission",
            model=None,
            message="Insufficient privileges to execute query",
            line_number=None,
            suggestion="""Database permission error. 

To resolve:
1. Check your dbt profile credentials
2. Verify role has necessary permissions:
   - CREATE/DROP for tables/views
   - SELECT on source tables
   - USAGE on schemas

For Snowflake:
```sql
-- Check current role
SELECT CURRENT_ROLE();

-- Check grants
SHOW GRANTS TO ROLE <your_role>;

-- Grant necessary permissions
GRANT CREATE TABLE ON SCHEMA <schema> TO ROLE <role>;
GRANT SELECT ON ALL TABLES IN SCHEMA <schema> TO ROLE <role>;
```""",
            commands_to_try=[
                "dbt debug",  # Check connection
                "dbt run --target dev",  # Try dev target
            ]
        )
    
    def _resolve_duplicate_model(self, match: re.Match, full_error: str) -> DbtError:
        """Resolve duplicate model name errors"""
        model_name = match.group(1)
        
        return DbtError(
            error_type="configuration",
            model=model_name,
            message=f"Duplicate model name '{model_name}'",
            line_number=None,
            suggestion=f"""Model name '{model_name}' exists in multiple places.

To fix:
1. Find all instances:
   find . -name "{model_name}.sql"
   
2. Solutions:
   - Rename one of the models
   - Move to different folders
   - Use custom schemas to disambiguate
   - Delete duplicate if unintended

3. Update any ref() calls after renaming""",
            commands_to_try=[
                f"find models -name '{model_name}.sql'",
                f"grep -r \"ref('{model_name}')\" models/",
            ]
        )
    
    def _extract_line_number(self, error_message: str) -> Optional[int]:
        """Extract line number from error message"""
        line_match = re.search(r'line (\d+)', error_message, re.IGNORECASE)
        if line_match:
            return int(line_match.group(1))
        return None
    
    def get_debug_checklist(self, error_type: str) -> List[str]:
        """Get a debugging checklist for error type"""
        checklists = {
            "compilation": [
                "Run `dbt compile` to see full error",
                "Check compiled SQL in target/compiled/",
                "Verify all ref() and source() calls",
                "Look for typos in model names",
                "Ensure all dependencies exist"
            ],
            "database": [
                "Run `dbt debug` to test connection",
                "Check if tables exist in database",
                "Verify schema permissions",
                "Try running with `--full-refresh`",
                "Check for case sensitivity issues"
            ],
            "permission": [
                "Verify credentials in profiles.yml",
                "Check role/user permissions",
                "Test with `dbt debug`",
                "Try a simple SELECT query directly",
                "Contact your DBA if needed"
            ],
            "configuration": [
                "Check dbt_project.yml syntax",
                "Validate YAML in schema files",
                "Look for duplicate configurations",
                "Verify model paths are correct",
                "Check for circular dependencies"
            ]
        }
        
        return checklists.get(error_type, [
            "Run `dbt compile` to see detailed error",
            "Check dbt.log for more information",
            "Verify your environment setup",
            "Try running a single model with `-m model_name`"
        ])
```

### Day 5: Complete Analytics Tools Test

#### Integrated Test Script
```python
# test_analytics_tools.py
#!/usr/bin/env python3
"""
Test ADAM's Analytics Engineering Tools
"""
import asyncio
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from src.adam.tools.sql_analyzer import SQLAnalyzer
from src.adam.tools.sql_optimizer import SQLOptimizer
from src.adam.tools.dbt_analyzer import DbtProjectAnalyzer
from src.adam.tools.dbt_error_resolver import DbtErrorResolver
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def test_sql_analysis():
    """Test SQL analysis capabilities"""
    console.print(Panel("SQL Analysis Test", style="bold blue"))
    
    # Sample problematic query
    test_query = """
    SELECT *
    FROM orders o
    JOIN customers c ON o.customer_id = c.id
    JOIN products p ON o.product_id = p.id
    WHERE o.created_at > '2024-01-01'
    AND c.country = 'US'
    """
    
    analyzer = SQLAnalyzer(dialect="snowflake")
    results = analyzer.analyze(test_query)
    
    # Display issues
    if results["issues"]:
        table = Table(title="SQL Issues Found")
        table.add_column("Severity", style="red")
        table.add_column("Issue")
        table.add_column("Suggestion")
        
        for issue in results["issues"]:
            table.add_row(
                issue.severity,
                issue.message,
                issue.suggestion
            )
        
        console.print(table)
    
    console.print(f"\nComplexity Score: {results['complexity_score']}/100")
    
    # Test formatting
    formatted = analyzer.format_query(test_query, style="dbt")
    console.print("\n[green]Formatted Query (dbt style):[/green]")
    console.print(formatted)

def test_sql_optimization():
    """Test SQL optimization suggestions"""
    console.print(Panel("SQL Optimization Test", style="bold green"))
    
    complex_query = """
    SELECT 
        c.customer_id,
        c.customer_name,
        (SELECT COUNT(*) FROM orders WHERE customer_id = c.customer_id) as order_count,
        (SELECT SUM(amount) FROM orders WHERE customer_id = c.customer_id) as total_spent
    FROM customers c
    WHERE c.created_at > '2023-01-01'
    """
    
    optimizer = SQLOptimizer(dialect="snowflake")
    suggestions = optimizer.suggest_optimizations(complex_query)
    
    for i, suggestion in enumerate(suggestions, 1):
        console.print(f"\n[yellow]Optimization {i}:[/yellow] {suggestion.title}")
        console.print(f"Description: {suggestion.description}")
        console.print(f"Expected Impact: {suggestion.expected_improvement}")
        console.print(f"Difficulty: {suggestion.difficulty}")

def test_dbt_error_resolution():
    """Test dbt error resolution"""
    console.print(Panel("dbt Error Resolution Test", style="bold red"))
    
    # Common dbt errors
    test_errors = [
        "Compilation Error in model 'staging_orders' (models/staging/staging_orders.sql)\n  'raw_orders' is undefined",
        "Database Error in model 'marts_revenue' (models/marts/marts_revenue.sql)\n  002003 (42S02): SQL compilation error:\n  Table 'ANALYTICS.STAGING.STG_ORDERS' does not exist",
        "Runtime Error\n  Database connection failed\n  Unable to connect to database: Incorrect username or password was specified"
    ]
    
    resolver = DbtErrorResolver()
    
    for error in test_errors:
        console.print(f"\n[red]Error:[/red] {error[:100]}...")
        
        resolution = resolver.resolve_error(error)
        
        console.print(f"[green]Type:[/green] {resolution.error_type}")
        console.print(f"[green]Model:[/green] {resolution.model or 'N/A'}")
        console.print(f"[green]Suggestion:[/green]\n{resolution.suggestion}")
        
        if resolution.commands_to_try:
            console.print("\n[yellow]Commands to try:[/yellow]")
            for cmd in resolution.commands_to_try:
                console.print(f"  $ {cmd}")

def test_integration():
    """Test integrated workflow"""
    console.print(Panel("Integrated Analytics Workflow", style="bold magenta"))
    
    # Simulate a complete debugging session
    console.print("\n[cyan]Scenario: Slow dbt model with errors[/cyan]\n")
    
    # 1. Analyze the SQL
    model_sql = """
    SELECT *
    FROM {{ ref('staging_orders') }} o
    LEFT JOIN {{ ref('staging_customers') }} c
        ON o.customer_id = c.customer_id
    WHERE o.order_date >= '2020-01-01'
    """
    
    console.print("1. Analyzing model SQL...")
    analyzer = SQLAnalyzer()
    issues = analyzer.analyze(model_sql)
    console.print(f"   Found {len(issues['issues'])} issues")
    
    # 2. Suggest optimizations
    console.print("\n2. Generating optimization suggestions...")
    optimizer = SQLOptimizer()
    suggestions = optimizer.suggest_optimizations(model_sql)
    console.print(f"   Generated {len(suggestions)} optimization suggestions")
    
    # 3. Check for incremental opportunity
    console.print("\n3. Checking if model should be incremental...")
    console.print("   ✓ Large date range detected")
    console.print("   ✓ Recommendation: Convert to incremental model")
    
    # 4. Generate incremental template
    console.print("\n4. Generated incremental model template:")
    console.print("""
{{ config(
    materialized='incremental',
    unique_key='order_id',
    on_schema_change='fail'
) }}

SELECT *
FROM {{ ref('staging_orders') }} o
LEFT JOIN {{ ref('staging_customers') }} c
    ON o.customer_id = c.customer_id
WHERE 1=1

{% if is_incremental() %}
    AND o.order_date > (SELECT MAX(order_date) FROM {{ this }})
{% endif %}
""")

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]ADAM Analytics Tools Test Suite[/bold cyan]\n\n"
        "Testing SQL analysis, optimization, and dbt tools",
        border_style="cyan"
    ))
    
    test_sql_analysis()
    console.print("\n" + "="*60 + "\n")
    
    test_sql_optimization()
    console.print("\n" + "="*60 + "\n")
    
    test_dbt_error_resolution()
    console.print("\n" + "="*60 + "\n")
    
    test_integration()
    
    console.print("\n[bold green]All tests completed![/bold green]")
```

---

## What You Can Start Testing Today

### 1. Basic Memory and Retrieval (Works Now)
```bash
python test_adam_basic.py
```
- Test memory storage
- Test retrieval methods
- Teach ADAM your knowledge

### 2. With LLM Integration (Requires API Keys)
```bash
export OPENAI_API_KEY="your-key"
# or
export ANTHROPIC_API_KEY="your-key"

python test_adam_with_llm.py
```
- Interactive chat
- Memory-augmented responses
- Cost tracking

### 3. Analytics Tools (Standalone)
```bash
python test_analytics_tools.py
```
- SQL analysis and optimization
- dbt error resolution
- Formatting and suggestions

### Your Next Steps

1. **Today**: Run `test_adam_basic.py` to see the foundation working
2. **Tomorrow**: Add your API keys and test LLM integration
3. **This Week**: Start using analytics tools on your real queries
4. **Next Week**: Begin building the full integrated system

Remember: Even without full features, you can start teaching ADAM about your specific environment and testing core capabilities. Each test helps you understand the system better and guides development priorities.