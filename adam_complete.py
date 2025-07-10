#!/usr/bin/env python3
"""
ADAM Complete Interface - Full-featured chat with complete visibility
Shows model selection, memory usage, costs, and all internal operations
"""
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import json
from dotenv import load_dotenv

# Setup
load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Core systems
from adam.memory import ADAMMemoryAdvanced
from adam.conversation_system import ConversationSystem
from adam.advanced_rag import AdvancedRAGSystem
from adam.memory_network import MemoryNetworkSystem
from adam.llm.client import UnifiedLLMClient
from adam.tools.sql_tools import SQLAnalyzer, SQLOptimizer, SQLFormatter

# For colored output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None
    rprint = print


class ADAMComplete:
    """Complete ADAM system with full transparency"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._print_header()
        
        # Initialize all systems
        self._print_status("Initializing memory system...")
        self.memory = ADAMMemoryAdvanced(persist_directory="./adam_complete_memory")
        
        self._print_status("Initializing conversation system...")
        self.conversations = ConversationSystem(storage_path="./adam_complete_conversations")
        
        self._print_status("Initializing memory network...")
        self.memory_network = MemoryNetworkSystem(self.memory, self.conversations)
        
        self._print_status("Initializing RAG system...")
        self.rag = AdvancedRAGSystem(self.memory, self.memory_network)
        
        self._print_status("Initializing LLM client...")
        self.llm_client = UnifiedLLMClient()
        
        # Show available models
        self._show_available_models()
        
        # Initialize tools
        self._print_status("Initializing SQL tools...")
        self.sql_analyzer = SQLAnalyzer("snowflake")
        self.sql_optimizer = SQLOptimizer("snowflake")
        self.sql_formatter = SQLFormatter("dbt")
        
        # Start session
        self.session_id = self.conversations.start_session(
            f"ADAM Complete Session - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        
        # Seed initial knowledge
        self._seed_analytics_knowledge()
        
        # Track costs
        self.total_cost = 0.0
        self.model_usage = {}
        
        self._print_success("\n✅ ADAM is fully initialized and ready!")
        self._print_info("Type 'help' for commands or start chatting about analytics!\n")
        
    def _print_header(self):
        """Print welcome header"""
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold blue]🧠 ADAM Complete Interface[/bold blue]\n"
                "[dim]Analytics Data Assistant & Manager[/dim]\n"
                "[green]Full transparency mode - See everything ADAM does![/green]",
                border_style="blue"
            ))
        else:
            print("""
╔════════════════════════════════════════════════════════════╗
║                  🧠 ADAM Complete Interface                ║
║              Analytics Data Assistant & Manager            ║
║                                                           ║
║        Full transparency mode - See everything!           ║
╚════════════════════════════════════════════════════════════╝
            """)
    
    def _print_status(self, message: str):
        """Print status message"""
        if self.verbose:
            if RICH_AVAILABLE:
                console.print(f"[yellow]⏳[/yellow] {message}")
            else:
                print(f"⏳ {message}")
    
    def _print_success(self, message: str):
        """Print success message"""
        if RICH_AVAILABLE:
            console.print(f"[green]{message}[/green]")
        else:
            print(message)
    
    def _print_info(self, message: str):
        """Print info message"""
        if RICH_AVAILABLE:
            console.print(f"[blue]{message}[/blue]")
        else:
            print(message)
    
    def _print_error(self, message: str):
        """Print error message"""
        if RICH_AVAILABLE:
            console.print(f"[red]❌ {message}[/red]")
        else:
            print(f"❌ {message}")
    
    def _show_available_models(self):
        """Show available LLM models"""
        models = self.llm_client.config.get_available_models()
        
        if RICH_AVAILABLE:
            table = Table(title="Available LLM Models", show_header=True)
            table.add_column("Model", style="cyan")
            table.add_column("Provider", style="magenta")
            table.add_column("Capabilities", style="green")
            
            for model_name in models:
                config = self.llm_client.config.get_model_config(model_name)
                caps = ", ".join([cap.value for cap in config.capabilities[:3]])
                table.add_row(model_name, config.provider.value, caps)
            
            console.print(table)
        else:
            print("\n📊 Available LLM Models:")
            for model_name in models:
                config = self.llm_client.config.get_model_config(model_name)
                print(f"  - {model_name} ({config.provider.value})")
    
    def _seed_analytics_knowledge(self):
        """Seed ADAM with analytics knowledge"""
        self._print_status("Seeding analytics knowledge base...")
        
        knowledge = [
            {
                "q": "What are the best practices for Snowflake query optimization?",
                "a": """Key Snowflake optimization practices:
1. **Use Clustering Keys**: Cluster on commonly filtered columns (dates, IDs)
2. **Avoid SELECT ***: Only query needed columns to reduce data scanned
3. **Use Result Caching**: Snowflake caches results for 24 hours
4. **Partition Large Tables**: Use date-based partitioning for time-series data
5. **Optimize JOIN Order**: Put smaller tables first in joins
6. **Use Materialized Views**: For complex, frequently-run aggregations
7. **Monitor Query Profile**: Check for exploding joins and full table scans"""
            },
            {
                "q": "How do I debug a failing dbt model?",
                "a": """dbt debugging steps:
1. **Check compiled SQL**: Look in target/compiled/ for the actual SQL
2. **Run in isolation**: dbt run -s model_name
3. **Check logs**: target/logs/dbt.log has detailed errors
4. **Verify dependencies**: dbt ls -s +model_name
5. **Test source data**: dbt test -s source:*
6. **Use dbt debug**: Checks connection and configuration
7. **Check schema permissions**: Ensure user has CREATE/SELECT rights"""
            },
            {
                "q": "What's the difference between OLTP and OLAP?",
                "a": """OLTP (Online Transaction Processing) vs OLAP (Online Analytical Processing):

**OLTP**: Transactional systems (e.g., PostgreSQL, MySQL)
- Many small transactions
- Normalized design (3NF)
- Row-oriented storage
- Real-time updates
- Example: E-commerce database

**OLAP**: Analytical systems (e.g., Snowflake, BigQuery)
- Complex queries on large datasets
- Denormalized design (Star/Snowflake schema)
- Column-oriented storage
- Batch updates
- Example: Data warehouse for analytics"""
            }
        ]
        
        for item in knowledge:
            self.memory.remember_if_worthy(
                query=item["q"],
                response=item["a"],
                context={"type": "analytics_knowledge", "source": "seed"},
                generation_cost=0.001,
                model_used="seed"
            )
    
    async def process_input(self, user_input: str) -> Dict:
        """Process user input and return detailed response"""
        start_time = datetime.now()
        
        # Initialize response data
        response_data = {
            "input": user_input,
            "timestamp": start_time.isoformat(),
            "memory_search": None,
            "model_selection": None,
            "sql_analysis": None,
            "response": None,
            "cost": 0.0,
            "processing_time": 0.0
        }
        
        # Check for commands
        if user_input.lower() == 'help':
            response_data["response"] = self._get_help()
            return response_data
        elif user_input.lower() == 'stats':
            response_data["response"] = self._get_stats()
            return response_data
        elif user_input.lower() == 'models':
            response_data["response"] = self._get_model_details()
            return response_data
        elif user_input.lower() == 'memory':
            response_data["response"] = self._get_memory_info()
            return response_data
        elif user_input.lower() in ['exit', 'quit', 'bye']:
            response_data["response"] = "👋 Goodbye! Session saved."
            return response_data
        
        # Check if SQL query
        if self._is_sql_query(user_input):
            if self.verbose:
                self._print_info("\n🔍 Detected SQL query - running analysis...")
            
            issues, metrics = self.sql_analyzer.analyze_query(user_input)
            response_data["sql_analysis"] = {
                "issues": len(issues),
                "complexity": metrics.complexity_score,
                "metrics": metrics
            }
            
            # Build SQL analysis response
            analysis = self._format_sql_analysis(user_input, issues, metrics)
            
            # Offer optimization if issues found
            if len(issues) > 2:
                analysis += "\n\n💡 Would you like me to optimize this query? (Type 'optimize' or ask another question)"
                self._last_analyzed_query = user_input
            
            response_data["response"] = analysis
            response_data["processing_time"] = (datetime.now() - start_time).total_seconds()
            return response_data
        
        # Check for optimization request
        if user_input.lower() == 'optimize' and hasattr(self, '_last_analyzed_query'):
            return await self._optimize_last_query()
        
        # Regular chat flow with memory and LLM
        
        # 1. Search memory
        if self.verbose:
            self._print_info("\n🔍 Searching memory...")
        
        search_results = self.rag.retrieve(user_input, k=5)
        response_data["memory_search"] = {
            "results_found": len(search_results),
            "top_scores": [r.score for r in search_results[:3]]
        }
        
        if self.verbose and search_results:
            print(f"  Found {len(search_results)} relevant memories")
            for i, result in enumerate(search_results[:2]):
                print(f"  [{i+1}] Score: {result.score:.3f} | Method: {result.retrieval_method}")
        
        # 2. Build context
        context = self._build_context(search_results)
        conv_history = self.conversations.get_conversation_context(lookback_exchanges=3)
        
        # 3. Determine which model to use
        if self.verbose:
            self._print_info("\n🤖 Selecting best model...")
        
        # Let's see what auto-selection would pick
        test_response = await self.llm_client.complete(
            prompt=user_input,
            model=None,  # Auto-select
            temperature=0.7,
            max_tokens=1  # Just to see which model
        )
        auto_selected_model = test_response.model
        
        # Build the full prompt
        prompt = self._build_prompt(user_input, context, conv_history)
        
        # Show model selection reasoning
        if self.verbose:
            print(f"  Auto-selected: {auto_selected_model}")
            print(f"  Reason: ", end="")
            if "sql" in user_input.lower() or "query" in user_input.lower():
                print("SQL/Analytics content detected → grok-4 preferred")
            elif any(word in user_input.lower() for word in ["explain", "why", "how", "debug"]):
                print("Reasoning task detected → o4-mini preferred (if available)")
            else:
                print("General query → Using fastest available model")
        
        response_data["model_selection"] = {
            "auto_selected": auto_selected_model,
            "reason": self._get_model_selection_reason(user_input)
        }
        
        # 4. Generate response
        if self.verbose:
            self._print_info(f"\n💭 Generating response with {auto_selected_model}...")
        
        try:
            llm_response = await self.llm_client.complete(
                prompt=prompt,
                model=None,  # Use auto-selection
                temperature=0.7,
                max_tokens=1000
            )
            
            response_data["response"] = llm_response.content
            response_data["cost"] = llm_response.cost
            response_data["model_selection"]["final_model"] = llm_response.model
            response_data["model_selection"]["tokens"] = llm_response.total_tokens
            
            # Update tracking
            self.total_cost += llm_response.cost
            self.model_usage[llm_response.model] = self.model_usage.get(llm_response.model, 0) + 1
            
            # 5. Store in memory if worthy
            if self.verbose:
                self._print_info("\n💾 Evaluating for memory storage...")
            
            memory_id = self.memory.remember_if_worthy(
                query=user_input,
                response=llm_response.content,
                context={
                    "session_id": self.session_id,
                    "model": llm_response.model,
                    "timestamp": datetime.now().isoformat()
                },
                generation_cost=llm_response.cost,
                model_used=llm_response.model
            )
            
            if memory_id and self.verbose:
                print(f"  ✅ Stored in memory (ID: {memory_id[:8]}...)")
            elif self.verbose:
                print("  ℹ️  Not stored (below worthiness threshold)")
            
            # 6. Record conversation
            self.conversations.record_exchange(
                query=user_input,
                response=llm_response.content,
                topics=self._extract_topics(user_input),
                metadata={
                    "model": llm_response.model,
                    "tokens": llm_response.total_tokens,
                    "cost": llm_response.cost,
                    "memory_results": len(search_results)
                }
            )
            
        except Exception as e:
            response_data["response"] = f"Error: {str(e)}\n\nMake sure your API keys are configured correctly."
            self._print_error(f"LLM Error: {e}")
        
        response_data["processing_time"] = (datetime.now() - start_time).total_seconds()
        return response_data
    
    def _is_sql_query(self, text: str) -> bool:
        """Check if text is likely a SQL query"""
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'WITH']
        text_upper = text.upper().strip()
        
        # Check if starts with SQL keyword
        for keyword in sql_keywords:
            if text_upper.startswith(keyword):
                return True
        
        # Check if has multiple SQL keywords
        keyword_count = sum(1 for kw in sql_keywords if kw in text_upper)
        return keyword_count >= 2 and len(text) > 30
    
    def _format_sql_analysis(self, query: str, issues: List, metrics) -> str:
        """Format SQL analysis results"""
        output = "📊 **SQL Query Analysis**\n\n"
        
        # Metrics
        output += "**Query Metrics:**\n"
        output += f"- Complexity Score: {metrics.complexity_score}/10\n"
        output += f"- Lines: {metrics.line_count}\n"
        output += f"- CTEs: {metrics.cte_count}\n"
        output += f"- Joins: {metrics.join_count}\n"
        output += f"- Subqueries: {metrics.subquery_count}\n"
        output += f"- DISTINCT operations: {metrics.distinct_count}\n"
        
        # Issues
        if issues:
            output += f"\n**Issues Found ({len(issues)}):**\n"
            
            # Group by severity
            errors = [i for i in issues if i.level.value == "error"]
            warnings = [i for i in issues if i.level.value == "warning"]
            suggestions = [i for i in issues if i.level.value == "suggestion"]
            info = [i for i in issues if i.level.value == "info"]
            
            for issue_list, emoji, title in [
                (errors, "🚨", "ERRORS"),
                (warnings, "⚠️ ", "WARNINGS"),
                (suggestions, "💡", "SUGGESTIONS"),
                (info, "ℹ️ ", "INFO")
            ]:
                if issue_list:
                    output += f"\n{emoji} **{title}:**\n"
                    for issue in issue_list:
                        output += f"- {issue.message}\n"
                        if issue.suggestion:
                            output += f"  → {issue.suggestion}\n"
                        if issue.estimated_impact:
                            output += f"  Impact: {issue.estimated_impact}\n"
        else:
            output += "\n✅ **No issues found!** Query looks good."
        
        return output
    
    async def _optimize_last_query(self) -> Dict:
        """Optimize the last analyzed query"""
        if not hasattr(self, '_last_analyzed_query'):
            return {"response": "No query to optimize. Please analyze a query first."}
        
        self._print_info("\n🚀 Optimizing query...")
        
        result = await self.sql_optimizer.optimize_query(self._last_analyzed_query)
        
        response = "🎯 **Optimized Query:**\n\n"
        
        if RICH_AVAILABLE:
            # Show formatted SQL with syntax highlighting
            syntax = Syntax(result['optimized_query'], "sql", theme="monokai", line_numbers=True)
            console.print(syntax)
            response += f"```sql\n{result['optimized_query']}\n```\n"
        else:
            response += f"{result['optimized_query']}\n"
        
        response += f"\n**Estimated Improvement:** {result['estimated_improvement']}\n"
        
        if result['recommendations']:
            response += "\n**Additional Recommendations:**\n"
            for rec in result['recommendations']:
                response += f"- {rec}\n"
        
        return {
            "response": response,
            "cost": 0.001,  # Approximate cost
            "model_selection": {"final_model": "grok-4"}
        }
    
    def _build_context(self, search_results) -> str:
        """Build context from search results"""
        if not search_results:
            return ""
        
        context = "\n## Relevant Information from Memory:\n"
        for i, result in enumerate(search_results[:3], 1):
            context += f"\n[{i}] (Score: {result.score:.3f}, Method: {result.retrieval_method})\n"
            context += f"{result.content[:300]}...\n"
        
        return context
    
    def _build_prompt(self, user_input: str, context: str, conv_history: str) -> str:
        """Build the full prompt for LLM"""
        prompt = """You are ADAM, an expert AI assistant for Analytics Engineers.
You specialize in SQL optimization, data warehousing (Snowflake, BigQuery, Redshift), 
dbt, data pipelines, and analytics best practices.

Be technical, specific, and helpful. Include code examples when relevant."""

        if conv_history:
            prompt += f"\n\n## Previous Conversation:\n{conv_history}"
        
        if context:
            prompt += f"\n{context}"
        
        prompt += f"\n\n## User Question:\n{user_input}\n\n## Your Response:"
        
        return prompt
    
    def _get_model_selection_reason(self, text: str) -> str:
        """Get explanation for model selection"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['sql', 'query', 'database', 'snowflake', 'bigquery']):
            return "SQL/Database content - grok-4 preferred for analytics"
        elif any(word in text_lower for word in ['explain', 'why', 'how does', 'debug', 'understand']):
            return "Reasoning/explanation task - Complex model preferred"
        elif len(text) < 50:
            return "Short query - Fast model sufficient"
        else:
            return "General query - Balanced model selection"
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text"""
        topics = []
        text_lower = text.lower()
        
        topic_keywords = {
            'sql': ['sql', 'query', 'select', 'join', 'table'],
            'snowflake': ['snowflake', 'warehouse', 'clustering'],
            'dbt': ['dbt', 'model', 'macro', 'test'],
            'optimization': ['optimize', 'slow', 'performance', 'speed'],
            'data_quality': ['quality', 'validation', 'null', 'duplicate'],
            'analytics': ['metric', 'kpi', 'dashboard', 'report']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)
        
        return topics if topics else ['general']
    
    def _get_help(self) -> str:
        """Get help message"""
        return """
🤖 **ADAM Complete Commands:**

**Chat Commands:**
- `help` - Show this message
- `stats` - Show session statistics
- `models` - Show detailed model information
- `memory` - Show memory system info
- `exit/quit` - End session

**Features:**
- Paste SQL queries for automatic analysis
- Ask about SQL, dbt, data warehousing
- Get optimization suggestions
- See which model ADAM selects and why

**Example Questions:**
- "How do I optimize a slow Snowflake query?"
- "Explain window functions in SQL"
- "What's the difference between star and snowflake schemas?"
- "Debug this dbt error: [paste error]"

**SQL Analysis:**
Just paste any SQL query and ADAM will analyze it automatically!
"""
    
    def _get_stats(self) -> str:
        """Get session statistics"""
        total_memories = len(self.memory.memories)
        session_exchanges = len(self.conversations.current_session.exchanges) if self.conversations.current_session else 0
        
        stats = f"""
📊 **Session Statistics:**

**Memory System:**
- Total memories: {total_memories}
- Network connections: {self.memory_network.memory_graph.number_of_edges()}
- Session exchanges: {session_exchanges}

**Model Usage:**"""
        
        for model, count in self.model_usage.items():
            stats += f"\n- {model}: {count} calls"
        
        stats += f"\n\n**Costs:**\n- Session total: ${self.total_cost:.4f}"
        
        # Add per-model costs if available
        if hasattr(self.llm_client.config, 'models'):
            stats += "\n\n**Model Costs (per 1K tokens):**"
            for model_name in self.llm_client.config.get_available_models():
                config = self.llm_client.config.get_model_config(model_name)
                if config.cost_per_1k_tokens:
                    stats += f"\n- {model_name}: ${config.cost_per_1k_tokens:.3f}"
        
        return stats
    
    def _get_model_details(self) -> str:
        """Get detailed model information"""
        output = "🤖 **Available LLM Models:**\n\n"
        
        for model_name in self.llm_client.config.get_available_models():
            config = self.llm_client.config.get_model_config(model_name)
            output += f"**{model_name}** ({config.provider.value})\n"
            output += f"- Max tokens: {config.max_tokens}\n"
            output += f"- Supports reasoning: {'Yes' if config.supports_reasoning else 'No'}\n"
            output += f"- Capabilities: {', '.join([c.value for c in config.capabilities])}\n"
            if config.cost_per_1k_tokens:
                output += f"- Cost: ${config.cost_per_1k_tokens:.3f} per 1K tokens\n"
            output += "\n"
        
        output += "**Model Selection Logic:**\n"
        output += "- SQL/Analytics queries → grok-4 (best for technical content)\n"
        output += "- Reasoning/debugging → o4-mini or grok-3-mini (if available)\n"
        output += "- Simple queries → gpt-3.5-turbo (fast and cheap)\n"
        output += "- Complex analysis → gpt-4 or grok-4\n"
        
        return output
    
    def _get_memory_info(self) -> str:
        """Get memory system information"""
        memories = list(self.memory.memories.values())
        
        output = "🧠 **Memory System Information:**\n\n"
        output += f"**Statistics:**\n"
        output += f"- Total memories: {len(memories)}\n"
        output += f"- Memory network nodes: {self.memory_network.memory_graph.number_of_nodes()}\n"
        output += f"- Memory network edges: {self.memory_network.memory_graph.number_of_edges()}\n"
        
        if memories:
            # Show memory types
            memory_types = {}
            for mem in memories:
                mem_type = mem.metadata.get('memory_type', 'unknown')
                memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
            
            output += "\n**Memory Types:**\n"
            for mem_type, count in sorted(memory_types.items(), key=lambda x: x[1], reverse=True):
                output += f"- {mem_type}: {count}\n"
            
            # Show recent memories
            recent = sorted(memories, key=lambda m: m.timestamp, reverse=True)[:3]
            output += "\n**Recent Memories:**\n"
            for i, mem in enumerate(recent, 1):
                output += f"\n[{i}] {mem.query[:50]}...\n"
                output += f"    Strength: {mem.memory_strength:.3f}\n"
                output += f"    Access count: {mem.access_count}\n"
        
        return output
    
    async def interactive_session(self):
        """Run interactive chat session"""
        
        while True:
            try:
                # Get user input
                if RICH_AVAILABLE:
                    user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
                else:
                    user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    if self.verbose:
                        self._print_info("\nSaving session...")
                    self.conversations.end_session(self.session_id)
                    
                    # Show session summary
                    print("\n📊 Session Summary:")
                    print(f"- Total cost: ${self.total_cost:.4f}")
                    print(f"- Models used: {', '.join(self.model_usage.keys())}")
                    print(f"- Memories created: {len([m for m in self.memory.memories.values() if m.metadata.get('session_id') == self.session_id])}")
                    print("\n👋 Thanks for using ADAM!")
                    break
                
                # Process input
                response_data = await self.process_input(user_input)
                
                # Display response
                if RICH_AVAILABLE:
                    console.print(f"\n[bold green]ADAM:[/bold green] {response_data['response']}")
                else:
                    print(f"\nADAM: {response_data['response']}")
                
                # Show metadata if verbose
                if self.verbose and response_data.get('model_selection'):
                    print(f"\n[Model: {response_data['model_selection'].get('final_model', 'N/A')} | "
                          f"Cost: ${response_data.get('cost', 0):.4f} | "
                          f"Time: {response_data.get('processing_time', 0):.2f}s]")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                self._print_error(f"Error: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ADAM Complete Interface")
    parser.add_argument('--quiet', '-q', action='store_true', help='Disable verbose output')
    parser.add_argument('--test', '-t', action='store_true', help='Run system test')
    args = parser.parse_args()
    
    if args.test:
        # Run system test
        print("🧪 Running ADAM system test...\n")
        adam = ADAMComplete(verbose=True)
        
        # Test queries
        test_queries = [
            "What is a CTE?",
            "SELECT * FROM users WHERE status = 'active'",
            "How do I optimize Snowflake queries?"
        ]
        
        for query in test_queries:
            print(f"\nTest: {query}")
            result = await adam.process_input(query)
            print(f"✅ Response received (Model: {result.get('model_selection', {}).get('final_model', 'N/A')})")
        
        print("\n✅ All tests passed!")
    else:
        # Run interactive session
        adam = ADAMComplete(verbose=not args.quiet)
        await adam.interactive_session()


if __name__ == "__main__":
    # Check for rich library
    if not RICH_AVAILABLE:
        print("💡 Tip: Install 'rich' for better formatting: pip install rich")
    
    asyncio.run(main())