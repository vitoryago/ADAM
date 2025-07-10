#!/usr/bin/env python3
"""
Interactive ADAM Chat - Test ADAM's capabilities
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Setup
load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adam.memory import ADAMMemoryAdvanced
from adam.conversation_system import ConversationSystem
from adam.advanced_rag import AdvancedRAGSystem
from adam.memory_network import MemoryNetworkSystem
from adam.llm.client import UnifiedLLMClient
from adam.tools.sql_tools import SQLAnalyzer, SQLOptimizer


class ADAMChat:
    """Interactive ADAM system combining all capabilities"""
    
    def __init__(self):
        print("🧠 Initializing ADAM...")
        
        # Initialize core systems
        self.memory = ADAMMemoryAdvanced(persist_directory="./adam_chat_memory")
        self.conversations = ConversationSystem(storage_path="./adam_chat_conversations")
        self.memory_network = MemoryNetworkSystem(self.memory, self.conversations)
        self.rag = AdvancedRAGSystem(self.memory, self.memory_network)
        self.llm_client = UnifiedLLMClient()
        
        # Initialize tools
        self.sql_analyzer = SQLAnalyzer("snowflake")
        self.sql_optimizer = SQLOptimizer("snowflake")
        
        # Start conversation session
        self.session_id = self.conversations.start_session("Interactive ADAM Chat")
        
        # Pre-seed some analytics knowledge
        self._seed_knowledge()
        
        print("✅ ADAM is ready! Type 'help' for commands or just start chatting.\n")
        
    def _seed_knowledge(self):
        """Seed ADAM with some initial analytics knowledge"""
        analytics_facts = [
            ("What is a CTE in SQL?", 
             "A CTE (Common Table Expression) is a temporary named result set that exists within the scope of a single SQL statement. CTEs are defined using the WITH clause and improve query readability and maintainability."),
            
            ("How do I optimize Snowflake queries?",
             "Key Snowflake optimizations: 1) Use clustering keys on filter columns, 2) Avoid SELECT *, 3) Use result caching, 4) Partition large tables, 5) Use materialized views for complex aggregations, 6) Monitor query profiles for bottlenecks."),
            
            ("What's the difference between UNION and UNION ALL?",
             "UNION removes duplicate rows and requires sorting (expensive), while UNION ALL keeps all rows including duplicates (faster). Use UNION ALL when you know there are no duplicates or when duplicates are acceptable."),
        ]
        
        for query, response in analytics_facts:
            self.memory.remember_if_worthy(
                query=query,
                response=response,
                context={"type": "analytics_knowledge", "source": "seed"},
                generation_cost=0.001,
                model_used="seed"
            )
    
    async def process_message(self, user_input: str) -> str:
        """Process user input and generate response"""
        
        # Check for special commands
        if user_input.lower() == 'help':
            return self._get_help_message()
        elif user_input.lower() == 'stats':
            return self._get_stats()
        elif user_input.lower() == 'clear':
            return self._clear_screen()
        elif user_input.lower() in ['exit', 'quit', 'bye']:
            return "👋 Goodbye! Your conversation has been saved."
        
        # Check if it's a SQL query for analysis
        if self._looks_like_sql(user_input):
            return await self._analyze_sql(user_input)
        
        # Search memory for relevant information
        search_results = self.rag.retrieve(user_input, k=3)
        
        # Build context from memory
        context = ""
        if search_results:
            context = "\n\nRelevant information from memory:\n"
            for result in search_results[:2]:
                context += f"- {result.content[:200]}...\n"
        
        # Get conversation history
        conv_context = self.conversations.get_conversation_context(lookback_exchanges=3)
        
        # Use LLM to generate response
        prompt = f"""You are ADAM, an AI assistant specialized in Analytics Engineering, SQL, and data pipelines.

Previous conversation:
{conv_context if conv_context else "No previous context"}

{context}

User question: {user_input}

Provide a helpful, concise response. If the question is about SQL or data, be specific and technical."""

        try:
            response = await self.llm_client.complete(
                prompt=prompt,
                model=None,  # Auto-select
                temperature=0.7
            )
            
            llm_response = response.content
            
            # Record the exchange
            self.conversations.record_exchange(
                query=user_input,
                response=llm_response,
                topics=self._extract_topics(user_input),
                metadata={
                    "model": response.model,
                    "tokens": response.total_tokens,
                    "memory_results": len(search_results)
                }
            )
            
            # Store in memory if it's valuable
            self.memory.remember_if_worthy(
                query=user_input,
                response=llm_response,
                context={
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat()
                },
                generation_cost=response.cost,
                model_used=response.model
            )
            
            # Add metadata footer
            footer = f"\n\n[Model: {response.model} | Memory hits: {len(search_results)} | Cost: ${response.cost:.4f}]"
            
            return llm_response + footer
            
        except Exception as e:
            return f"❌ Error: {str(e)}\n\nMake sure your API keys are set correctly."
    
    def _looks_like_sql(self, text: str) -> bool:
        """Check if input looks like SQL"""
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'CREATE', 'INSERT', 'UPDATE', 'DELETE']
        text_upper = text.upper()
        return any(keyword in text_upper for keyword in sql_keywords)
    
    async def _analyze_sql(self, query: str) -> str:
        """Analyze SQL query"""
        print("🔍 Analyzing SQL query...")
        
        # Analyze the query
        issues, metrics = self.sql_analyzer.analyze_query(query)
        
        response = "📊 SQL Analysis Results:\n\n"
        response += f"**Query Metrics:**\n"
        response += f"- Complexity: {metrics.complexity_score}/10\n"
        response += f"- Lines: {metrics.line_count}\n"
        response += f"- CTEs: {metrics.cte_count}\n"
        response += f"- Joins: {metrics.join_count}\n"
        
        if issues:
            response += f"\n**Issues Found ({len(issues)}):**\n"
            for i, issue in enumerate(issues, 1):
                response += f"\n{i}. [{issue.level.value.upper()}] {issue.message}\n"
                if issue.suggestion:
                    response += f"   💡 {issue.suggestion}\n"
        else:
            response += "\n✅ No issues found!"
        
        # Offer optimization
        if issues and len(issues) > 2:
            response += "\n\nWould you like me to optimize this query? (Type 'yes' or paste another query)"
            
        return response
    
    def _extract_topics(self, text: str) -> list:
        """Extract topics from text"""
        topics = []
        
        # SQL/Database topics
        if any(word in text.lower() for word in ['sql', 'query', 'database', 'table']):
            topics.append('sql')
        
        # Analytics topics  
        if any(word in text.lower() for word in ['snowflake', 'bigquery', 'redshift', 'warehouse']):
            topics.append('data_warehouse')
            
        # dbt topics
        if 'dbt' in text.lower():
            topics.append('dbt')
            
        # General analytics
        if any(word in text.lower() for word in ['analytics', 'metrics', 'kpi', 'dashboard']):
            topics.append('analytics')
            
        return topics if topics else ['general']
    
    def _get_help_message(self) -> str:
        """Get help message"""
        return """
🤖 ADAM Commands:
- **help** - Show this message
- **stats** - Show memory and conversation statistics  
- **clear** - Clear the screen
- **exit/quit** - End the session

💡 You can:
- Ask questions about SQL, analytics, data engineering
- Paste SQL queries for analysis
- Get help with dbt, Snowflake, BigQuery
- Learn about optimization techniques

📝 Examples:
- "How do I optimize a slow Snowflake query?"
- "What's the difference between UNION and UNION ALL?"
- "SELECT * FROM orders WHERE status = 'pending'" (I'll analyze it)
- "Explain window functions in SQL"
"""
    
    def _get_stats(self) -> str:
        """Get system statistics"""
        memories = len(self.memory.memories)
        exchanges = len(self.conversations.current_session.exchanges) if self.conversations.current_session else 0
        
        return f"""
📊 ADAM Statistics:
- Memories stored: {memories}
- Exchanges in session: {exchanges}
- Available LLM models: {', '.join(self.llm_client.config.get_available_models())}
- Memory network connections: {self.memory_network.memory_graph.number_of_edges()}
"""
    
    def _clear_screen(self) -> str:
        """Clear screen command"""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
        return "Screen cleared. Continue chatting!"
    
    async def chat_loop(self):
        """Main chat loop"""
        print("💬 Chat with ADAM about analytics, SQL, and data engineering!\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\n" + await self.process_message(user_input))
                    break
                
                # Process and respond
                print("\nADAM: ", end="", flush=True)
                response = await self.process_message(user_input)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Let's continue...\n")


async def main():
    """Main entry point"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                    🧠 ADAM CHAT 🧠                        ║
║        Analytics Data Assistant & Manager                 ║
║                                                          ║
║  Your AI companion for SQL, dbt, and data engineering    ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Create and run ADAM
    adam = ADAMChat()
    await adam.chat_loop()
    
    # Cleanup
    print("\nSaving conversation...")
    adam.conversations.end_session(adam.session_id)
    print("Session saved. Thanks for chatting with ADAM!")


if __name__ == "__main__":
    asyncio.run(main())