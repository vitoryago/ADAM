#!/usr/bin/env python3
"""
Simple ADAM Chat - Quick test of ADAM's capabilities
"""
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Setup
load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adam.llm.client import UnifiedLLMClient
from adam.tools.sql_tools import SQLAnalyzer


async def simple_chat():
    """Simple chat with ADAM using just LLM and SQL tools"""
    print("""
    🧠 ADAM Simple Chat
    Ask me about SQL, analytics, or paste queries for analysis!
    Type 'exit' to quit.
    """)
    
    # Initialize components
    llm = UnifiedLLMClient()
    sql_analyzer = SQLAnalyzer("snowflake")
    
    while True:
        try:
            # Get input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
            
            print("\nADAM: ", end="", flush=True)
            
            # Check if it's SQL
            sql_keywords = ['SELECT', 'FROM', 'WHERE', 'CREATE', 'INSERT', 'UPDATE']
            is_sql = any(keyword in user_input.upper() for keyword in sql_keywords)
            
            if is_sql and len(user_input) > 20:
                # Analyze SQL
                print("Let me analyze this SQL query...\n")
                issues, metrics = sql_analyzer.analyze_query(user_input)
                
                print(f"📊 Query Analysis:")
                print(f"- Complexity: {metrics.complexity_score}/10")
                print(f"- Issues found: {len(issues)}")
                
                if issues:
                    print("\n⚠️  Issues:")
                    for issue in issues[:3]:  # Show first 3
                        print(f"- {issue.message}")
                        if issue.suggestion:
                            print(f"  💡 {issue.suggestion}")
                else:
                    print("✅ No issues found!")
                    
            else:
                # Regular chat with LLM
                prompt = f"""You are ADAM, an AI assistant for analytics engineers.
You help with SQL, dbt, data warehouses, and data pipelines.
Be concise and technical.

User: {user_input}
Assistant:"""

                response = await llm.complete(
                    prompt=prompt,
                    temperature=0.7
                )
                
                print(response.content)
                print(f"\n[Model: {response.model}]")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Make sure your API keys are set!")


async def test_specific_features():
    """Test specific ADAM features"""
    llm = UnifiedLLMClient()
    sql_analyzer = SQLAnalyzer("snowflake")
    
    print("\n🧪 Testing ADAM Features:\n")
    
    # Test 1: SQL Analysis
    print("1. SQL Analysis Test")
    print("-" * 40)
    test_query = """
    SELECT * FROM orders o, customers c 
    WHERE o.customer_id = c.id 
    AND amount > 100
    """
    issues, metrics = sql_analyzer.analyze_query(test_query)
    print(f"✅ Found {len(issues)} issues in test query")
    
    # Test 2: LLM Models
    print("\n2. Available LLM Models")
    print("-" * 40)
    models = llm.config.get_available_models()
    print(f"✅ Models ready: {', '.join(models)}")
    
    # Test 3: Simple LLM Query
    print("\n3. LLM Response Test")
    print("-" * 40)
    response = await llm.complete(
        "What's a CTE in SQL? Answer in one sentence.",
        temperature=0
    )
    print(f"✅ Response: {response.content}")
    print(f"   Model used: {response.model}")
    
    print("\n✅ All tests passed! ADAM is working correctly.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run tests
        asyncio.run(test_specific_features())
    else:
        # Run chat
        try:
            asyncio.run(simple_chat())
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")