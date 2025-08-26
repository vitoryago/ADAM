#!/usr/bin/env python3
"""
Test script for DBT Knowledge Integration
Tests that ADAM properly detects DBT queries and enhances them with knowledge
"""

import asyncio
import sys
from pathlib import Path

# Add src path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_dbt_knowledge():
    """Test DBT knowledge service"""
    print("Testing DBT Knowledge Integration...")
    print("=" * 50)
    
    # Test 1: DBT Knowledge Service standalone
    try:
        from adam_v2.services.dbt_knowledge_service import DBTKnowledgeService, ModelLayer, DBTModelRequest
        
        dbt_service = DBTKnowledgeService()
        print("✓ DBT Knowledge Service initialized")
        
        # Test detection
        test_queries = [
            "Create a DBT model from this PDT",
            "How do I write an incremental model?",
            "What's the best practice for staging layers?",
            "Generate a fact table for user transactions",
            "Regular query without DBT context"
        ]
        
        for query in test_queries:
            detected = dbt_service.detect_dbt_context(query)
            print(f"  Query: '{query[:40]}...' → DBT: {detected}")
        
        # Test knowledge retrieval
        knowledge = dbt_service.get_relevant_knowledge("Create an incremental DBT model")
        print(f"\n✓ Knowledge retrieval: Found {len(knowledge['detected_need'])} needs")
        for need in knowledge['detected_need']:
            print(f"  - {need}")
        
        # Test model generation
        request = DBTModelRequest(
            model_name="stg_test__users",
            layer=ModelLayer.STAGING,
            source_table="users",
            source_schema="raw",
            grain="one row per user"
        )
        
        model_sql = dbt_service.generate_model_template(request)
        print(f"\n✓ Model generation: Created {len(model_sql.split(chr(10)))} lines of SQL")
        print("  First 3 lines:")
        for line in model_sql.split('\n')[:3]:
            print(f"    {line}")
        
    except Exception as e:
        print(f"✗ DBT Knowledge Service error: {e}")
        return False
    
    print("\n" + "=" * 50)
    
    # Test 2: Integration with LLM Service
    try:
        from adam_v2.services.llm_service import LLMService
        
        # Initialize LLM service (it will auto-load DBT knowledge)
        llm_service = LLMService(project_id="test_project")
        
        if llm_service.dbt_knowledge:
            print("✓ DBT Knowledge integrated with LLM Service")
            
            # Test that DBT context is detected and enhanced
            test_message = "Convert this Looker PDT to a DBT incremental model"
            
            # Check if it would be enhanced
            if llm_service.dbt_knowledge.detect_dbt_context(test_message):
                print(f"✓ DBT context detected for: '{test_message}'")
                
                # Get enhanced prompt
                enhanced = llm_service.dbt_knowledge.enhance_query_with_dbt_context(test_message)
                if "[DBT Knowledge Base Context]" in enhanced:
                    print("✓ Query enhanced with DBT context")
                    print(f"  Original length: {len(test_message)}")
                    print(f"  Enhanced length: {len(enhanced)}")
            else:
                print("✗ DBT context not detected")
        else:
            print("⚠ DBT Knowledge not loaded in LLM Service (files may be missing)")
            
    except Exception as e:
        print(f"✗ LLM Service integration error: {e}")
        return False
    
    print("\n" + "=" * 50)
    
    # Test 3: Check knowledge files exist
    knowledge_files = [
        Path("DBT_KNOWLEDGE.md"),
        Path("dbt_patterns.yaml")
    ]
    
    print("Checking knowledge files:")
    all_exist = True
    for file in knowledge_files:
        if file.exists():
            size = file.stat().st_size
            print(f"  ✓ {file.name}: {size:,} bytes")
        else:
            print(f"  ✗ {file.name}: NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n⚠ Some knowledge files are missing. DBT integration may not work fully.")
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("  - DBT Knowledge Service: ✓")
    print("  - LLM Integration: ✓" if llm_service.dbt_knowledge else "⚠")
    print("  - Knowledge Files: ✓" if all_exist else "⚠")
    
    return True

async def test_example_conversion():
    """Test converting a Looker PDT to DBT model"""
    print("\n" + "=" * 50)
    print("Example: Converting Looker PDT to DBT Model")
    print("=" * 50)
    
    try:
        from adam_v2.services.dbt_knowledge_service import DBTKnowledgeService, ModelLayer, DBTModelRequest
        
        dbt_service = DBTKnowledgeService()
        
        # Create a request for fact table
        request = DBTModelRequest(
            model_name="fct_user_transactions",
            layer=ModelLayer.MART_FACT,
            source_table="int_transactions__enriched",
            business_logic="Aggregate user transactions with revenue calculations",
            grain="one row per transaction",
            unique_key="transaction_id",
            timestamp_column="created_at",
            cluster_keys=["user_id", "created_at"],
            domain="finance"
        )
        
        # Generate the model
        model_sql = dbt_service.generate_model_template(request)
        
        print("Generated DBT Model:")
        print("-" * 40)
        print(model_sql)
        print("-" * 40)
        
        # Get suggestions
        suggestions = dbt_service.suggest_improvements(model_sql)
        if suggestions:
            print("\nSuggested Improvements:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
        else:
            print("\n✓ No improvements needed!")
            
    except Exception as e:
        print(f"Error in example conversion: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("\n🤖 ADAM DBT Knowledge Integration Test\n")
    
    # Run tests
    asyncio.run(test_dbt_knowledge())
    asyncio.run(test_example_conversion())
    
    print("\n✅ All tests completed!")
    print("\nNext steps:")
    print("1. Restart the backend to load DBT knowledge")
    print("2. Ask ADAM to convert a Looker PDT to DBT")
    print("3. ADAM will automatically use DBT best practices!")
    print("\nExample queries to test:")
    print("  - 'Convert this PDT to a DBT incremental model'")
    print("  - 'Create a staging model for user data from Snowflake'")
    print("  - 'What are DBT best practices for fact tables?'")
    print("  - 'Write a macro for safe division in DBT'")