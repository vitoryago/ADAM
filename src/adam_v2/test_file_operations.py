#!/usr/bin/env python3
"""Test file operations for ADAM"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.file_operation_handler import FileOperationHandler

def test_marketing_files():
    """Test listing marketing _inc.sql files"""
    handler = FileOperationHandler()
    
    # Set a test workspace path (adjust as needed)
    handler.file_tools.workspace_path = "/Users/vitoryago"
    
    # Test queries
    test_queries = [
        "list all the _inc.sql files in the marketing folder",
        "check the marketing folder for _inc models",
        "go through the marketing models and list all _inc.sql files",
        "show me all _inc files in de-dbt-analytics/marketing/models/main/tables"
    ]
    
    print("Testing file operations for marketing folder...")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        result = handler.process_query(query)
        
        if result.get('operations_performed'):
            print(f"Operations: {result['operations_performed']}")
            
        if result.get('formatted_output'):
            print("Output:")
            print(result['formatted_output'])
        else:
            print("No output generated")
            
        if result.get('raw_results'):
            for raw in result['raw_results']:
                if raw and raw.get('error'):
                    print(f"Error: {raw['error']}")

if __name__ == "__main__":
    test_marketing_files()