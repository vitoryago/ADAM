#!/usr/bin/env python3
"""Test file operations with the correct GitHub path"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.file_operation_handler import FileOperationHandler

def test_marketing_files():
    """Test listing marketing _inc.sql files with correct path"""
    handler = FileOperationHandler()
    
    # Set the correct workspace path
    handler.file_tools.workspace_path = "/Users/vitoryago"
    
    # Test query similar to what the user asked
    query = "go through the folder marketing and bring all the sql models with _inc in the end"
    
    print(f"Testing query: {query}")
    print("=" * 60)
    
    result = handler.process_query(query)
    
    if result.get('operations_performed'):
        print(f"Operations: {result['operations_performed']}")
        
    if result.get('formatted_output'):
        print("Output:")
        print(result['formatted_output'][:2000])  # First 2000 chars
    else:
        print("No output generated")
        
    if result.get('raw_results'):
        for raw in result['raw_results']:
            if raw and raw.get('error'):
                print(f"Error: {raw['error']}")
            elif raw and raw.get('matches'):
                print(f"Found {len(raw['matches'])} files")

if __name__ == "__main__":
    test_marketing_files()