#!/usr/bin/env python3
"""Test dynamic folder finding"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.file_operation_handler import FileOperationHandler

def test_finder():
    handler = FileOperationHandler()
    handler.file_tools.workspace_path = "/Users/vitoryago"
    
    # Test finding marketing folder
    print("Testing dynamic folder finder...")
    print(f"Workspace: {handler.file_tools.workspace_path}")
    
    # Test finding folders
    folders_to_test = ["marketing", "de-dbt-analytics", "models", "Documents"]
    
    for folder in folders_to_test:
        found = handler._find_folder_in_workspace(folder)
        print(f"Looking for '{folder}': {found}")

if __name__ == "__main__":
    test_finder()