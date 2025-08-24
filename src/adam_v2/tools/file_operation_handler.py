"""
File Operation Handler
Intelligently processes file-related queries
"""

import os
import re
import logging
from typing import Dict, Any, Optional, List
from tools.file_system_tools import FileSystemTools

logger = logging.getLogger(__name__)

class FileOperationHandler:
    """Handles file operations based on query intent"""
    
    def __init__(self, workspace_path: Optional[str] = None):
        self.file_tools = FileSystemTools(workspace_path)
        # Default workspace if not provided
        if not workspace_path:
            self.file_tools.workspace_path = "/Users/vitoryago"
        
    def process_query(self, query: str, content: str = "") -> Dict[str, Any]:
        """
        Process a query and execute any file operations needed
        
        Returns:
            Dict with operation results and formatted output
        """
        query_lower = query.lower()
        results = {
            "operations_performed": [],
            "formatted_output": "",
            "raw_results": []
        }
        
        # Pattern matching for different file operations
        patterns = {
            # Looking for files with specific patterns
            r"(check|list|show|find|go through).*(folder|directory|files).*(_inc|\.sql|\.py)": self._handle_file_search,
            # Specific folder + pattern combinations
            r"(marketing|models|src|tests).*(folder|files|models).*(_inc|\.sql)": self._handle_folder_pattern_search,
            # General directory listing
            r"(list|show|what's in|check).*(folder|directory|path)": self._handle_directory_listing,
            # File reading
            r"(read|open|show|view).*(file|\.sql|\.py|\.txt)": self._handle_file_reading,
            # File creation/writing
            r"(create|write|make|generate|save).*(file|\.sql|\.py|model)": self._handle_file_creation,
        }
        
        # Check each pattern
        for pattern, handler in patterns.items():
            if re.search(pattern, query_lower):
                # Pass content for file creation handlers
                if handler == self._handle_file_creation:
                    result = handler(query, content)
                else:
                    result = handler(query)
                if result:
                    results["operations_performed"].append(result["operation"])
                    results["formatted_output"] += result["output"]
                    results["raw_results"].append(result["raw"])
        
        # If no pattern matched but query mentions files/folders, do a general search
        if not results["operations_performed"] and any(word in query_lower for word in ['file', 'folder', 'directory', 'model']):
            result = self._handle_general_search(query)
            if result:
                results["operations_performed"].append(result["operation"])
                results["formatted_output"] += result["output"]
                results["raw_results"].append(result["raw"])
        
        return results
    
    def _handle_file_search(self, query: str) -> Optional[Dict[str, Any]]:
        """Handle searching for files with specific patterns"""
        query_lower = query.lower()
        
        # Extract pattern and path
        pattern = "*_inc.sql" if "_inc" in query_lower else "*.sql"
        search_path = ""
        
        # Extract folder path if mentioned
        # Build path dynamically based on what's mentioned in the query
        path_parts = []
        
        # Check for project-specific keywords
        if "de-dbt-analytics" in query_lower or "dbt-analytics" in query_lower:
            # Search for this specific project folder anywhere in workspace
            search_path = self._find_folder_in_workspace("de-dbt-analytics")
            if not search_path:
                search_path = "de-dbt-analytics"
        elif "marketing" in query_lower and "models" in query_lower:
            # Look for marketing/models pattern
            search_path = self._find_folder_in_workspace("marketing/models")
            if not search_path:
                search_path = "marketing/models"
        elif "marketing" in query_lower:
            search_path = self._find_folder_in_workspace("marketing")
            if not search_path:
                search_path = "marketing"
        elif "models" in query_lower:
            search_path = self._find_folder_in_workspace("models")
            if not search_path:
                search_path = "models"
        else:
            search_path = ""
        
        # Execute search
        result = self.file_tools.find_files(pattern, search_path)
        
        if result.get('error'):
            # Log the error and try alternative paths
            logger.warning(f"Search failed in {search_path}: {result.get('error')}")
            
            # Try alternative paths - search dynamically
            alt_paths = []
            
            # Try to find common folder patterns
            if "_inc" in pattern or "marketing" in query_lower:
                # Look for marketing-related paths
                marketing_path = self._find_folder_in_workspace("marketing")
                if marketing_path:
                    alt_paths.append(marketing_path)
                    alt_paths.append(os.path.join(marketing_path, "models"))
                    alt_paths.append(os.path.join(marketing_path, "models/main/tables"))
            
            # Add generic fallbacks
            alt_paths.extend([
                "models",
                "src",
                "lib",
                "."  # Current directory
            ])
            for alt_path in alt_paths:
                if alt_path != search_path:
                    result = self.file_tools.find_files(pattern, alt_path)
                    if not result.get('error'):
                        search_path = alt_path
                        break
            
            if result.get('error'):
                return None
        
        # Format output
        output = f"\n[File Search Results - Pattern: {pattern} in {search_path or 'workspace'}]\n"
        matches = result.get('matches', [])
        
        if matches:
            output += f"Found {len(matches)} files:\n\n"
            
            # Read and summarize each file
            for match in matches[:10]:  # Limit to 10 files
                file_path = match['path']
                output += f"📄 **{file_path}**\n"
                
                # Read file for summary
                file_content = self.file_tools.read_file(file_path, max_lines=30)
                if not file_content.get('error'):
                    content = file_content['content']
                    # Extract key information
                    output += self._extract_file_summary(content, file_path)
                    output += "\n"
        else:
            output += "No files found matching the pattern.\n"
        
        return {
            "operation": "file_search",
            "output": output,
            "raw": result
        }
    
    def _handle_folder_pattern_search(self, query: str) -> Optional[Dict[str, Any]]:
        """Handle searching in specific folders with patterns"""
        query_lower = query.lower()
        
        # Determine folder and pattern
        folder = ""
        pattern = "*_inc.sql" if "_inc" in query_lower else "*.sql"
        
        if "marketing" in query_lower:
            # Find marketing folder dynamically
            folder = self._find_folder_in_workspace("marketing")
            if not folder:
                folder = "marketing"  # Fallback
        elif "staging" in query_lower:
            folder = self._find_folder_in_workspace("staging")
            if not folder:
                folder = "staging"  # Fallback
        
        if not folder:
            return None
        
        # List directory with pattern
        result = self.file_tools.list_directory(folder, pattern)
        
        if result.get('error'):
            # Try find_files as fallback
            result = self.file_tools.find_files(pattern, folder)
            items = result.get('matches', [])
        else:
            items = result.get('items', [])
        
        # Format output
        output = f"\n[Files in {folder} matching {pattern}]\n"
        
        if items:
            output += f"Found {len(items)} files:\n\n"
            for item in items[:15]:
                file_path = item.get('path', item.get('name', ''))
                output += f"• {file_path}\n"
                
                # Read and summarize _inc files
                if '_inc' in file_path:
                    file_content = self.file_tools.read_file(file_path, max_lines=20)
                    if not file_content.get('error'):
                        output += self._extract_inc_summary(file_content['content'])
        else:
            output += "No matching files found.\n"
        
        return {
            "operation": "folder_pattern_search",
            "output": output,
            "raw": result
        }
    
    def _handle_directory_listing(self, query: str) -> Optional[Dict[str, Any]]:
        """Handle directory listing requests"""
        # Extract path from query
        path = self._extract_path_from_query(query)
        
        result = self.file_tools.list_directory(path)
        
        if result.get('error'):
            return None
        
        output = f"\n[Directory: {path or 'workspace root'}]\n"
        items = result.get('items', [])
        
        if items:
            dirs = [i for i in items if i['type'] == 'directory']
            files = [i for i in items if i['type'] == 'file']
            
            if dirs:
                output += "\nFolders:\n"
                for d in dirs[:10]:
                    output += f"📁 {d['name']}\n"
            
            if files:
                output += "\nFiles:\n"
                for f in files[:20]:
                    output += f"📄 {f['name']}\n"
        else:
            output += "Empty directory.\n"
        
        return {
            "operation": "directory_listing",
            "output": output,
            "raw": result
        }
    
    def _handle_file_reading(self, query: str) -> Optional[Dict[str, Any]]:
        """Handle file reading requests"""
        # Extract file path from query
        file_path = self._extract_file_path_from_query(query)
        
        if not file_path:
            return None
        
        result = self.file_tools.read_file(file_path)
        
        if result.get('error'):
            return None
        
        output = f"\n[File: {file_path}]\n"
        output += f"```sql\n{result['content'][:2000]}\n```\n"
        
        if result.get('truncated'):
            output += f"(Showing first {result.get('lines_shown', 0)} of {result.get('total_lines', 0)} lines)\n"
        
        return {
            "operation": "file_reading",
            "output": output,
            "raw": result
        }
    
    def _handle_file_creation(self, query: str, content: str = "") -> Optional[Dict[str, Any]]:
        """Handle file creation requests"""
        # Extract file path from query
        file_path = self._extract_file_path_from_query(query)
        
        if not file_path:
            # Try to build path from context
            query_lower = query.lower()
            if "_inc" in query_lower and ".sql" in query_lower:
                # Creating an incremental SQL model
                if "marketing" in query_lower:
                    base_path = self._find_folder_in_workspace("marketing/models/main/tables")
                    if base_path:
                        # Extract filename from query or generate one
                        import re
                        filename_match = re.search(r'(\w+_inc)\.sql', query_lower)
                        if filename_match:
                            filename = filename_match.group(0)
                        else:
                            filename = "new_model_inc.sql"
                        file_path = os.path.join(base_path, filename)
                    else:
                        file_path = "new_model_inc.sql"
                else:
                    file_path = "new_model_inc.sql"
            else:
                return None
        
        # Write the file
        result = self.file_tools.write_file(file_path, content)
        
        if result.get('error'):
            return None
        
        output = f"\n[File Created: {file_path}]\n"
        output += f"✅ Successfully created file: {result.get('full_path', file_path)}\n"
        output += f"Size: {result.get('size', 0)} bytes\n"
        
        return {
            "operation": "file_creation",
            "output": output,
            "raw": result
        }
    
    def _handle_general_search(self, query: str) -> Optional[Dict[str, Any]]:
        """Handle general file/folder queries"""
        # Try to be helpful with a general search
        return None
    
    def _extract_file_summary(self, content: str, file_path: str) -> str:
        """Extract summary from file content"""
        summary = ""
        lines = content.split('\n')[:10]
        
        # For SQL files
        if file_path.endswith('.sql'):
            # Look for config block
            if '{{' in content:
                config_match = re.search(r'\{\{(.*?)\}\}', content, re.DOTALL)
                if config_match:
                    summary += f"  Config: {config_match.group(1)[:100]}\n"
            
            # Look for CTEs or main query
            if 'with' in content.lower() or 'select' in content.lower():
                summary += "  Type: SQL transformation\n"
            
            # Check for incremental
            if 'is_incremental()' in content:
                summary += "  Type: Incremental model\n"
        
        return summary or "  (File preview available)\n"
    
    def _extract_inc_summary(self, content: str) -> str:
        """Extract summary specifically for _inc files"""
        summary = "  → Incremental model"
        
        # Look for key patterns
        if 'delete' in content.lower():
            summary += " with deletes"
        if 'merge' in content.lower():
            summary += " using merge"
        if 'is_incremental()' in content:
            if 'where' in content.lower():
                summary += " with timestamp filter"
        
        return summary + "\n"
    
    def _extract_path_from_query(self, query: str) -> str:
        """Extract path from query"""
        query_lower = query.lower()
        
        if "marketing" in query_lower:
            # Find marketing folder dynamically
            path = self._find_folder_in_workspace("marketing")
            return path if path else "marketing"
        elif "staging" in query_lower:
            path = self._find_folder_in_workspace("staging")
            return path if path else "staging"
        elif "models" in query_lower:
            path = self._find_folder_in_workspace("models")
            return path if path else "models"
        
        return ""
    
    def _find_folder_in_workspace(self, folder_name: str) -> Optional[str]:
        """Find a folder anywhere in the workspace"""
        import os
        from pathlib import Path
        
        # Start from workspace root
        workspace_root = Path(self.file_tools.workspace_path)
        
        # First check if the folder exists directly
        direct_path = workspace_root / folder_name
        if direct_path.exists() and direct_path.is_dir():
            return folder_name
        
        # Search for the folder recursively (limit depth for performance)
        for root, dirs, files in os.walk(workspace_root):
            # Limit search depth to avoid searching too deep
            depth = root.count(os.sep) - str(workspace_root).count(os.sep)
            if depth > 5:  # Don't go more than 5 levels deep
                dirs[:] = []  # Don't recurse further
                continue
                
            for dir_name in dirs:
                full_path = os.path.join(root, dir_name)
                # Check if this directory ends with or contains our target
                if full_path.endswith(folder_name) or folder_name in full_path:
                    # Return relative path from workspace
                    return os.path.relpath(full_path, workspace_root)
        
        return None
    
    def _extract_file_path_from_query(self, query: str) -> Optional[str]:
        """Extract file path from query"""
        # Look for file paths in the query
        # This is simplified - could use more sophisticated parsing
        words = query.split()
        for word in words:
            if '.sql' in word or '.py' in word:
                return word
        return None