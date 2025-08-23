"""
File Operation Handler
Intelligently processes file-related queries
"""

import re
import logging
from typing import Dict, Any, Optional, List
from tools.file_system_tools import FileSystemTools

logger = logging.getLogger(__name__)

class FileOperationHandler:
    """Handles file operations based on query intent"""
    
    def __init__(self, workspace_path: Optional[str] = None):
        self.file_tools = FileSystemTools(workspace_path)
        
    def process_query(self, query: str) -> Dict[str, Any]:
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
        }
        
        # Check each pattern
        for pattern, handler in patterns.items():
            if re.search(pattern, query_lower):
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
        if "marketing" in query_lower:
            # Check for full path patterns
            if "de-dbt-analytics" in query_lower or "dbt" in query_lower:
                search_path = "de-dbt-analytics/marketing/models/main/tables"
            elif "marketing/models" in query_lower:
                search_path = "marketing/models/main/tables"
            else:
                # Default to the most likely path
                search_path = "de-dbt-analytics/marketing/models/main/tables"
        elif "models" in query_lower:
            search_path = "models"
        
        # Execute search
        result = self.file_tools.find_files(pattern, search_path)
        
        if result.get('error'):
            # Log the error and try alternative paths
            logger.warning(f"Search failed in {search_path}: {result.get('error')}")
            
            # Try alternative paths
            alt_paths = [
                "de-dbt-analytics/marketing/models/main/tables",
                "marketing/models/main/tables", 
                "marketing/models",
                "models/marketing",
                "models"
            ]
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
            # Use the correct marketing path
            folder = "de-dbt-analytics/marketing/models/main/tables"
        elif "staging" in query_lower:
            folder = "models/staging"
        
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
            # Use the correct marketing path
            return "de-dbt-analytics/marketing/models/main/tables"
        elif "staging" in query_lower:
            return "models/staging"
        elif "models" in query_lower:
            return "models"
        
        return ""
    
    def _extract_file_path_from_query(self, query: str) -> Optional[str]:
        """Extract file path from query"""
        # Look for file paths in the query
        # This is simplified - could use more sophisticated parsing
        words = query.split()
        for word in words:
            if '.sql' in word or '.py' in word:
                return word
        return None