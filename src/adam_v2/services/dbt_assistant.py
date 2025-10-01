"""
DBT Assistant for ADAM
Natural language interface to DBT projects - works without manifest.json!
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import re

from ..dbt_analyzer.direct_reader import DirectDBTReader
from ..dbt_analyzer.optimizer import SQLOptimizer

logger = logging.getLogger(__name__)


class DBTAssistant:
    """
    Conversational interface to DBT projects
    Understands natural language queries about DBT models
    """

    def __init__(self):
        self.readers: Dict[str, DirectDBTReader] = {}
        self.optimizer = SQLOptimizer()
        self.current_project: Optional[str] = None

    def detect_dbt_projects(self, workspace_path: str) -> List[Dict[str, Any]]:
        """Find all DBT projects in workspace"""
        try:
            projects = []
            workspace = Path(workspace_path)

            for dbt_project_file in workspace.rglob("dbt_project.yml"):
                project_dir = dbt_project_file.parent

                # Quick check if it has models
                models_dir = project_dir / "models"
                has_models = models_dir.exists()

                if has_models:
                    model_count = len(list(models_dir.rglob("*.sql")))
                else:
                    model_count = 0

                projects.append({
                    "path": str(project_dir),
                    "name": project_dir.name,
                    "has_models": has_models,
                    "model_count": model_count,
                    "relative_path": str(project_dir.relative_to(workspace))
                })

            return projects

        except Exception as e:
            logger.error(f"Error detecting DBT projects: {e}")
            return []

    def load_project(self, project_path: str, force_reload: bool = False) -> Dict[str, Any]:
        """Load a DBT project"""
        try:
            if project_path in self.readers and not force_reload:
                return {
                    "success": True,
                    "message": "Project already loaded",
                    "cached": True
                }

            reader = DirectDBTReader(project_path)
            if not reader.read_project():
                return {
                    "success": False,
                    "error": "Failed to read DBT project. Check if dbt_project.yml exists."
                }

            self.readers[project_path] = reader
            self.current_project = project_path

            stats = reader.get_project_stats()

            return {
                "success": True,
                "project_name": stats["project_name"],
                "project_path": project_path,
                "statistics": stats,
                "cached": False
            }

        except Exception as e:
            logger.error(f"Error loading project: {e}")
            return {"success": False, "error": str(e)}

    def get_current_reader(self) -> Optional[DirectDBTReader]:
        """Get reader for current project"""
        if not self.current_project or self.current_project not in self.readers:
            return None
        return self.readers[self.current_project]

    def handle_query(self, query: str, project_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle natural language queries about DBT models
        """
        # Use specified project or current
        if project_path:
            if project_path not in self.readers:
                load_result = self.load_project(project_path)
                if not load_result["success"]:
                    return load_result
            reader = self.readers[project_path]
        else:
            reader = self.get_current_reader()
            if not reader:
                return {
                    "success": False,
                    "error": "No DBT project loaded. Load a project first."
                }

        query_lower = query.lower()

        # Pattern matching for different query types
        if "tell me about" in query_lower or "what is" in query_lower or "explain" in query_lower:
            # Model description query
            model_name = self._extract_model_name(query, reader)
            if model_name:
                return self._describe_model(reader, model_name)

        elif "depends on" in query_lower or "upstream" in query_lower or "uses" in query_lower:
            # Dependency query
            model_name = self._extract_model_name(query, reader)
            if model_name:
                return self._get_dependencies(reader, model_name)

        elif "used by" in query_lower or "downstream" in query_lower or "impacts" in query_lower:
            # Downstream query
            model_name = self._extract_model_name(query, reader)
            if model_name:
                return self._get_downstream(reader, model_name)

        elif "complex" in query_lower:
            # Complexity query
            return self._get_complex_models(reader)

        elif "optimize" in query_lower or "improve" in query_lower:
            # Optimization query
            model_name = self._extract_model_name(query, reader)
            if model_name:
                platform = "snowflake" if "snowflake" in query_lower else "redshift"
                return self._get_optimizations(reader, model_name, platform)

        elif "search" in query_lower or "find" in query_lower:
            # Search query
            search_term = self._extract_search_term(query)
            if search_term:
                return self._search_models(reader, search_term)

        elif "how many" in query_lower or "count" in query_lower or "stats" in query_lower:
            # Statistics query
            return self._get_stats(reader)

        elif "list" in query_lower or "show all" in query_lower:
            # List query
            if "test" in query_lower:
                return self._list_models_with_tests(reader)
            else:
                return self._list_models(reader)

        # Default: try to extract model name and describe
        model_name = self._extract_model_name(query, reader)
        if model_name:
            return self._describe_model(reader, model_name)

        return {
            "success": False,
            "error": "I didn't understand your query. Try asking about a specific model, or use commands like 'tell me about', 'depends on', 'used by', 'optimize', etc."
        }

    def _extract_model_name(self, query: str, reader: DirectDBTReader) -> Optional[str]:
        """Extract model name from query"""
        # Try to find exact matches in the query
        query_lower = query.lower()

        for model_name in reader.models.keys():
            if model_name.lower() in query_lower:
                return model_name

        # Try to extract quoted strings
        matches = re.findall(r'["\']([^"\']+)["\']', query)
        if matches:
            potential_name = matches[0]
            if potential_name in reader.models:
                return potential_name

        return None

    def _extract_search_term(self, query: str) -> Optional[str]:
        """Extract search term from query"""
        # Look for "search for X" or "find X"
        patterns = [
            r'search\s+for\s+["\']?([^"\']+)["\']?',
            r'find\s+["\']?([^"\']+)["\']?',
            r'models\s+with\s+["\']?([^"\']+)["\']?',
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _describe_model(self, reader: DirectDBTReader, model_name: str) -> Dict[str, Any]:
        """Get detailed description of a model"""
        model = reader.get_model(model_name)
        if not model:
            return {"success": False, "error": f"Model '{model_name}' not found"}

        upstream = reader.get_upstream_models(model_name)
        downstream = reader.get_downstream_models(model_name)

        return {
            "success": True,
            "type": "model_description",
            "model": {
                "name": model.name,
                "path": model.path,
                "description": model.description or "No description available",
                "complexity_score": model.complexity_score,
                "materialization": model.materialization or "view",
                "upstream_models": upstream,
                "downstream_models": downstream,
                "num_ctes": len(model.ctes),
                "uses_window_functions": model.uses_window_functions,
                "uses_joins": model.uses_joins,
                "uses_aggregations": model.uses_aggregations,
                "tests": model.tests,
                "columns": model.columns,
                "sources": [f"{s[0]}.{s[1]}" for s in model.sources]
            }
        }

    def _get_dependencies(self, reader: DirectDBTReader, model_name: str) -> Dict[str, Any]:
        """Get model dependencies"""
        model = reader.get_model(model_name)
        if not model:
            return {"success": False, "error": f"Model '{model_name}' not found"}

        upstream = reader.get_upstream_models(model_name)

        dependencies = []
        for dep_name in upstream:
            dep_model = reader.get_model(dep_name)
            if dep_model:
                dependencies.append({
                    "name": dep_name,
                    "complexity": dep_model.complexity_score,
                    "materialization": dep_model.materialization or "view"
                })

        return {
            "success": True,
            "type": "dependencies",
            "model": model_name,
            "upstream_models": dependencies,
            "sources": [f"{s[0]}.{s[1]}" for s in model.sources]
        }

    def _get_downstream(self, reader: DirectDBTReader, model_name: str) -> Dict[str, Any]:
        """Get downstream models"""
        downstream = reader.get_downstream_models(model_name)

        downstream_models = []
        for dep_name in downstream:
            dep_model = reader.get_model(dep_name)
            if dep_model:
                downstream_models.append({
                    "name": dep_name,
                    "complexity": dep_model.complexity_score,
                    "materialization": dep_model.materialization or "view"
                })

        return {
            "success": True,
            "type": "downstream",
            "model": model_name,
            "downstream_models": downstream_models,
            "count": len(downstream_models)
        }

    def _get_complex_models(self, reader: DirectDBTReader, threshold: int = 15) -> Dict[str, Any]:
        """Get list of complex models"""
        complex_models = [
            {
                "name": model.name,
                "complexity": model.complexity_score,
                "num_ctes": len(model.ctes),
                "uses_window_functions": model.uses_window_functions,
                "materialization": model.materialization or "view"
            }
            for model in reader.models.values()
            if model.complexity_score >= threshold
        ]

        # Sort by complexity
        complex_models.sort(key=lambda x: x["complexity"], reverse=True)

        return {
            "success": True,
            "type": "complex_models",
            "threshold": threshold,
            "models": complex_models[:20],  # Top 20
            "total_count": len(complex_models)
        }

    def _get_optimizations(self, reader: DirectDBTReader, model_name: str, platform: str) -> Dict[str, Any]:
        """Get optimization suggestions"""
        model = reader.get_model(model_name)
        if not model:
            return {"success": False, "error": f"Model '{model_name}' not found"}

        suggestions = self.optimizer.analyze_sql(model.sql, platform)

        return {
            "success": True,
            "type": "optimizations",
            "model": model_name,
            "platform": platform,
            "suggestions": [
                {
                    "severity": s.severity,
                    "category": s.category,
                    "title": s.title,
                    "description": s.description,
                    "estimated_impact": s.estimated_impact
                }
                for s in suggestions
            ]
        }

    def _search_models(self, reader: DirectDBTReader, search_term: str) -> Dict[str, Any]:
        """Search for models"""
        results = []
        search_lower = search_term.lower()

        for name, model in reader.models.items():
            # Search in name
            if search_lower in name.lower():
                results.append({
                    "name": name,
                    "match_type": "name",
                    "description": model.description or "No description"
                })
                continue

            # Search in description
            if model.description and search_lower in model.description.lower():
                results.append({
                    "name": name,
                    "match_type": "description",
                    "description": model.description[:100]
                })
                continue

            # Search in SQL
            if search_lower in model.sql.lower():
                results.append({
                    "name": name,
                    "match_type": "sql",
                    "description": model.description or "Found in SQL"
                })

        return {
            "success": True,
            "type": "search_results",
            "query": search_term,
            "results": results[:20],  # Top 20
            "total_count": len(results)
        }

    def _get_stats(self, reader: DirectDBTReader) -> Dict[str, Any]:
        """Get project statistics"""
        stats = reader.get_project_stats()
        return {
            "success": True,
            "type": "statistics",
            "stats": stats
        }

    def _list_models(self, reader: DirectDBTReader, limit: int = 50) -> Dict[str, Any]:
        """List models"""
        models = [
            {
                "name": model.name,
                "complexity": model.complexity_score,
                "materialization": model.materialization or "view",
                "description": model.description or "No description"
            }
            for model in sorted(reader.models.values(), key=lambda m: m.complexity_score, reverse=True)
        ]

        return {
            "success": True,
            "type": "model_list",
            "models": models[:limit],
            "total_count": len(models)
        }

    def _list_models_with_tests(self, reader: DirectDBTReader) -> Dict[str, Any]:
        """List models that have tests"""
        models = [
            {
                "name": model.name,
                "test_count": len(model.tests),
                "tests": model.tests
            }
            for model in reader.models.values()
            if model.tests
        ]

        return {
            "success": True,
            "type": "models_with_tests",
            "models": models,
            "total_count": len(models)
        }


# Global assistant instance
_dbt_assistant = None

def get_dbt_assistant() -> DBTAssistant:
    """Get or create the global DBT assistant instance"""
    global _dbt_assistant
    if _dbt_assistant is None:
        _dbt_assistant = DBTAssistant()
    return _dbt_assistant
