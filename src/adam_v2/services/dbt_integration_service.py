"""
DBT Integration Service for ADAM Chat
Automatically detects DBT-related questions and provides contextual answers
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from .dbt_assistant import get_dbt_assistant

logger = logging.getLogger(__name__)


class DBTIntegrationService:
    """
    Integrates DBT knowledge into ADAM's conversational flow
    """

    def __init__(self):
        self.assistant = get_dbt_assistant()
        self._dbt_keywords = [
            "dbt", "model", "models", "lineage", "upstream", "downstream",
            "materialization", "incremental", "table", "view",
            "ref(", "source(", "schema.yml", "dbt_project.yml",
            "compile", "test", "run", "snapshot"
        ]

    def is_dbt_question(self, message: str, workspace_path: Optional[str] = None) -> bool:
        """
        Detect if the user is asking about DBT
        """
        message_lower = message.lower()

        # Check for explicit DBT keywords
        if any(keyword in message_lower for keyword in self._dbt_keywords):
            return True

        # Check if workspace has DBT projects
        if workspace_path:
            try:
                projects = self.assistant.detect_dbt_projects(workspace_path)
                if projects:
                    # Check if message mentions any model names from the projects
                    for project in projects[:2]:  # Only check first 2 projects for performance
                        project_path = os.path.join(workspace_path, project['relative_path'])
                        if project_path in self.assistant.readers:
                            reader = self.assistant.readers[project_path]
                            # Check if any model name is mentioned
                            for model_name in list(reader.models.keys())[:100]:  # Check first 100 models
                                if model_name.lower() in message_lower:
                                    return True
            except Exception as e:
                logger.warning(f"Error checking for DBT models: {e}")

        return False

    def ensure_project_loaded(self, workspace_path: str) -> Optional[str]:
        """
        Ensure a DBT project is loaded, return project path if successful
        """
        try:
            # Check if current project is loaded
            if self.assistant.current_project:
                return self.assistant.current_project

            # Try to detect and load first DBT project
            projects = self.assistant.detect_dbt_projects(workspace_path)
            if not projects:
                return None

            # Load the first project with models
            for project in projects:
                if project['model_count'] > 0:
                    project_path = os.path.join(workspace_path, project['relative_path'])
                    result = self.assistant.load_project(project_path)
                    if result['success']:
                        logger.info(f"Auto-loaded DBT project: {project['name']}")
                        return project_path

            return None

        except Exception as e:
            logger.error(f"Error loading DBT project: {e}")
            return None

    def get_dbt_context(self, message: str, workspace_path: Optional[str] = None) -> str:
        """
        Get DBT context to add to the conversation
        Returns formatted context string to add to the message
        """
        if not workspace_path:
            return ""

        try:
            # Ensure a project is loaded
            project_path = self.ensure_project_loaded(workspace_path)
            if not project_path:
                return ""

            # Try to handle the query
            result = self.assistant.handle_query(message, project_path)

            if not result['success']:
                return ""

            # Format the result as context
            return self._format_dbt_result(result)

        except Exception as e:
            logger.error(f"Error getting DBT context: {e}")
            return ""

    def _format_dbt_result(self, result: Dict[str, Any]) -> str:
        """
        Format DBT assistant result as context for LLM
        """
        query_type = result.get('type', 'unknown')

        if query_type == 'model_description':
            model = result['model']
            context = f"\n\n=== DBT Model: {model['name']} ===\n"
            context += f"Description: {model['description']}\n"
            context += f"Complexity: {model['complexity_score']}\n"
            context += f"Materialization: {model['materialization']}\n"

            if model['upstream_models']:
                context += f"Depends on: {', '.join(model['upstream_models'][:5])}\n"

            if model['downstream_models']:
                context += f"Used by: {', '.join(model['downstream_models'][:5])}\n"

            if model['tests']:
                context += f"Tests: {len(model['tests'])} tests defined\n"

            if model['uses_window_functions']:
                context += "Uses window functions\n"

            return context

        elif query_type == 'dependencies':
            model_name = result['model']
            upstream = result['upstream_models']

            context = f"\n\n=== Dependencies for {model_name} ===\n"
            if upstream:
                context += f"Upstream models ({len(upstream)}):\n"
                for dep in upstream[:10]:
                    context += f"  - {dep['name']} (complexity: {dep['complexity']})\n"

            if result['sources']:
                context += f"Sources: {', '.join(result['sources'])}\n"

            return context

        elif query_type == 'downstream':
            model_name = result['model']
            downstream = result['downstream_models']

            context = f"\n\n=== Models using {model_name} ===\n"
            if downstream:
                context += f"Downstream models ({len(downstream)}):\n"
                for dep in downstream[:10]:
                    context += f"  - {dep['name']} (complexity: {dep['complexity']})\n"
            else:
                context += "No downstream models found\n"

            return context

        elif query_type == 'complex_models':
            models = result['models']
            total = result['total_count']

            context = f"\n\n=== Complex DBT Models (threshold: {result['threshold']}) ===\n"
            context += f"Found {total} complex models\n"
            context += "Top models:\n"
            for model in models[:10]:
                context += f"  - {model['name']} (complexity: {model['complexity']})\n"

            return context

        elif query_type == 'search_results':
            query = result['query']
            results = result['results']
            total = result['total_count']

            context = f"\n\n=== DBT Models matching '{query}' ===\n"
            context += f"Found {total} results\n"
            for res in results[:10]:
                context += f"  - {res['name']} ({res['match_type']})\n"

            return context

        elif query_type == 'statistics':
            stats = result['stats']

            context = f"\n\n=== DBT Project Statistics ===\n"
            context += f"Project: {stats['project_name']}\n"
            context += f"Total models: {stats['total_models']}\n"
            context += f"Total sources: {stats['total_sources']}\n"
            context += f"Complexity:\n"
            context += f"  Simple: {stats['complexity']['simple']}\n"
            context += f"  Medium: {stats['complexity']['medium']}\n"
            context += f"  Complex: {stats['complexity']['complex']}\n"

            return context

        elif query_type == 'optimizations':
            model = result['model']
            suggestions = result['suggestions']

            context = f"\n\n=== Optimizations for {model} ({result['platform']}) ===\n"
            for i, sug in enumerate(suggestions, 1):
                context += f"{i}. [{sug['severity']}] {sug['title']}\n"
                context += f"   {sug['description']}\n"

            return context

        return ""

    def get_available_projects_info(self, workspace_path: str) -> str:
        """
        Get information about available DBT projects
        """
        try:
            projects = self.assistant.detect_dbt_projects(workspace_path)
            if not projects:
                return ""

            info = "\n\n=== Available DBT Projects ===\n"
            for proj in projects[:5]:  # Limit to 5 projects
                info += f"  - {proj['name']}: {proj['model_count']} models at {proj['relative_path']}\n"

            return info

        except Exception as e:
            logger.error(f"Error getting projects info: {e}")
            return ""


# Global service instance
_dbt_integration_service = None

def get_dbt_integration_service() -> DBTIntegrationService:
    """Get or create the global DBT integration service instance"""
    global _dbt_integration_service
    if _dbt_integration_service is None:
        _dbt_integration_service = DBTIntegrationService()
    return _dbt_integration_service
