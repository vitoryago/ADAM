"""
DBT Service for ADAM
Provides DBT analysis capabilities through the API
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from ..dbt_analyzer.parser import DBTManifestParser
from ..dbt_analyzer.memory import DBTMemorySystem
from ..dbt_analyzer.lineage import DBTLineageTracker
from ..dbt_analyzer.optimizer import SQLOptimizer, OptimizationSuggestion
from ..dbt_analyzer.documentation import DBTDocGenerator

logger = logging.getLogger(__name__)


class DBTService:
    """
    Service layer for DBT functionality in ADAM
    """

    def __init__(self, memory_path: str = "./data/dbt_memory"):
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(parents=True, exist_ok=True)

        # Cache for parsers by project path
        self.parsers: Dict[str, DBTManifestParser] = {}
        self.memory_systems: Dict[str, DBTMemorySystem] = {}
        self.lineage_trackers: Dict[str, DBTLineageTracker] = {}

        # Shared optimizer
        self.optimizer = SQLOptimizer()

    def _get_or_create_parser(self, project_path: str, project_id: str) -> Optional[DBTManifestParser]:
        """Get or create a parser for a DBT project"""
        cache_key = f"{project_path}:{project_id}"

        if cache_key not in self.parsers:
            parser = DBTManifestParser(project_path)
            if not parser.parse():
                logger.error(f"Failed to parse DBT project at {project_path}")
                return None
            self.parsers[cache_key] = parser

        return self.parsers[cache_key]

    def _get_or_create_memory(self, project_id: str) -> DBTMemorySystem:
        """Get or create a memory system for a project"""
        if project_id not in self.memory_systems:
            self.memory_systems[project_id] = DBTMemorySystem(
                memory_path=str(self.memory_path),
                project_id=project_id
            )
        return self.memory_systems[project_id]

    def _get_or_create_lineage_tracker(self, project_path: str, project_id: str) -> Optional[DBTLineageTracker]:
        """Get or create a lineage tracker for a project"""
        cache_key = f"{project_path}:{project_id}"

        if cache_key not in self.lineage_trackers:
            parser = self._get_or_create_parser(project_path, project_id)
            if not parser:
                return None

            tracker = DBTLineageTracker()
            tracker.build_lineage_graph(parser.models, parser.sources)
            self.lineage_trackers[cache_key] = tracker

        return self.lineage_trackers[cache_key]

    def ingest_project(self, project_path: str, project_id: str) -> Dict[str, Any]:
        """
        Ingest a DBT project into ADAM's memory
        """
        try:
            parser = self._get_or_create_parser(project_path, project_id)
            if not parser:
                return {
                    "success": False,
                    "error": "Failed to parse manifest.json. Run 'dbt compile' or 'dbt parse' first."
                }

            memory = self._get_or_create_memory(project_id)
            stats = memory.ingest_dbt_project(parser)

            return {
                "success": True,
                "project_id": project_id,
                "project_path": project_path,
                "statistics": stats
            }

        except Exception as e:
            logger.error(f"Error ingesting DBT project: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_model_documentation(
        self,
        project_path: str,
        project_id: str,
        model_name: str,
        platform: str = "redshift"
    ) -> Dict[str, Any]:
        """
        Generate documentation for a specific model
        """
        try:
            parser = self._get_or_create_parser(project_path, project_id)
            if not parser:
                return {"success": False, "error": "Failed to parse project"}

            lineage_tracker = self._get_or_create_lineage_tracker(project_path, project_id)
            if not lineage_tracker:
                return {"success": False, "error": "Failed to build lineage"}

            doc_generator = DBTDocGenerator(parser, lineage_tracker, self.optimizer)

            documentation = doc_generator.generate_model_documentation(
                model_name,
                include_lineage=True,
                include_tests=True,
                include_optimizations=True,
                platform=platform
            )

            return {
                "success": True,
                "model_name": model_name,
                "documentation": documentation
            }

        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            return {"success": False, "error": str(e)}

    def analyze_lineage(
        self,
        project_path: str,
        project_id: str,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Analyze lineage for a model
        """
        try:
            parser = self._get_or_create_parser(project_path, project_id)
            if not parser:
                return {"success": False, "error": "Failed to parse project"}

            # Find model ID
            model_id = None
            for mid, model in parser.models.items():
                if model.name == model_name:
                    model_id = mid
                    break

            if not model_id:
                return {"success": False, "error": f"Model '{model_name}' not found"}

            lineage_tracker = self._get_or_create_lineage_tracker(project_path, project_id)
            if not lineage_tracker:
                return {"success": False, "error": "Failed to build lineage"}

            impact = lineage_tracker.calculate_impact_analysis(model_id)

            return {
                "success": True,
                "model_name": model_name,
                "impact_analysis": impact
            }

        except Exception as e:
            logger.error(f"Error analyzing lineage: {e}")
            return {"success": False, "error": str(e)}

    def get_optimizations(
        self,
        project_path: str,
        project_id: str,
        model_name: str,
        platform: str = "redshift"
    ) -> Dict[str, Any]:
        """
        Get optimization suggestions for a model
        """
        try:
            parser = self._get_or_create_parser(project_path, project_id)
            if not parser:
                return {"success": False, "error": "Failed to parse project"}

            # Get model SQL
            sql = parser.get_model_sql(model_name, compiled=True)
            if not sql:
                return {"success": False, "error": f"Model '{model_name}' not found"}

            # Analyze SQL
            suggestions = self.optimizer.analyze_sql(sql, platform)

            return {
                "success": True,
                "model_name": model_name,
                "platform": platform,
                "suggestions": [
                    {
                        "severity": s.severity,
                        "category": s.category,
                        "title": s.title,
                        "description": s.description,
                        "current_pattern": s.current_pattern,
                        "suggested_pattern": s.suggested_pattern,
                        "estimated_impact": s.estimated_impact
                    }
                    for s in suggestions
                ]
            }

        except Exception as e:
            logger.error(f"Error getting optimizations: {e}")
            return {"success": False, "error": str(e)}

    def search_models(
        self,
        project_id: str,
        query: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Search for models in memory
        """
        try:
            memory = self._get_or_create_memory(project_id)
            results = memory.search_models(query, limit)

            return {
                "success": True,
                "query": query,
                "results": results
            }

        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return {"success": False, "error": str(e)}

    def get_project_stats(
        self,
        project_path: str,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Get statistics for a DBT project
        """
        try:
            parser = self._get_or_create_parser(project_path, project_id)
            if not parser:
                return {"success": False, "error": "Failed to parse project"}

            # Get test coverage
            coverage = parser.get_test_coverage()

            # Count materializations
            materializations = {}
            for model in parser.models.values():
                mat = model.materialization
                materializations[mat] = materializations.get(mat, 0) + 1

            return {
                "success": True,
                "project_id": project_id,
                "statistics": {
                    "total_models": len(parser.models),
                    "total_sources": len(parser.sources),
                    "total_tests": len(parser.tests),
                    "total_exposures": len(parser.exposures),
                    "test_coverage": coverage,
                    "materializations": materializations
                }
            }

        except Exception as e:
            logger.error(f"Error getting project stats: {e}")
            return {"success": False, "error": str(e)}

    def list_models(
        self,
        project_path: str,
        project_id: str,
        tag: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List all models in a project, optionally filtered by tag
        """
        try:
            parser = self._get_or_create_parser(project_path, project_id)
            if not parser:
                return {"success": False, "error": "Failed to parse project"}

            if tag:
                models = parser.get_models_by_tag(tag)
            else:
                models = list(parser.models.values())

            return {
                "success": True,
                "total_models": len(models),
                "models": [
                    {
                        "name": m.name,
                        "description": m.description,
                        "materialization": m.materialization,
                        "database": m.database,
                        "schema": m.schema,
                        "tags": m.tags
                    }
                    for m in models
                ]
            }

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {"success": False, "error": str(e)}

    def find_dbt_projects(self, workspace_path: str) -> List[Dict[str, str]]:
        """
        Find all DBT projects in a workspace
        """
        try:
            projects = []
            workspace = Path(workspace_path)

            # Search for dbt_project.yml files
            for dbt_project_file in workspace.rglob("dbt_project.yml"):
                project_dir = dbt_project_file.parent

                # Check if it has a target/manifest.json
                manifest_path = project_dir / "target" / "manifest.json"
                has_manifest = manifest_path.exists()

                projects.append({
                    "path": str(project_dir),
                    "name": project_dir.name,
                    "has_manifest": has_manifest,
                    "manifest_path": str(manifest_path) if has_manifest else None
                })

            return projects

        except Exception as e:
            logger.error(f"Error finding DBT projects: {e}")
            return []

    def clear_cache(self, project_id: Optional[str] = None):
        """
        Clear cached parsers and trackers
        """
        if project_id:
            # Clear specific project
            keys_to_remove = [k for k in self.parsers.keys() if k.endswith(f":{project_id}")]
            for key in keys_to_remove:
                self.parsers.pop(key, None)
                self.lineage_trackers.pop(key, None)
            self.memory_systems.pop(project_id, None)
        else:
            # Clear all
            self.parsers.clear()
            self.lineage_trackers.clear()
            self.memory_systems.clear()


# Global service instance
_dbt_service = None

def get_dbt_service() -> DBTService:
    """Get or create the global DBT service instance"""
    global _dbt_service
    if _dbt_service is None:
        _dbt_service = DBTService()
    return _dbt_service