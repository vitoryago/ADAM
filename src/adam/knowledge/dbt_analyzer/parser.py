"""
DBT Manifest Parser
Parses and understands DBT project structure from manifest.json
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class DBTModel:
    """Represents a DBT model with all its metadata"""
    unique_id: str
    name: str
    description: str = ""
    raw_sql: str = ""
    compiled_sql: str = ""
    depends_on: List[str] = field(default_factory=list)
    columns: Dict[str, Dict] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    database: Optional[str] = None
    schema: Optional[str] = None
    alias: Optional[str] = None
    materialization: str = "view"
    tests: List[Dict] = field(default_factory=list)
    metrics: List[Dict] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        """Get fully qualified name"""
        parts = []
        if self.database:
            parts.append(self.database)
        if self.schema:
            parts.append(self.schema)
        parts.append(self.alias or self.name)
        return ".".join(parts)

    @property
    def is_incremental(self) -> bool:
        """Check if model is incremental"""
        return self.materialization == "incremental"

    @property
    def is_ephemeral(self) -> bool:
        """Check if model is ephemeral"""
        return self.materialization == "ephemeral"


@dataclass
class DBTSource:
    """Represents a DBT source"""
    unique_id: str
    name: str
    source_name: str
    description: str = ""
    database: Optional[str] = None
    schema: Optional[str] = None
    tables: List[Dict] = field(default_factory=list)
    freshness: Optional[Dict] = None


@dataclass
class DBTTest:
    """Represents a DBT test"""
    unique_id: str
    name: str
    test_type: str  # generic, singular
    raw_sql: str = ""
    compiled_sql: str = ""
    depends_on: List[str] = field(default_factory=list)
    column_name: Optional[str] = None
    severity: str = "error"


@dataclass
class DBTExposure:
    """Represents a DBT exposure"""
    unique_id: str
    name: str
    type: str  # dashboard, notebook, analysis, ml, application
    description: str = ""
    depends_on: List[str] = field(default_factory=list)
    owner: Dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None


class DBTManifestParser:
    """
    Parses DBT manifest.json to extract complete project structure
    """

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.manifest_path = self.project_path / "target" / "manifest.json"
        self.models: Dict[str, DBTModel] = {}
        self.sources: Dict[str, DBTSource] = {}
        self.tests: Dict[str, DBTTest] = {}
        self.exposures: Dict[str, DBTExposure] = {}
        self.macros: Dict[str, Dict] = {}
        self.project_metadata: Dict[str, Any] = {}

    def parse(self) -> bool:
        """
        Parse the DBT manifest file
        Returns True if successful, False otherwise
        """
        if not self.manifest_path.exists():
            logger.error(f"Manifest not found at {self.manifest_path}")
            logger.info("Run 'dbt compile' or 'dbt run' to generate manifest.json")
            return False

        try:
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)

            # Parse metadata
            self.project_metadata = manifest.get("metadata", {})

            # Parse models
            for model_id, model_data in manifest.get("nodes", {}).items():
                if model_data["resource_type"] == "model":
                    self._parse_model(model_id, model_data)
                elif model_data["resource_type"] == "test":
                    self._parse_test(model_id, model_data)
                elif model_data["resource_type"] == "source":
                    self._parse_source(model_id, model_data)

            # Parse sources (in separate section)
            for source_id, source_data in manifest.get("sources", {}).items():
                self._parse_source(source_id, source_data)

            # Parse exposures
            for exposure_id, exposure_data in manifest.get("exposures", {}).items():
                self._parse_exposure(exposure_id, exposure_data)

            # Parse macros
            for macro_id, macro_data in manifest.get("macros", {}).items():
                self.macros[macro_id] = macro_data

            logger.info(f"Parsed {len(self.models)} models, {len(self.sources)} sources, "
                       f"{len(self.tests)} tests, {len(self.exposures)} exposures")
            return True

        except Exception as e:
            logger.error(f"Error parsing manifest: {e}")
            return False

    def _parse_model(self, model_id: str, data: Dict):
        """Parse a single model from manifest"""
        model = DBTModel(
            unique_id=model_id,
            name=data["name"],
            description=data.get("description", ""),
            raw_sql=data.get("raw_sql", ""),
            compiled_sql=data.get("compiled_sql", ""),
            depends_on=[dep for dep in data.get("depends_on", {}).get("nodes", [])],
            columns=data.get("columns", {}),
            config=data.get("config", {}),
            tags=data.get("tags", []),
            database=data.get("database"),
            schema=data.get("schema"),
            alias=data.get("alias"),
            materialization=data.get("config", {}).get("materialized", "view")
        )

        # Extract tests for this model
        if "tests" in data:
            model.tests = data["tests"]

        # Extract metrics if defined
        if "metrics" in data:
            model.metrics = data["metrics"]

        self.models[model_id] = model

    def _parse_source(self, source_id: str, data: Dict):
        """Parse a source from manifest"""
        source = DBTSource(
            unique_id=source_id,
            name=data.get("name", ""),
            source_name=data.get("source_name", ""),
            description=data.get("description", ""),
            database=data.get("database"),
            schema=data.get("schema"),
            tables=data.get("tables", []),
            freshness=data.get("freshness")
        )
        self.sources[source_id] = source

    def _parse_test(self, test_id: str, data: Dict):
        """Parse a test from manifest"""
        test = DBTTest(
            unique_id=test_id,
            name=data["name"],
            test_type="generic" if "test_metadata" in data else "singular",
            raw_sql=data.get("raw_sql", ""),
            compiled_sql=data.get("compiled_sql", ""),
            depends_on=data.get("depends_on", {}).get("nodes", []),
            column_name=data.get("column_name"),
            severity=data.get("config", {}).get("severity", "error")
        )
        self.tests[test_id] = test

    def _parse_exposure(self, exposure_id: str, data: Dict):
        """Parse an exposure from manifest"""
        exposure = DBTExposure(
            unique_id=exposure_id,
            name=data["name"],
            type=data.get("type", "dashboard"),
            description=data.get("description", ""),
            depends_on=data.get("depends_on", {}).get("nodes", []),
            owner=data.get("owner", {}),
            url=data.get("url")
        )
        self.exposures[exposure_id] = exposure

    def get_model_lineage(self, model_name: str) -> Dict[str, List[str]]:
        """
        Get upstream and downstream dependencies for a model
        """
        model = self._find_model_by_name(model_name)
        if not model:
            return {"upstream": [], "downstream": []}

        upstream = model.depends_on
        downstream = []

        # Find models that depend on this one
        for other_id, other_model in self.models.items():
            if model.unique_id in other_model.depends_on:
                downstream.append(other_id)

        return {
            "upstream": upstream,
            "downstream": downstream
        }

    def _find_model_by_name(self, name: str) -> Optional[DBTModel]:
        """Find a model by its name"""
        for model in self.models.values():
            if model.name == name:
                return model
        return None

    def get_model_sql(self, model_name: str, compiled: bool = True) -> Optional[str]:
        """Get SQL for a specific model"""
        model = self._find_model_by_name(model_name)
        if model:
            return model.compiled_sql if compiled else model.raw_sql
        return None

    def get_all_incremental_models(self) -> List[DBTModel]:
        """Get all incremental models in the project"""
        return [m for m in self.models.values() if m.is_incremental]

    def get_models_by_tag(self, tag: str) -> List[DBTModel]:
        """Get all models with a specific tag"""
        return [m for m in self.models.values() if tag in m.tags]

    def get_test_coverage(self) -> Dict[str, Dict]:
        """
        Get test coverage statistics for the project
        """
        models_with_tests = set()
        columns_with_tests = set()

        for test in self.tests.values():
            for dep in test.depends_on:
                if dep.startswith("model."):
                    models_with_tests.add(dep)
                    if test.column_name:
                        columns_with_tests.add(f"{dep}.{test.column_name}")

        total_models = len(self.models)
        tested_models = len(models_with_tests)

        return {
            "total_models": total_models,
            "tested_models": tested_models,
            "coverage_percentage": (tested_models / total_models * 100) if total_models > 0 else 0,
            "total_tests": len(self.tests),
            "columns_tested": len(columns_with_tests)
        }

    def export_to_json(self) -> Dict:
        """Export parsed data to JSON format"""
        return {
            "metadata": self.project_metadata,
            "models": {k: v.__dict__ for k, v in self.models.items()},
            "sources": {k: v.__dict__ for k, v in self.sources.items()},
            "tests": {k: v.__dict__ for k, v in self.tests.items()},
            "exposures": {k: v.__dict__ for k, v in self.exposures.items()},
            "statistics": {
                "model_count": len(self.models),
                "source_count": len(self.sources),
                "test_count": len(self.tests),
                "exposure_count": len(self.exposures),
                "test_coverage": self.get_test_coverage()
            }
        }