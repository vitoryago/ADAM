"""
Direct DBT Project Reader - No manifest.json needed!
Reads raw .sql files, schema.yml, and dbt_project.yml directly.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class DBTModel:
    """Represents a DBT model from raw SQL file"""
    name: str
    path: str
    sql: str
    project_name: str

    # Extracted from SQL
    refs: List[str] = field(default_factory=list)  # {{ ref('model_name') }}
    sources: List[Tuple[str, str]] = field(default_factory=list)  # {{ source('schema', 'table') }}
    ctes: List[str] = field(default_factory=list)
    materialization: Optional[str] = None

    # From schema.yml
    description: Optional[str] = None
    columns: Dict[str, str] = field(default_factory=dict)  # column_name -> description
    tests: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # SQL patterns
    uses_window_functions: bool = False
    uses_joins: bool = False
    uses_aggregations: bool = False
    complexity_score: int = 0


@dataclass
class DBTSource:
    """Represents a DBT source from schema.yml"""
    name: str
    schema: str
    database: Optional[str]
    tables: List[str] = field(default_factory=list)
    description: Optional[str] = None


class DirectDBTReader:
    """
    Reads DBT projects directly from source files without needing manifest.json.
    Analyzes SQL files, schema.yml, and dbt_project.yml.
    """

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.models: Dict[str, DBTModel] = {}
        self.sources: Dict[str, DBTSource] = {}
        self.project_config: Dict = {}
        self.project_name: str = ""

    def read_project(self) -> bool:
        """Main entry point - reads entire DBT project"""
        try:
            # 1. Read dbt_project.yml to get project name and config
            if not self._read_project_config():
                logger.error("Failed to read dbt_project.yml")
                return False

            # 2. Find and read all schema.yml files for metadata
            schema_metadata = self._read_all_schema_files()

            # 3. Find and parse all .sql model files
            self._read_all_sql_files(schema_metadata)

            logger.info(f"✅ Read {len(self.models)} models and {len(self.sources)} sources")
            return True

        except Exception as e:
            logger.error(f"Error reading project: {e}")
            return False

    def _read_project_config(self) -> bool:
        """Read dbt_project.yml"""
        config_file = self.project_path / "dbt_project.yml"
        if not config_file.exists():
            logger.error(f"dbt_project.yml not found at {config_file}")
            return False

        with open(config_file) as f:
            self.project_config = yaml.safe_load(f)

        self.project_name = self.project_config.get("name", "unknown")
        logger.info(f"📦 Project: {self.project_name}")
        return True

    def _read_all_schema_files(self) -> Dict:
        """Find and read all schema.yml files for metadata"""
        metadata = {
            "models": {},
            "sources": {}
        }

        # Find all YAML files in models/ directory
        models_dir = self.project_path / "models"
        if not models_dir.exists():
            logger.warning(f"models/ directory not found at {models_dir}")
            return metadata

        for yaml_file in models_dir.rglob("*.yml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)

                # Extract model metadata
                if "models" in data:
                    for model in data["models"]:
                        model_name = model.get("name")
                        metadata["models"][model_name] = {
                            "description": model.get("description"),
                            "columns": {col.get("name"): col.get("description", "")
                                      for col in model.get("columns", [])},
                            "tests": self._extract_tests(model),
                            "tags": model.get("tags", [])
                        }

                # Extract source metadata
                if "sources" in data:
                    for source in data["sources"]:
                        source_name = source.get("name")
                        for table in source.get("tables", []):
                            table_name = table.get("name")
                            key = f"{source_name}.{table_name}"

                            self.sources[key] = DBTSource(
                                name=table_name,
                                schema=source.get("schema", source_name),
                                database=source.get("database"),
                                tables=[],
                                description=table.get("description")
                            )

            except Exception as e:
                logger.warning(f"Error reading {yaml_file}: {e}")

        logger.info(f"📄 Found metadata for {len(metadata['models'])} models, {len(self.sources)} sources")
        return metadata

    def _extract_tests(self, model_data: Dict) -> List[str]:
        """Extract test names from model YAML"""
        tests = []

        # Model-level tests
        if "tests" in model_data:
            tests.extend([str(t) for t in model_data["tests"]])

        # Column-level tests
        for col in model_data.get("columns", []):
            if "tests" in col:
                col_name = col.get("name")
                tests.extend([f"{col_name}: {t}" for t in col["tests"]])

        return tests

    def _read_all_sql_files(self, schema_metadata: Dict):
        """Find and parse all .sql model files"""
        models_dir = self.project_path / "models"
        if not models_dir.exists():
            return

        for sql_file in models_dir.rglob("*.sql"):
            try:
                model_name = sql_file.stem
                relative_path = str(sql_file.relative_to(self.project_path))

                with open(sql_file) as f:
                    sql = f.read()

                # Parse SQL to extract dependencies and patterns
                model = DBTModel(
                    name=model_name,
                    path=relative_path,
                    sql=sql,
                    project_name=self.project_name
                )

                # Extract metadata
                self._parse_sql_dependencies(model)
                self._parse_sql_patterns(model)
                self._extract_materialization(model)

                # Add schema metadata if available
                if model_name in schema_metadata["models"]:
                    meta = schema_metadata["models"][model_name]
                    model.description = meta.get("description")
                    model.columns = meta.get("columns", {})
                    model.tests = meta.get("tests", [])
                    model.tags = meta.get("tags", [])

                self.models[model_name] = model

            except Exception as e:
                logger.warning(f"Error reading {sql_file}: {e}")

    def _parse_sql_dependencies(self, model: DBTModel):
        """Extract {{ ref() }} and {{ source() }} from SQL"""
        sql = model.sql

        # Find all {{ ref('model_name') }}
        ref_pattern = r"{{\s*ref\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*}}"
        model.refs = re.findall(ref_pattern, sql)

        # Find all {{ source('schema', 'table') }}
        source_pattern = r"{{\s*source\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)\s*}}"
        model.sources = re.findall(source_pattern, sql)

    def _parse_sql_patterns(self, model: DBTModel):
        """Analyze SQL for common patterns and complexity"""
        sql = model.sql.upper()

        # Extract CTEs
        cte_pattern = r'(\w+)\s+AS\s*\('
        model.ctes = re.findall(cte_pattern, model.sql, re.IGNORECASE)

        # Check for window functions
        window_functions = ['ROW_NUMBER', 'RANK', 'DENSE_RANK', 'LAG', 'LEAD',
                          'FIRST_VALUE', 'LAST_VALUE', 'PARTITION BY']
        model.uses_window_functions = any(func in sql for func in window_functions)

        # Check for joins
        model.uses_joins = bool(re.search(r'\b(INNER|LEFT|RIGHT|FULL|CROSS)\s+JOIN\b', sql))

        # Check for aggregations
        agg_functions = ['SUM(', 'COUNT(', 'AVG(', 'MIN(', 'MAX(', 'GROUP BY']
        model.uses_aggregations = any(func in sql for func in agg_functions)

        # Calculate complexity score
        model.complexity_score = (
            len(model.ctes) * 2 +
            len(model.refs) * 1 +
            len(model.sources) * 1 +
            (5 if model.uses_window_functions else 0) +
            (3 if model.uses_joins else 0) +
            (2 if model.uses_aggregations else 0)
        )

    def _extract_materialization(self, model: DBTModel):
        """Extract materialization from config block in SQL"""
        # Look for {{ config(materialized='view') }}
        config_pattern = r"{{\s*config\s*\([^)]*materialized\s*=\s*['\"]([^'\"]+)['\"]"
        match = re.search(config_pattern, model.sql, re.IGNORECASE)
        if match:
            model.materialization = match.group(1)

    def get_model(self, model_name: str) -> Optional[DBTModel]:
        """Get a specific model"""
        return self.models.get(model_name)

    def get_upstream_models(self, model_name: str) -> List[str]:
        """Get all models that this model depends on"""
        model = self.models.get(model_name)
        if not model:
            return []

        return model.refs

    def get_downstream_models(self, model_name: str) -> List[str]:
        """Get all models that depend on this model"""
        downstream = []
        for name, model in self.models.items():
            if model_name in model.refs:
                downstream.append(name)
        return downstream

    def get_source_usage(self, source_schema: str, source_table: str) -> List[str]:
        """Find all models using a specific source"""
        models_using_source = []
        for name, model in self.models.items():
            if (source_schema, source_table) in model.sources:
                models_using_source.append(name)
        return models_using_source

    def get_all_models(self) -> Dict[str, Dict]:
        """Return all models as plain dicts for compatibility with ColumnIntelligenceEngine.

        The returned dict maps model_name -> {'sql': ..., 'columns': ...}
        so that callers expecting dict-style access work correctly.
        """
        result: Dict[str, Dict] = {}
        for name, model in self.models.items():
            result[name] = {
                "sql": model.sql,
                "columns": {
                    col_name: {"description": col_desc}
                    for col_name, col_desc in model.columns.items()
                },
                "description": model.description,
                "path": model.path,
            }
        return result

    def get_project_stats(self) -> Dict:
        """Get overall project statistics"""
        total_models = len(self.models)
        total_sources = len(self.sources)

        # Complexity distribution
        simple = sum(1 for m in self.models.values() if m.complexity_score < 5)
        medium = sum(1 for m in self.models.values() if 5 <= m.complexity_score < 15)
        complex = sum(1 for m in self.models.values() if m.complexity_score >= 15)

        # Materialization distribution
        materializations = {}
        for model in self.models.values():
            mat = model.materialization or "view"
            materializations[mat] = materializations.get(mat, 0) + 1

        return {
            "project_name": self.project_name,
            "total_models": total_models,
            "total_sources": total_sources,
            "complexity": {
                "simple": simple,
                "medium": medium,
                "complex": complex
            },
            "materializations": materializations,
            "total_refs": sum(len(m.refs) for m in self.models.values()),
            "total_source_refs": sum(len(m.sources) for m in self.models.values())
        }
