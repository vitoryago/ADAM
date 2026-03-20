"""
DBT API Router for ADAM 4.0
Provides endpoints for dbt column documentation, analysis, schema generation,
and model generation -- consumed by the VS Code extension (adamClient.ts).
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class DocumentModelRequest(BaseModel):
    """Request body for POST /columns/document-model"""
    project_path: str = Field(..., description="Absolute path to the dbt project root")
    model_name: str = Field(..., description="Name of the dbt model to document")
    use_ai: bool = Field(default=True, description="Use AI to enhance descriptions")
    preserve_existing: bool = Field(default=True, description="Keep existing descriptions")


class AnalyzeColumnsRequest(BaseModel):
    """Request body for POST /columns/analyze"""
    project_path: str = Field(..., description="Absolute path to the dbt project root")
    use_ai: bool = Field(default=True, description="Use AI to enhance analysis")


class CommonColumnsRequest(BaseModel):
    """Request body for POST /columns/common-columns"""
    project_path: str = Field(..., description="Absolute path to the dbt project root")
    min_occurrences: int = Field(default=2, ge=1, description="Minimum model occurrences")


class GenerateSchemaRequest(BaseModel):
    """Request body for POST /columns/generate-schema"""
    project_path: str = Field(..., description="Absolute path to the dbt project root")
    use_ai: bool = Field(default=True, description="Use AI to enhance descriptions")
    include_tests: bool = Field(default=True, description="Include suggested tests")


class GenerateModelRequest(BaseModel):
    """Request body for POST /generate"""
    model_name: str = Field(..., description="Name for the generated model")
    source_table: str = Field(..., description="Source table name")
    layer: str = Field(default="staging", description="Model layer (staging, intermediate, mart_fact, mart_dimension, utility)")
    source_schema: Optional[str] = Field(default=None, description="Source schema name")
    business_logic: Optional[str] = Field(default=None, description="Business logic description")
    grain: Optional[str] = Field(default=None, description="Model grain")
    unique_key: Optional[str] = Field(default=None, description="Unique key column")
    timestamp_column: Optional[str] = Field(default=None, description="Timestamp column for incremental")
    cluster_keys: Optional[List[str]] = Field(default=None, description="Cluster key columns")
    domain: Optional[str] = Field(default=None, description="Business domain")


# ---------------------------------------------------------------------------
# Lazy-loaded service singletons
# ---------------------------------------------------------------------------

_column_doc_service = None
_knowledge_service = None


def _get_column_doc_service():
    """Lazy-load DBTColumnDocumentationService to avoid import-time failures."""
    global _column_doc_service
    if _column_doc_service is None:
        try:
            from adam.knowledge.dbt_analyzer.column_intelligence import (
                DBTColumnDocumentationService,
            )
            _column_doc_service = DBTColumnDocumentationService()
        except Exception as e:
            logger.error(f"Failed to initialise DBTColumnDocumentationService: {e}")
            raise
    return _column_doc_service


def _get_knowledge_service():
    """Lazy-load DBTKnowledgeService."""
    global _knowledge_service
    if _knowledge_service is None:
        from adam.knowledge.dbt_knowledge import get_dbt_knowledge_service
        _knowledge_service = get_dbt_knowledge_service()
    return _knowledge_service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_project_path(project_path: str) -> Optional[str]:
    """Return an error message if the project path looks invalid, else None."""
    import os
    if not project_path:
        return "project_path is required"
    if not os.path.isdir(project_path):
        return f"Project path does not exist or is not a directory: {project_path}"
    dbt_project_file = os.path.join(project_path, "dbt_project.yml")
    if not os.path.isfile(dbt_project_file):
        return f"No dbt_project.yml found in {project_path}"
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/columns/document-model")
async def document_model_columns(request: DocumentModelRequest) -> Dict[str, Any]:
    """Generate column documentation for a specific dbt model.

    Returns a dict with ``columns``, ``total_columns``, etc.  When the
    project path is invalid or the analysis fails the endpoint still returns
    HTTP 200 with an ``error`` key so the extension can display a friendly
    message.
    """
    path_error = _validate_project_path(request.project_path)
    if path_error:
        return {"error": path_error, "columns": {}, "total_columns": 0}

    try:
        service = _get_column_doc_service()
        result = await service.document_model_columns(
            project_path=request.project_path,
            model_name=request.model_name,
            use_ai=request.use_ai,
            preserve_existing=request.preserve_existing,
        )
        return result
    except Exception as e:
        logger.error(f"document_model_columns failed: {e}")
        return {"error": str(e), "columns": {}, "total_columns": 0}


@router.post("/columns/analyze")
async def analyze_columns(request: AnalyzeColumnsRequest) -> Dict[str, Any]:
    """Analyze all columns across a dbt project.

    Returns a documentation coverage report with pattern detection results.
    """
    path_error = _validate_project_path(request.project_path)
    if path_error:
        return {"error": path_error, "summary": {}, "common_columns": [], "patterns_detected": {}, "foreign_keys": []}

    try:
        service = _get_column_doc_service()
        result = await service.analyze_columns(
            project_path=request.project_path,
            use_ai=request.use_ai,
        )
        return result
    except Exception as e:
        logger.error(f"analyze_columns failed: {e}")
        return {"error": str(e), "summary": {}, "common_columns": [], "patterns_detected": {}, "foreign_keys": []}


@router.post("/columns/common-columns")
async def find_common_columns(request: CommonColumnsRequest) -> List[Dict[str, Any]]:
    """Find columns that appear in multiple models.

    Returns a list of common column records, each containing name,
    occurrence count, models list, and a suggested description.
    """
    path_error = _validate_project_path(request.project_path)
    if path_error:
        return []

    try:
        service = _get_column_doc_service()
        engine = service._get_or_create_engine(request.project_path)

        if not engine.column_catalog:
            engine.analyze_project_columns()

        common = engine.find_common_columns(min_occurrences=request.min_occurrences)

        return [
            {
                "name": name,
                "occurrence_count": len(info.appears_in_models),
                "appears_in": list(info.appears_in_models),
                "has_documentation": bool(info.existing_descriptions),
                "pattern": info.pattern.pattern_type if info.pattern else None,
                "suggested_description": engine.suggest_column_description(name),
                "is_foreign_key": info.is_foreign_key,
                "references": (
                    f"{info.references_table}.{info.references_column}"
                    if info.references_table else None
                ),
            }
            for name, info in common
        ]
    except Exception as e:
        logger.error(f"find_common_columns failed: {e}")
        return []


@router.post("/columns/generate-schema")
async def generate_schema(request: GenerateSchemaRequest) -> str:
    """Generate a schema.yml file with column documentation and tests.

    Returns the YAML content as a string, ready to be saved by the extension.
    """
    path_error = _validate_project_path(request.project_path)
    if path_error:
        return f"# Error: {path_error}"

    try:
        import yaml as pyyaml

        service = _get_column_doc_service()
        engine = service._get_or_create_engine(request.project_path)

        if not engine.column_catalog:
            engine.analyze_project_columns()

        suggestions = engine.export_suggestions_to_yaml()

        if not suggestions:
            return "# No models found to generate schema for"

        # Build the schema.yml structure
        models_list = []
        for model_name, model_data in suggestions.items():
            model_entry: Dict[str, Any] = {
                "name": model_data["name"],
                "description": f"Documentation for {model_data['name']}",
                "columns": [],
            }
            for col in model_data.get("columns", []):
                col_entry: Dict[str, Any] = {
                    "name": col["name"],
                    "description": col.get("description", ""),
                }
                if request.include_tests and "tests" in col:
                    col_entry["tests"] = col["tests"]
                model_entry["columns"].append(col_entry)
            models_list.append(model_entry)

        schema = {"version": 2, "models": models_list}
        return pyyaml.dump(schema, default_flow_style=False, sort_keys=False)

    except Exception as e:
        logger.error(f"generate_schema failed: {e}")
        return f"# Error generating schema: {e}"


@router.post("/generate")
async def generate_dbt_model(request: GenerateModelRequest) -> Dict[str, Any]:
    """Generate a dbt model template from the knowledge service.

    Returns a dict with ``sql``, ``documentation``, and ``tests``.
    """
    try:
        from adam.knowledge.dbt_knowledge import DBTModelRequest as KnowledgeModelRequest, ModelLayer

        # Map string layer to enum
        layer_map = {
            "staging": ModelLayer.STAGING,
            "intermediate": ModelLayer.INTERMEDIATE,
            "mart_fact": ModelLayer.MART_FACT,
            "mart_dimension": ModelLayer.MART_DIMENSION,
            "utility": ModelLayer.UTILITY,
        }
        layer = layer_map.get(request.layer, ModelLayer.STAGING)

        knowledge_request = KnowledgeModelRequest(
            model_name=request.model_name,
            layer=layer,
            source_table=request.source_table,
            source_schema=request.source_schema,
            business_logic=request.business_logic,
            grain=request.grain,
            unique_key=request.unique_key,
            timestamp_column=request.timestamp_column,
            cluster_keys=request.cluster_keys,
            domain=request.domain,
        )

        service = _get_knowledge_service()
        sql = service.generate_model_template(knowledge_request)
        suggestions = service.suggest_improvements(sql)

        # Build a simple YAML documentation block
        documentation = (
            f"version: 2\n\nmodels:\n  - name: {request.model_name}\n"
            f"    description: \"Generated {request.layer} model for {request.source_table}\"\n"
        )

        # Derive basic tests from the layer
        tests: List[str] = ["not_null", "unique"]
        if layer == ModelLayer.MART_FACT and request.unique_key:
            tests.append(f"unique on {request.unique_key}")

        return {
            "sql": sql,
            "documentation": documentation,
            "tests": tests,
            "suggestions": suggestions,
        }
    except Exception as e:
        logger.error(f"generate_dbt_model failed: {e}")
        return {
            "sql": "",
            "documentation": "",
            "tests": [],
            "error": str(e),
        }
