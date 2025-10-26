"""
DBT Column Documentation API Routes
Endpoints for intelligent column documentation generation
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import logging

from ..services.dbt_column_service import DBTColumnDocumentationService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize service
column_service = DBTColumnDocumentationService()

# Request/Response Models
class ColumnAnalysisRequest(BaseModel):
    project_path: str = Field(..., description="Path to DBT project")
    use_ai: bool = Field(True, description="Use AI for enhanced descriptions")

class DocumentModelRequest(BaseModel):
    project_path: str = Field(..., description="Path to DBT project")
    model_name: str = Field(..., description="Name of the model to document")
    use_ai: bool = Field(True, description="Use AI for description generation")
    preserve_existing: bool = Field(True, description="Preserve existing descriptions")

class CommonColumnsRequest(BaseModel):
    project_path: str = Field(..., description="Path to DBT project")
    min_occurrences: int = Field(2, description="Minimum occurrences to be considered common")

class GenerateSchemaRequest(BaseModel):
    project_path: str = Field(..., description="Path to DBT project")
    model_path: Optional[str] = Field(None, description="Specific model path to generate for")
    use_ai: bool = Field(True, description="Use AI for enhanced descriptions")
    include_tests: bool = Field(True, description="Include test suggestions")

class ImprovementsRequest(BaseModel):
    project_path: str = Field(..., description="Path to DBT project")
    priority: str = Field("high", description="Priority filter: high, medium, low, or all")

@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_columns(request: ColumnAnalysisRequest):
    """
    Analyze all columns in a DBT project
    Detects patterns, finds common columns, and provides documentation insights
    """
    try:
        logger.info(f"Analyzing columns for project: {request.project_path}")

        result = await column_service.analyze_columns(
            project_path=request.project_path,
            use_ai=request.use_ai
        )

        return result
    except Exception as e:
        logger.error(f"Column analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/document-model", response_model=Dict[str, Any])
async def document_model_columns(request: DocumentModelRequest):
    """
    Generate column documentation for a specific model
    """
    try:
        logger.info(f"Documenting columns for model: {request.model_name}")

        result = await column_service.document_model_columns(
            project_path=request.project_path,
            model_name=request.model_name,
            use_ai=request.use_ai,
            preserve_existing=request.preserve_existing
        )

        return result
    except Exception as e:
        logger.error(f"Model documentation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/common-columns", response_model=List[Dict[str, Any]])
async def find_common_columns(request: CommonColumnsRequest):
    """
    Find columns that appear in multiple models
    These are candidates for standardized documentation
    """
    try:
        logger.info(f"Finding common columns in project: {request.project_path}")

        result = await column_service.find_common_columns(
            project_path=request.project_path,
            min_occurrences=request.min_occurrences
        )

        return result
    except Exception as e:
        logger.error(f"Common columns search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-schema", response_model=str)
async def generate_schema_yml(request: GenerateSchemaRequest):
    """
    Generate schema.yml content with column documentation
    Returns YAML string ready to be saved
    """
    try:
        logger.info(f"Generating schema.yml for project: {request.project_path}")

        yaml_content = await column_service.generate_schema_yml(
            project_path=request.project_path,
            model_path=request.model_path,
            use_ai=request.use_ai,
            include_tests=request.include_tests
        )

        return yaml_content
    except Exception as e:
        logger.error(f"Schema generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/suggest-improvements", response_model=List[Dict[str, Any]])
async def suggest_improvements(request: ImprovementsRequest):
    """
    Suggest documentation improvements prioritized by impact
    """
    try:
        logger.info(f"Suggesting improvements for project: {request.project_path}")

        result = await column_service.suggest_documentation_improvements(
            project_path=request.project_path,
            priority=request.priority
        )

        return result
    except Exception as e:
        logger.error(f"Improvement suggestions failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cache/{project_id}")
async def clear_cache(project_id: str):
    """Clear cached column analysis for a project"""
    try:
        column_service.clear_cache(project_id)
        return {"message": f"Cache cleared for project {project_id}"}
    except Exception as e:
        logger.error(f"Cache clear failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "dbt-column-documentation"}