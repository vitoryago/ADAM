"""
Tools API Router for ADAM v2
Provides endpoints for web search, code execution, and file generation
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from adam.tools import WebSearchTool, CodeExecutor, FileGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["tools"])

# Initialize tools
web_search = WebSearchTool(provider='auto', cache_results=True)
code_executor = CodeExecutor(use_docker=False)  # Start without Docker for simplicity
file_generator = FileGenerator()


# Request/Response models
class WebSearchRequest(BaseModel):
    """Web search request model"""
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, ge=1, le=20, description="Maximum results to return")
    include_content: bool = Field(False, description="Include full page content")
    provider: Optional[str] = Field('auto', description="Search provider")


class WebSearchResponse(BaseModel):
    """Web search response model"""
    query: str
    results: List[Dict[str, Any]]
    timestamp: datetime
    cached: bool = False


class CodeExecutionRequest(BaseModel):
    """Code execution request model"""
    code: str = Field(..., description="Code to execute")
    language: Optional[str] = Field(None, description="Programming language")
    stdin: Optional[str] = Field(None, description="Input for the program")
    timeout: Optional[int] = Field(30, ge=1, le=300, description="Execution timeout in seconds")


class CodeExecutionResponse(BaseModel):
    """Code execution response model"""
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    language: str
    success: bool
    truncated: bool = False


class FileGenerationRequest(BaseModel):
    """File generation request model"""
    file_type: str = Field(..., description="Type of file to generate")
    name: str = Field(..., description="Name for the file/component")
    options: Optional[Dict[str, Any]] = Field({}, description="Additional options")
    save_to_disk: bool = Field(False, description="Save file to disk")


class FileGenerationResponse(BaseModel):
    """File generation response model"""
    filename: str
    content: str
    file_type: str
    created_at: datetime
    saved: bool = False
    path: Optional[str] = None


# Endpoints
@router.post("/search", response_model=WebSearchResponse)
async def search_web(request: WebSearchRequest):
    """
    Search the web using available search providers
    """
    try:
        # Check if using different provider
        if request.provider != 'auto':
            search_tool = WebSearchTool(provider=request.provider)
        else:
            search_tool = web_search
        
        # Perform search
        results = search_tool.search(
            query=request.query,
            max_results=request.max_results,
            include_content=request.include_content
        )
        
        # Convert to dict format
        results_dict = [r.to_dict() for r in results]
        
        # Check if cached
        cached = request.query in search_tool.cache if search_tool.cache_results else False
        
        return WebSearchResponse(
            query=request.query,
            results=results_dict,
            timestamp=datetime.now(),
            cached=cached
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/summarize")
async def search_and_summarize(request: WebSearchRequest):
    """
    Search and provide a structured summary
    """
    try:
        summary = web_search.search_and_summarize(
            query=request.query,
            max_results=request.max_results
        )
        return summary
        
    except Exception as e:
        logger.error(f"Search summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute", response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest):
    """
    Execute code in a sandboxed environment
    """
    try:
        # Update timeout if specified
        executor = code_executor
        if request.timeout != 30:
            executor = CodeExecutor(use_docker=False, timeout=request.timeout)
        
        # Execute code
        result = executor.execute(
            code=request.code,
            language=request.language,
            stdin=request.stdin
        )
        
        return CodeExecutionResponse(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            execution_time=result.execution_time,
            language=result.language,
            success=result.success,
            truncated=result.truncated
        )
        
    except Exception as e:
        logger.error(f"Code execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/execute/languages")
async def get_supported_languages():
    """
    Get list of supported programming languages
    """
    try:
        # Test which languages are available
        test_results = code_executor.test_setup()
        
        languages = []
        for lang, available in test_results.items():
            languages.append({
                'language': lang,
                'available': available,
                'docker_required': lang not in ['python', 'javascript', 'sql']
            })
        
        return {
            'languages': languages,
            'docker_available': code_executor.use_docker
        }
        
    except Exception as e:
        logger.error(f"Failed to get languages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=FileGenerationResponse)
async def generate_file(request: FileGenerationRequest):
    """
    Generate a file from templates
    """
    try:
        # Generate file
        generated = file_generator.generate(
            file_type=request.file_type,
            name=request.name,
            **request.options
        )
        
        # Save if requested
        path = None
        if request.save_to_disk:
            from pathlib import Path
            # Save to a temp directory for safety
            temp_dir = Path("/tmp/adam_generated")
            temp_dir.mkdir(exist_ok=True)
            path = generated.save(temp_dir)
            
        return FileGenerationResponse(
            filename=generated.filename,
            content=generated.content,
            file_type=generated.file_type,
            created_at=generated.created_at,
            saved=request.save_to_disk,
            path=str(path) if path else None
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"File generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate/templates")
async def get_file_templates():
    """
    Get list of available file templates
    """
    templates = file_generator.list_templates()
    
    # Group by category
    categories = {
        'python': ['python_script', 'python_class', 'python_test'],
        'javascript': ['javascript_module', 'react_component'],
        'docker': ['docker', 'docker_compose'],
        'config': ['requirements', 'package_json', 'gitignore', 'env'],
        'documentation': ['readme'],
        'database': ['sql_schema'],
        'api': ['api_endpoint']
    }
    
    return {
        'templates': templates,
        'categories': categories
    }


@router.post("/generate/project")
async def generate_project(
    project_type: str = Field(..., description="Type of project (python, react, api)"),
    name: str = Field(..., description="Project name")
):
    """
    Generate a complete project structure
    """
    try:
        files = []
        
        if project_type == 'python':
            # Generate Python project
            files.extend([
                file_generator.generate('python_class', name),
                file_generator.generate('python_test', name),
                file_generator.generate('readme', name),
                file_generator.generate('requirements', 'requirements'),
                file_generator.generate('gitignore', 'gitignore'),
                file_generator.generate('docker', 'Dockerfile')
            ])
            
        elif project_type == 'react':
            # Generate React project
            files.extend([
                file_generator.generate('react_component', f"{name}App"),
                file_generator.generate('react_component', f"{name}Header"),
                file_generator.generate('react_component', f"{name}Main"),
                file_generator.generate('package_json', name),
                file_generator.generate('readme', name)
            ])
            
        elif project_type == 'api':
            # Generate API project
            files.extend([
                file_generator.generate('api_endpoint', name),
                file_generator.generate('python_class', f"{name}Service"),
                file_generator.generate('python_test', f"{name}API"),
                file_generator.generate('docker_compose', 'docker-compose'),
                file_generator.generate('readme', name)
            ])
        else:
            raise ValueError(f"Unknown project type: {project_type}")
        
        # Convert to response format
        response_files = []
        for f in files:
            response_files.append({
                'filename': f.filename,
                'content': f.content[:1000] + '...' if len(f.content) > 1000 else f.content,
                'file_type': f.file_type
            })
        
        return {
            'project_type': project_type,
            'project_name': name,
            'files': response_files,
            'file_count': len(files)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Project generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check for tools
@router.get("/health")
async def tools_health():
    """Check health of tools service"""
    return {
        'status': 'healthy',
        'tools': {
            'web_search': 'available',
            'code_executor': 'available',
            'file_generator': 'available'
        },
        'timestamp': datetime.now()
    }