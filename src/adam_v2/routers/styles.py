"""
Response Style Management Router for ADAM v2.0
Provides endpoints to manage and select response styles
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Optional
from pydantic import BaseModel

from database import get_db
from models import Project
from services.response_style_service import ResponseStyleService, ResponseStyle

router = APIRouter(prefix="/api", tags=["styles"])

# Response models
class StyleInfo(BaseModel):
    """Information about a response style"""
    name: str
    description: str
    temperature: float
    length_preference: str
    is_current: bool = False

class StyleSettings(BaseModel):
    """Settings for response style"""
    style: str  # concise, normal, explanatory, formal, friendly, educational, creative
    
class ProjectStyleUpdate(BaseModel):
    """Update project's default response style"""
    response_style: str

# Initialize style service
style_service = ResponseStyleService()

@router.get("/styles", response_model=Dict[str, StyleInfo])
async def get_available_styles():
    """Get all available response styles with their configurations"""
    styles = {}
    current_style = style_service.get_current_style()
    
    for style_enum in ResponseStyle:
        config = style_service.get_style_config(style_enum)
        styles[style_enum.value] = StyleInfo(
            name=config.name,
            description=style_service.get_available_styles().get(style_enum.value, ""),
            temperature=config.temperature,
            length_preference=config.length_preference,
            is_current=(style_enum == current_style)
        )
    
    return styles

@router.get("/styles/current")
async def get_current_style():
    """Get the current response style"""
    current = style_service.get_current_style()
    config = style_service.get_style_config(current)
    
    return {
        "style": current.value,
        "name": config.name,
        "temperature": config.temperature,
        "length_preference": config.length_preference
    }

@router.post("/styles/set")
async def set_global_style(settings: StyleSettings):
    """Set the global default response style"""
    try:
        style_enum = ResponseStyle(settings.style)
        style_service.set_style(style_enum)
        config = style_service.get_style_config(style_enum)
        
        return {
            "message": f"Response style set to {config.name}",
            "style": style_enum.value,
            "temperature": config.temperature,
            "length_preference": config.length_preference
        }
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style: {settings.style}. Valid options: {', '.join([s.value for s in ResponseStyle])}"
        )

@router.post("/projects/{project_id}/style")
async def update_project_style(
    project_id: str,
    style_update: ProjectStyleUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a project's default response style"""
    # Validate style
    try:
        style_enum = ResponseStyle(style_update.response_style)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style: {style_update.response_style}"
        )
    
    # Get project
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Update project settings
    if not project.settings:
        project.settings = {}
    
    project.settings["response_style"] = style_update.response_style
    
    await db.commit()
    await db.refresh(project)
    
    config = style_service.get_style_config(style_enum)
    
    return {
        "message": f"Project response style updated to {config.name}",
        "project_id": project_id,
        "style": style_update.response_style,
        "temperature": config.temperature,
        "length_preference": config.length_preference
    }

@router.get("/projects/{project_id}/style")
async def get_project_style(
    project_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a project's response style settings"""
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    style_str = project.settings.get("response_style", "normal") if project.settings else "normal"
    
    try:
        style_enum = ResponseStyle(style_str)
        config = style_service.get_style_config(style_enum)
        
        return {
            "project_id": project_id,
            "style": style_str,
            "name": config.name,
            "description": style_service.get_available_styles().get(style_str, ""),
            "temperature": config.temperature,
            "length_preference": config.length_preference
        }
    except ValueError:
        # Fallback to normal if invalid style stored
        return {
            "project_id": project_id,
            "style": "normal",
            "name": "Normal",
            "description": "Balanced, default responses",
            "temperature": 0.7,
            "length_preference": "medium"
        }

@router.get("/styles/{style}/preview")
async def preview_style(style: str):
    """Preview what a response would look like with a specific style"""
    try:
        style_enum = ResponseStyle(style)
        config = style_service.get_style_config(style_enum)
        
        sample_prompt = "Explain how neural networks work"
        enhanced_prompt = style_service.enhance_prompt_for_style(sample_prompt, style_enum)
        
        return {
            "style": style,
            "name": config.name,
            "system_prompt": config.system_prompt,
            "temperature": config.temperature,
            "length_preference": config.length_preference,
            "sample_prompt": sample_prompt,
            "enhanced_prompt": enhanced_prompt,
            "description": style_service.get_available_styles().get(style, "")
        }
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style: {style}"
        )