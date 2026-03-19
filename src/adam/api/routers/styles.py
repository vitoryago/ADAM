"""
Response Style Management Router for ADAM 4.0
Provides endpoints to manage and select response styles
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Optional
from pydantic import BaseModel

from adam.database import get_db
from adam.api.models import Project
from sqlalchemy import select

router = APIRouter()

# Lazy-loaded style service
_style_service = None


def _get_style_service():
    """Lazy-load ResponseStyleService to avoid import-time failures."""
    global _style_service
    if _style_service is None:
        from adam.services.response_style_service import ResponseStyleService
        _style_service = ResponseStyleService()
    return _style_service


def _get_response_style_enum():
    """Lazy-load ResponseStyle enum."""
    from adam.services.response_style_service import ResponseStyle
    return ResponseStyle


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
    style: str

class ProjectStyleUpdate(BaseModel):
    """Update project's default response style"""
    response_style: str


@router.get("/styles", response_model=Dict[str, StyleInfo])
async def get_available_styles():
    """Get all available response styles with their configurations"""
    style_service = _get_style_service()
    ResponseStyle = _get_response_style_enum()

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
    style_service = _get_style_service()
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
    style_service = _get_style_service()
    ResponseStyle = _get_response_style_enum()

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
    ResponseStyle = _get_response_style_enum()

    try:
        style_enum = ResponseStyle(style_update.response_style)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style: {style_update.response_style}"
        )

    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.settings:
        project.settings = {}

    project.settings["response_style"] = style_update.response_style

    await db.commit()
    await db.refresh(project)

    style_service = _get_style_service()
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
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    style_str = project.settings.get("response_style", "normal") if project.settings else "normal"
    style_service = _get_style_service()
    ResponseStyle = _get_response_style_enum()

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
    style_service = _get_style_service()
    ResponseStyle = _get_response_style_enum()

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
