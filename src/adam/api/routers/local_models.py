"""Local Models API — exposes discovered local models to frontends."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class LocalModelResponse(BaseModel):
    """Response schema for a local model entry."""
    model_id: str
    display_name: str
    backend: str
    parameter_count: int
    quantization: str
    available: bool


def get_local_provider():
    """Get the global LocalModelProvider instance from the LLM service."""
    try:
        from adam.services.llm_service import LLMService
        service = LLMService()
        if service.llm_client and service.llm_client._local_provider:
            return service.llm_client._local_provider
    except Exception:
        pass
    from adam.llm.local_provider import LocalModelProvider
    return LocalModelProvider(endpoints=[])


@router.get("", response_model=List[LocalModelResponse])
async def list_local_models():
    """Return all available local models."""
    provider = get_local_provider()
    available = provider.get_available_models()
    return [
        LocalModelResponse(
            model_id=m.model_id,
            display_name=m.display_name,
            backend=m.backend,
            parameter_count=m.parameter_count,
            quantization=m.quantization,
            available=m.available,
        )
        for m in available
    ]
