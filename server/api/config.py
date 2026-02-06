"""Configuration API routes."""

from fastapi import APIRouter
from typing import List

from server.config import settings
from server.models.schemas import ModelInfo

router = APIRouter()


@router.get("/models", response_model=List[ModelInfo])
async def get_available_models():
    """Get list of all available LLM models."""
    return settings.get_all_models()
