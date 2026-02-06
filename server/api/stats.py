"""Statistics API routes."""

from fastapi import APIRouter
from typing import List

from server.models.schemas import ModelStats, RoleStats, ModelRoleStats

router = APIRouter()


@router.get("/models", response_model=List[ModelStats])
async def get_model_stats():
    """Get win rate statistics by model."""
    from server.storage.repository import GameRepository
    repo = GameRepository()
    return await repo.get_model_stats()


@router.get("/roles", response_model=List[RoleStats])
async def get_role_stats():
    """Get win rate statistics by role."""
    from server.storage.repository import GameRepository
    repo = GameRepository()
    return await repo.get_role_stats()


@router.get("/model-roles", response_model=List[ModelRoleStats])
async def get_model_role_stats():
    """Get win rate statistics by model and role combination."""
    from server.storage.repository import GameRepository
    repo = GameRepository()
    return await repo.get_model_role_stats()
