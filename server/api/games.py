"""Game management API routes."""

from fastapi import APIRouter, HTTPException
from typing import List, Optional

from server.models.schemas import GameCreate, GameResponse, GameSummary
from server.game.manager import GameManager
from server.storage.repository import GameRepository

router = APIRouter()


@router.post("", response_model=GameResponse)
async def create_game(config: GameCreate):
    """Create a new game."""
    # Validate player count matches config
    if len(config.players) != config.player_count:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {config.player_count} players, got {len(config.players)}"
        )
    
    # Validate seats
    seats = [p.seat for p in config.players]
    if len(set(seats)) != len(seats):
        raise HTTPException(status_code=400, detail="Duplicate seats")
    if set(seats) != set(range(config.player_count)):
        raise HTTPException(status_code=400, detail="Invalid seat numbers")
    
    # Validate LLM players have model info
    for p in config.players:
        if not p.is_human and (not p.model or not p.provider):
            raise HTTPException(
                status_code=400,
                detail=f"Player {p.name} is not human but missing model/provider"
            )
    
    manager = GameManager.get_instance()
    state = await manager.create_game(config)
    
    return GameResponse(
        id=state.id,
        status=state.status.value,
        phase=state.phase.value,
        player_count=state.player_count,
        players=[
            {
                "seat": p.seat,
                "name": p.name,
                "model_name": p.model_name,
                "is_human": p.is_human,
                "is_leader": p.seat == state.current_leader,
                "is_on_quest": p.seat in state.proposed_team,
            }
            for p in state.players
        ],
        current_round=state.current_round,
        current_leader=state.current_leader,
        vote_attempt=state.vote_attempt,
    )


@router.get("", response_model=List[GameSummary])
async def list_games():
    """List all games."""
    repo = GameRepository()
    return await repo.list_games()


@router.get("/{game_id}")
async def get_game(game_id: str, reveal_all: bool = False):
    """Get game by ID.
    
    Args:
        reveal_all: If True, reveal all hidden information (god mode).
    """
    # First check active games
    manager = GameManager.get_instance()
    engine = manager.get_game(game_id)
    
    if engine:
        # reveal_all if game finished OR god mode requested
        should_reveal = reveal_all or engine.state.status.value == "finished"
        return engine.state.to_dict(reveal_all=should_reveal)
    
    # Then check database
    repo = GameRepository()
    game = await repo.get_game(game_id, reveal_all=reveal_all)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    return game


@router.get("/{game_id}/replay")
async def get_game_replay(game_id: str):
    """Get game replay data."""
    repo = GameRepository()
    replay = await repo.get_game_replay(game_id)
    if not replay:
        raise HTTPException(status_code=404, detail="Game not found")
    return replay


@router.get("/{game_id}/actions/{action_id}/llm-details")
async def get_action_llm_details(game_id: str, action_id: str):
    """Get LLM call details for a specific action."""
    repo = GameRepository()
    details = await repo.get_action_llm_details(game_id, action_id)
    if not details:
        raise HTTPException(status_code=404, detail="Action not found")
    return details


@router.get("/{game_id}/discussion-llm-details")
async def get_discussion_llm_details(game_id: str, round_num: int, player_seat: int, timestamp: str, attempt: Optional[int] = None, action_type: str = "discussion"):
    """Get LLM call details for a discussion message by round, seat, attempt and timestamp."""
    repo = GameRepository()
    action_id = await repo.get_discussion_action_id(game_id, round_num, player_seat, timestamp, attempt, action_type=action_type)
    if not action_id:
        raise HTTPException(status_code=404, detail="Discussion action not found")
    
    details = await repo.get_action_llm_details(game_id, action_id)
    if not details:
        raise HTTPException(status_code=404, detail="Action details not found")
    return details


@router.get("/{game_id}/vote-llm-details")
async def get_vote_llm_details(game_id: str, round_num: int, attempt: int, player_seat: int, action_type: str = "team_vote"):
    """Get LLM call details for a vote action by round, attempt, seat and action type."""
    repo = GameRepository()
    action_id = await repo.get_vote_action_id(game_id, round_num, attempt, player_seat, action_type)
    if not action_id:
        raise HTTPException(status_code=404, detail="Vote action not found")
    
    details = await repo.get_action_llm_details(game_id, action_id)
    if not details:
        raise HTTPException(status_code=404, detail="Action details not found")
    return details
