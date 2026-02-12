"""In-memory repository for game storage without MongoDB.

Used when MongoDB is not available (e.g., training servers).
Provides the same interface as GameRepository for batch game running,
and supports direct export to JSONL without needing a database.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from server.models.schemas import (
    GameCreate, GameResponse, PlayerResponse,
    GameStatus, GamePhase,
)


class InMemoryRepository:
    """In-memory repository that replaces GameRepository when MongoDB is unavailable.

    Stores all game and action data in dictionaries, enabling:
    - Batch game running without MongoDB
    - Direct JSONL export from memory
    """

    def __init__(self):
        self.games: Dict[str, Dict[str, Any]] = {}  # game_id -> game_doc
        self.actions: Dict[str, List[Dict[str, Any]]] = {}  # game_id -> [action_docs]

    async def create_game(self, config: GameCreate) -> GameResponse:
        """Create a new game in memory."""
        game_id = str(uuid.uuid4())

        players = [
            {
                "seat": p.seat,
                "name": p.name,
                "role": None,
                "is_human": p.is_human,
                "model_name": p.model,
                "provider": p.provider,
            }
            for p in config.players
        ]

        game_doc = {
            "_id": game_id,
            "status": "waiting",
            "phase": "role_assignment",
            "player_count": config.player_count,
            "winner": None,
            "created_at": datetime.utcnow(),
            "finished_at": None,
            "players": players,
            "rounds": [],
        }

        self.games[game_id] = game_doc
        self.actions[game_id] = []

        return GameResponse(
            id=game_id,
            status=GameStatus.WAITING,
            phase=GamePhase.ROLE_ASSIGNMENT,
            player_count=config.player_count,
            players=[
                PlayerResponse(
                    seat=p.seat,
                    name=p.name,
                    model_name=p.model,
                    is_human=p.is_human,
                )
                for p in config.players
            ],
        )

    async def save_action(
        self,
        game_id: str,
        round_num: int,
        action_type: str,
        player_seat: int,
        content: Optional[str] = None,
        vote: Optional[bool] = None,
        target_seat: Optional[int] = None,
        vote_attempt: Optional[int] = None,
        proposed_team: Optional[List[int]] = None,
        llm_input: Optional[Dict[str, Any]] = None,
        llm_output: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a game action in memory. Returns an action ID."""
        action_id = str(uuid.uuid4())

        action_doc = {
            "_id": action_id,
            "game_id": game_id,
            "round_num": round_num,
            "action_type": action_type,
            "player_seat": player_seat,
            "content": content,
            "vote": vote,
            "target_seat": target_seat,
            "timestamp": datetime.utcnow(),
            "vote_attempt": vote_attempt,
            "proposed_team": proposed_team,
            "llm_input": llm_input,
            "llm_output": llm_output,
        }

        if game_id not in self.actions:
            self.actions[game_id] = []
        self.actions[game_id].append(action_doc)

        return action_id

    async def update_game_state(self, state):
        """Update game state in memory."""
        game_id = state.id
        if game_id not in self.games:
            return

        game = self.games[game_id]

        players = [
            {
                "seat": p.seat,
                "name": p.name,
                "role": p.role.value if p.role else None,
                "is_human": p.is_human,
                "model_name": p.model_name,
                "provider": p.provider,
            }
            for p in state.players
        ]

        game["status"] = state.status.value
        game["phase"] = state.phase.value
        game["winner"] = state.winner.value if state.winner else None
        game["finished_at"] = (
            datetime.fromisoformat(state.finished_at) if state.finished_at else None
        )
        game["players"] = players

    def set_batch_metadata(self, game_id: str, batch_id: str, batch_tag: Optional[str] = None):
        """Set batch metadata on a game (replaces MongoDB update_one)."""
        if game_id in self.games:
            self.games[game_id]["source"] = "batch"
            self.games[game_id]["batch_id"] = batch_id
            self.games[game_id]["batch_tag"] = batch_tag

    def get_finished_games(
        self,
        batch_id: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get finished games, optionally filtered by batch_id or tag."""
        results = []
        for game in self.games.values():
            if game.get("status") != "finished":
                continue
            if batch_id and game.get("batch_id") != batch_id:
                continue
            if tag and game.get("batch_tag") != tag:
                continue
            results.append(game)

        results.sort(key=lambda g: g.get("created_at", datetime.min))
        return results

    def get_game_actions(self, game_id: str) -> List[Dict[str, Any]]:
        """Get all actions for a game, sorted by timestamp."""
        actions = self.actions.get(game_id, [])
        return sorted(actions, key=lambda a: a.get("timestamp", datetime.min))
