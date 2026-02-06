"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class Role(str, Enum):
    MERLIN = "merlin"
    LOYAL_SERVANT = "loyal_servant"
    ASSASSIN = "assassin"
    MINION = "minion"


class Team(str, Enum):
    GOOD = "good"
    EVIL = "evil"


class GameStatus(str, Enum):
    WAITING = "waiting"
    IN_PROGRESS = "in_progress"
    FINISHED = "finished"


class GamePhase(str, Enum):
    ROLE_ASSIGNMENT = "role_assignment"
    NIGHT_PHASE = "night_phase"
    TEAM_SELECTION = "team_selection"
    DISCUSSION = "discussion"
    TEAM_VOTE = "team_vote"
    QUEST_EXECUTION = "quest_execution"
    ASSASSINATION = "assassination"
    GAME_OVER = "game_over"


class PlayerConfig(BaseModel):
    """Configuration for a single player."""
    seat: int
    name: str
    is_human: bool = False
    model: Optional[str] = None
    provider: Optional[str] = None


class GameCreate(BaseModel):
    """Request to create a new game."""
    player_count: int = Field(ge=5, le=10)
    players: List[PlayerConfig]


class PlayerResponse(BaseModel):
    """Player information in response."""
    seat: int
    name: str
    model_name: Optional[str] = None
    is_human: bool = False
    role: Optional[Role] = None
    team: Optional[Team] = None
    is_leader: bool = False
    is_on_quest: bool = False


class QuestResult(BaseModel):
    """Result of a quest."""
    round: int
    team_size: int
    success: Optional[bool] = None
    fail_votes: int = 0
    team_members: List[int] = []
    quest_votes: Dict[int, bool] = {}  # seat -> success (revealed only when game finished)


class VoteResult(BaseModel):
    """Result of a team vote."""
    round: int
    attempt: int
    votes: Dict[int, bool] = {}  # seat -> approve
    approved: bool = False


class DiscussionMessage(BaseModel):
    """A discussion message from a player."""
    seat: int
    player_name: str
    content: str
    round: int = 1
    attempt: int = 1
    timestamp: str


class GameResponse(BaseModel):
    """Full game state response."""
    id: str
    status: GameStatus
    phase: GamePhase
    player_count: int
    players: List[PlayerResponse]
    current_round: int = 1
    current_leader: int = 0
    vote_attempt: int = 1
    quest_results: List[QuestResult] = []
    vote_history: List[VoteResult] = []
    discussion_history: List[DiscussionMessage] = []
    proposed_team: List[int] = []
    winner: Optional[Team] = None
    assassinated_player: Optional[int] = None


class GameSummary(BaseModel):
    """Summary of a game for listing."""
    id: str
    status: GameStatus
    player_count: int
    winner: Optional[Team] = None
    created_at: str
    finished_at: Optional[str] = None


class ModelInfo(BaseModel):
    """Information about an available LLM model."""
    provider: str
    model: str
    display_name: str


class ModelStats(BaseModel):
    """Win rate statistics for a model."""
    model: str
    games_played: int
    wins: int
    win_rate: float


class RoleStats(BaseModel):
    """Win rate statistics for a role."""
    role: str
    games_played: int
    wins: int
    win_rate: float


class ModelRoleStats(BaseModel):
    """Win rate statistics for a model playing a specific role."""
    model: str
    role: str
    games_played: int
    wins: int
    win_rate: float
