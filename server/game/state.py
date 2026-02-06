"""Game state management."""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
import random

from server.game.roles import Role, Team, get_team, is_evil, can_see_evil, knows_teammates
from server.game.rules import get_rules, GameRules


class GamePhase(str, Enum):
    """Phases of the game."""
    ROLE_ASSIGNMENT = "role_assignment"
    NIGHT_PHASE = "night_phase"
    TEAM_SELECTION = "team_selection"
    DISCUSSION = "discussion"
    TEAM_VOTE = "team_vote"
    QUEST_EXECUTION = "quest_execution"
    ASSASSINATION_DISCUSSION = "assassination_discussion"  # Evil team discusses before assassination
    ASSASSINATION = "assassination"
    GAME_OVER = "game_over"


class GameStatus(str, Enum):
    """Overall game status."""
    WAITING = "waiting"
    IN_PROGRESS = "in_progress"
    FINISHED = "finished"


@dataclass
class Player:
    """Represents a player in the game."""
    seat: int
    name: str
    role: Optional[Role] = None
    is_human: bool = False
    model_name: Optional[str] = None
    provider: Optional[str] = None
    
    @property
    def team(self) -> Optional[Team]:
        if self.role is None:
            return None
        return get_team(self.role)


@dataclass
class QuestResult:
    """Result of a quest."""
    round: int
    team_size: int
    success: Optional[bool] = None
    fail_votes: int = 0
    team_members: List[int] = field(default_factory=list)
    quest_votes: Dict[int, bool] = field(default_factory=dict)  # seat -> success


@dataclass
class VoteResult:
    """Result of a team vote."""
    round: int
    attempt: int
    votes: Dict[int, bool] = field(default_factory=dict)  # seat -> approve
    approved: bool = False
    proposed_team: List[int] = field(default_factory=list)
    leader: int = 0


@dataclass
class DiscussionMessage:
    """A message in the discussion phase."""
    seat: int
    player_name: str
    content: str
    round: int = 1
    attempt: int = 1  # Which vote attempt this discussion belongs to
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class GameState:
    """Complete state of a game."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: GameStatus = GameStatus.WAITING
    phase: GamePhase = GamePhase.ROLE_ASSIGNMENT
    
    player_count: int = 5
    players: List[Player] = field(default_factory=list)
    
    current_round: int = 1  # 1-5
    current_leader: int = 0  # seat index
    vote_attempt: int = 1  # 1-5 (5 failures = evil wins)
    
    quest_results: List[QuestResult] = field(default_factory=list)
    vote_history: List[VoteResult] = field(default_factory=list)
    discussion_history: List[DiscussionMessage] = field(default_factory=list)
    
    proposed_team: List[int] = field(default_factory=list)
    current_votes: Dict[int, bool] = field(default_factory=dict)
    current_quest_votes: Dict[int, bool] = field(default_factory=dict)  # seat -> success
    
    winner: Optional[Team] = None
    assassinated_player: Optional[int] = None
    
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: Optional[str] = None
    
    # For tracking discussion progress
    current_discussion_seat: int = 0  # Next seat to check for speaker
    current_speaker_seat: Optional[int] = None  # The player who is currently speaking
    discussion_speakers_count: int = 0  # How many players have spoken (excluding leader)
    discussion_complete: bool = False
    
    # For tracking assassination discussion progress
    assassination_discussion_history: List[DiscussionMessage] = field(default_factory=list)
    assassination_discussion_speakers: List[int] = field(default_factory=list)  # Evil players who have spoken
    assassination_discussion_complete: bool = False
    
    # For human player waiting
    waiting_for_human: bool = False
    human_action_type: Optional[str] = None
    
    @property
    def rules(self) -> GameRules:
        return get_rules(self.player_count)
    
    @property
    def good_wins(self) -> int:
        return sum(1 for q in self.quest_results if q.success is True)
    
    @property
    def evil_wins(self) -> int:
        return sum(1 for q in self.quest_results if q.success is False)
    
    def get_player(self, seat: int) -> Optional[Player]:
        for p in self.players:
            if p.seat == seat:
                return p
        return None
    
    def get_leader(self) -> Player:
        return self.players[self.current_leader]
    
    def next_leader(self):
        """Move to the next leader."""
        self.current_leader = (self.current_leader + 1) % self.player_count
    
    def get_visible_evil_players(self, for_seat: int) -> List[int]:
        """Get the seats of evil players visible to a player."""
        player = self.get_player(for_seat)
        if player is None or player.role is None:
            return []
        
        visible = []
        
        # Merlin sees evil (except Mordred in extended game)
        if can_see_evil(player.role):
            for p in self.players:
                if p.role and is_evil(p.role):
                    visible.append(p.seat)
        
        # Evil players see each other (except Oberon in extended game)
        if knows_teammates(player.role):
            for p in self.players:
                if p.seat != for_seat and p.role and is_evil(p.role):
                    visible.append(p.seat)
        
        return visible
    
    def assign_roles(self):
        """Randomly assign roles to players."""
        roles = self.rules.roles.copy()
        random.shuffle(roles)
        
        for player, role in zip(self.players, roles):
            player.role = role
    
    def to_dict(self, for_seat: Optional[int] = None, reveal_all: bool = False) -> Dict[str, Any]:
        """Convert game state to dictionary for API response.
        
        Args:
            for_seat: If provided, only show role info visible to this player
            reveal_all: If True, reveal all information (for replay)
        """
        # Find human player seat for visibility check
        human_seat = None
        for p in self.players:
            if p.is_human:
                human_seat = p.seat
                break
        
        # Determine the viewer seat (for_seat takes priority, then human player)
        viewer_seat = for_seat if for_seat is not None else human_seat
        
        players_data = []
        for p in self.players:
            player_data = {
                "seat": p.seat,
                "name": p.name,
                "model_name": p.model_name,
                "is_human": p.is_human,
                "is_leader": p.seat == self.current_leader,
                "is_on_quest": p.seat in self.proposed_team,
            }
            
            # Role visibility
            if reveal_all or self.status == GameStatus.FINISHED:
                # God mode or game finished: show all
                player_data["role"] = p.role.value if p.role else None
                player_data["team"] = p.team.value if p.team else None
            elif viewer_seat is not None:
                if p.seat == viewer_seat:
                    # Viewer sees their own role
                    player_data["role"] = p.role.value if p.role else None
                    player_data["team"] = p.team.value if p.team else None
                elif p.seat in self.get_visible_evil_players(viewer_seat):
                    # Viewer can see this player is evil (e.g., Merlin seeing evil, or evil seeing teammates)
                    # For evil players seeing teammates, user wants to see specific role
                    viewer_player = self.get_player(viewer_seat)
                    if viewer_player and viewer_player.role and knows_teammates(viewer_player.role):
                        player_data["role"] = p.role.value if p.role else None
                    else:
                        player_data["role"] = None
                        
                    player_data["team"] = Team.EVIL.value
                else:
                    player_data["role"] = None
                    player_data["team"] = None
            else:
                player_data["role"] = None
                player_data["team"] = None
            
            players_data.append(player_data)
        
        return {
            "id": self.id,
            "status": self.status.value,
            "phase": self.phase.value,
            "player_count": self.player_count,
            "players": players_data,
            "current_round": self.current_round,
            "current_leader": self.current_leader,
            "vote_attempt": self.vote_attempt,
            "quest_results": [
                {
                    "round": q.round,
                    "team_size": q.team_size,
                    "success": q.success,
                    "fail_votes": q.fail_votes,  # Fail vote count is always public in Avalon
                    "team_members": q.team_members,
                    "quest_votes": q.quest_votes if reveal_all or self.status == GameStatus.FINISHED else {},
                }
                for q in self.quest_results
            ],
            "vote_history": [
                {
                    "round": v.round,
                    "attempt": v.attempt,
                    "votes": v.votes,
                    "approved": v.approved,
                }
                for v in self.vote_history
            ],
            "discussion_history": [
                {
                    "seat": d.seat,
                    "player_name": d.player_name,
                    "content": d.content,
                    "round": d.round,
                    "attempt": d.attempt,
                    "timestamp": d.timestamp,
                }
                for d in self.discussion_history
            ],
            "assassination_discussion_history": [
                {
                    "seat": d.seat,
                    "player_name": d.player_name,
                    "content": d.content,
                    "timestamp": d.timestamp,
                }
                for d in self.assassination_discussion_history
            ],
            "proposed_team": self.proposed_team,
            "winner": self.winner.value if self.winner else None,
            "assassinated_player": self.assassinated_player,
            "waiting_for_human": self.waiting_for_human,
            "human_action_type": self.human_action_type,
        }
