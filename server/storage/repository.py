"""Data access layer for game storage using MongoDB."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId
import uuid

from server.models.database import get_db
from server.models.schemas import (
    GameCreate, GameResponse, GameSummary, PlayerResponse,
    QuestResult, VoteResult, DiscussionMessage,
    ModelStats, RoleStats, ModelRoleStats, GameStatus, GamePhase
)
from server.game.state import GameState, Player
from server.game.state import GameStatus as StateGameStatus, GamePhase as StateGamePhase


class GameRepository:
    """Repository for game data operations using MongoDB."""
    
    async def create_game(self, config: GameCreate) -> GameResponse:
        """Create a new game in the database."""
        game_id = str(uuid.uuid4())
        db = get_db()
        
        # Build players list
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
        
        # Create game document
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
        
        await db.games.insert_one(game_doc)
        
        # Return response
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
    
    async def get_game(self, game_id: str, reveal_all: bool = False) -> Optional[GameResponse]:
        """Get a game by ID.
        
        Args:
            game_id: The game ID
            reveal_all: If True, reveal all hidden information (roles, quest votes, etc.)
        """
        db = get_db()
        game = await db.games.find_one({"_id": game_id})
        
        if not game:
            return None
        
        # Get actions for this game
        actions = await db.actions.find({"game_id": game_id}).sort("timestamp", 1).to_list(None)
        
        return self._game_to_response(game, actions, reveal_all=reveal_all)
    
    async def list_games(self, limit: int = 100) -> List[GameSummary]:
        """List all games."""
        db = get_db()
        cursor = db.games.find().sort("created_at", -1).limit(limit)
        games = await cursor.to_list(length=limit)
        
        return [
            GameSummary(
                id=g["_id"],
                status=GameStatus(g["status"]),
                player_count=g["player_count"],
                winner=g.get("winner"),
                created_at=g["created_at"].isoformat() if g.get("created_at") else "",
                finished_at=g["finished_at"].isoformat() if g.get("finished_at") else None,
            )
            for g in games
        ]
    
    async def update_game_state(self, state: GameState):
        """Update game state in database."""
        db = get_db()
        
        # Build player updates
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
        
        update_doc = {
            "$set": {
                "status": state.status.value,
                "phase": state.phase.value,
                "winner": state.winner.value if state.winner else None,
                "finished_at": datetime.fromisoformat(state.finished_at) if state.finished_at else None,
                "players": players,
            }
        }
        
        await db.games.update_one({"_id": state.id}, update_doc)
    
    async def save_quest_result(self, game_id: str, quest: QuestResult):
        """Save a quest result."""
        db = get_db()
        
        round_doc = {
            "round_num": quest.round,
            "team_members": quest.team_size,  # Store team size or actual members
            "success": quest.success,
            "fail_votes": quest.fail_votes,
        }
        
        await db.games.update_one(
            {"_id": game_id},
            {"$push": {"rounds": round_doc}}
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
        """Save a game action. Returns the action ID as string."""
        db = get_db()
        
        action_doc = {
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
        
        result = await db.actions.insert_one(action_doc)
        return str(result.inserted_id)
    
    async def get_game_replay(self, game_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get full game replay data."""
        db = get_db()
        game = await db.games.find_one({"_id": game_id})
        
        if not game:
            return None
        
        # Get actions sorted by timestamp
        actions = await db.actions.find({"game_id": game_id}).sort("timestamp", 1).to_list(None)
        
        # Build replay data
        replay = []
        
        current_state = {
            "id": game["_id"],
            "status": game["status"],
            "phase": "role_assignment",
            "player_count": game["player_count"],
            "players": [
                {
                    "seat": p["seat"],
                    "name": p["name"],
                    "role": p.get("role"),
                    "is_human": p.get("is_human", False),
                    "model_name": p.get("model_name"),
                }
                for p in game.get("players", [])
            ],
            "current_round": 1,
            "discussion_history": [],
            "vote_history": [],
            "quest_results": [],
        }
        
        replay.append(current_state.copy())
        
        for action in actions:
            if action["action_type"] == "discussion":
                current_state["discussion_history"].append({
                    "seat": action["player_seat"],
                    "player_name": next(
                        (p["name"] for p in game.get("players", []) if p["seat"] == action["player_seat"]),
                        "Unknown"
                    ),
                    "content": action.get("content"),
                    "timestamp": action["timestamp"].isoformat() if action.get("timestamp") else None,
                })
                current_state["phase"] = "discussion"
                replay.append(current_state.copy())
        
        # Add final state
        current_state["status"] = game["status"]
        current_state["winner"] = game.get("winner")
        current_state["phase"] = "game_over" if game["status"] == "finished" else game.get("phase")
        
        for round_data in game.get("rounds", []):
            current_state["quest_results"].append({
                "round": round_data["round_num"],
                "success": round_data.get("success"),
                "fail_votes": round_data.get("fail_votes", 0),
            })
        
        replay.append(current_state)
        
        return replay
    
    async def get_model_stats(self) -> List[ModelStats]:
        """Get win rate statistics by model."""
        db = get_db()
        games = await db.games.find({"status": "finished"}).to_list(None)
        
        model_stats: Dict[str, Dict[str, int]] = {}
        
        for game in games:
            for player in game.get("players", []):
                if player.get("model_name") and not player.get("is_human", False):
                    model = player["model_name"]
                    if model not in model_stats:
                        model_stats[model] = {"games": 0, "wins": 0}
                    
                    model_stats[model]["games"] += 1
                    
                    # Check if player won
                    player_team = "good" if player.get("role") in ["merlin", "loyal_servant"] else "evil"
                    if game.get("winner") == player_team:
                        model_stats[model]["wins"] += 1
        
        return [
            ModelStats(
                model=model,
                games_played=stats["games"],
                wins=stats["wins"],
                win_rate=stats["wins"] / stats["games"] if stats["games"] > 0 else 0,
            )
            for model, stats in model_stats.items()
        ]
    
    async def get_role_stats(self) -> List[RoleStats]:
        """Get win rate statistics by role."""
        db = get_db()
        games = await db.games.find({"status": "finished"}).to_list(None)
        
        role_stats: Dict[str, Dict[str, int]] = {}
        
        for game in games:
            for player in game.get("players", []):
                role = player.get("role")
                if role:
                    if role not in role_stats:
                        role_stats[role] = {"games": 0, "wins": 0}
                    
                    role_stats[role]["games"] += 1
                    
                    player_team = "good" if role in ["merlin", "loyal_servant"] else "evil"
                    if game.get("winner") == player_team:
                        role_stats[role]["wins"] += 1
        
        return [
            RoleStats(
                role=role,
                games_played=stats["games"],
                wins=stats["wins"],
                win_rate=stats["wins"] / stats["games"] if stats["games"] > 0 else 0,
            )
            for role, stats in role_stats.items()
        ]
    
    async def get_model_role_stats(self) -> List[ModelRoleStats]:
        """Get win rate statistics by model and role combination."""
        db = get_db()
        games = await db.games.find({"status": "finished"}).to_list(None)
        
        stats: Dict[tuple, Dict[str, int]] = {}
        
        for game in games:
            for player in game.get("players", []):
                model_name = player.get("model_name")
                role = player.get("role")
                if model_name and role and not player.get("is_human", False):
                    key = (model_name, role)
                    if key not in stats:
                        stats[key] = {"games": 0, "wins": 0}
                    
                    stats[key]["games"] += 1
                    
                    player_team = "good" if role in ["merlin", "loyal_servant"] else "evil"
                    if game.get("winner") == player_team:
                        stats[key]["wins"] += 1
        
        return [
            ModelRoleStats(
                model=model,
                role=role,
                games_played=s["games"],
                wins=s["wins"],
                win_rate=s["wins"] / s["games"] if s["games"] > 0 else 0,
            )
            for (model, role), s in stats.items()
        ]
    
    async def get_action_llm_details(self, game_id: str, action_id: str) -> Optional[Dict[str, Any]]:
        """Get LLM call details for a specific action."""
        db = get_db()
        
        try:
            obj_id = ObjectId(action_id)
        except Exception:
            return None
        
        action = await db.actions.find_one({"_id": obj_id, "game_id": game_id})
        
        if not action:
            return None
        
        return {
            "id": str(action["_id"]),
            "action_type": action.get("action_type"),
            "player_seat": action.get("player_seat"),
            "content": action.get("content"),
            "round_num": action.get("round_num"),
            "timestamp": action["timestamp"].isoformat() if action.get("timestamp") else None,
            "llm_input": action.get("llm_input"),
            "llm_output": action.get("llm_output"),
        }
    
    async def get_discussion_action_id(
        self, 
        game_id: str, 
        round_num: int, 
        player_seat: int, 
        timestamp: str, 
        vote_attempt: Optional[int] = None,
        action_type: str = "discussion",
    ) -> Optional[str]:
        """Find action ID by game, round, seat, vote_attempt and approximate timestamp."""
        db = get_db()
        
        # Build query
        query = {
            "game_id": game_id,
            "round_num": round_num,
            "player_seat": player_seat,
            "action_type": action_type,
        }
        
        if vote_attempt is not None:
            query["vote_attempt"] = vote_attempt
        
        actions = await db.actions.find(query).sort("timestamp", -1).to_list(None)
        
        if not actions:
            return None
        
        # If only one action, return it
        if len(actions) == 1:
            return str(actions[0]["_id"])
        
        # Find the closest match by timestamp
        try:
            target_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00") if "Z" in timestamp else timestamp)
            target_time = target_time.replace(tzinfo=None)
            
            best_match = None
            min_diff = float('inf')
            
            for action in actions:
                if action.get("timestamp"):
                    diff = abs((action["timestamp"] - target_time).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        best_match = action
            
            return str(best_match["_id"]) if best_match else str(actions[0]["_id"])
        except (ValueError, TypeError):
            # If timestamp parsing fails, return the most recent action
            return str(actions[0]["_id"])

    async def get_vote_action_id(
        self, 
        game_id: str, 
        round_num: int, 
        attempt: int, 
        player_seat: int, 
        action_type: str = "team_vote"
    ) -> Optional[str]:
        """Find vote action ID by game, round, attempt, seat and action type."""
        db = get_db()
        
        action = await db.actions.find_one({
            "game_id": game_id,
            "round_num": round_num,
            "player_seat": player_seat,
            "action_type": action_type,
            "vote_attempt": attempt,
        })
        
        if action:
            return str(action["_id"])
        
        # Fallback: try without vote_attempt filter (for older records)
        action = await db.actions.find_one(
            {
                "game_id": game_id,
                "round_num": round_num,
                "player_seat": player_seat,
                "action_type": action_type,
            },
            sort=[("timestamp", -1)]
        )
        
        if action:
            return str(action["_id"])
        return None

    def _game_to_response(self, game: dict, actions: list, reveal_all: bool = False) -> dict:
        """Convert a database game to a response dictionary.
        
        Args:
            game: The game document from MongoDB
            actions: List of action documents for this game
            reveal_all: If True, reveal all hidden information (roles, quest votes, etc.)
        """
        # Determine if we should reveal information
        should_reveal = reveal_all or game["status"] == "finished"
        
        players = game.get("players", [])
        
        # Build discussion history from actions
        discussion_history = []
        for action in actions:
            if action.get("action_type") == "discussion":
                player_name = next(
                    (p["name"] for p in players if p["seat"] == action.get("player_seat")),
                    "Unknown"
                )
                discussion_history.append({
                    "seat": action.get("player_seat"),
                    "player_name": player_name,
                    "content": action.get("content") or "",
                    "round": action.get("round_num"),
                    "attempt": action.get("vote_attempt") or 1,
                    "timestamp": action["timestamp"].isoformat() if action.get("timestamp") else None,
                })
        
        # Build vote history
        vote_history = []
        vote_actions = [a for a in actions if a.get("action_type") == "team_vote"]
        # Group by round and attempt
        votes_by_round: Dict[tuple, Dict[int, bool]] = {}
        for action in vote_actions:
            key = (action.get("round_num"), action.get("vote_attempt") or 1)
            if key not in votes_by_round:
                votes_by_round[key] = {}
            if action.get("vote") is not None:
                votes_by_round[key][action.get("player_seat")] = action.get("vote")
        
        for (round_num, attempt), votes in votes_by_round.items():
            approvals = sum(1 for v in votes.values() if v)
            vote_history.append({
                "round": round_num,
                "attempt": attempt,
                "votes": votes,
                "approved": approvals > len(votes) // 2,
            })
        
        # Build quest votes from actions
        quest_vote_actions = [a for a in actions if a.get("action_type") == "quest_vote"]
        quest_votes_by_round: Dict[int, Dict[int, bool]] = {}
        for action in quest_vote_actions:
            round_num = action.get("round_num")
            if round_num not in quest_votes_by_round:
                quest_votes_by_round[round_num] = {}
            if action.get("vote") is not None:
                quest_votes_by_round[round_num][action.get("player_seat")] = action.get("vote")
        
        # Build quest results with votes
        quest_results = []
        rounds = game.get("rounds", [])
        
        if rounds:
            for r in sorted(rounds, key=lambda r: r["round_num"]):
                team_members = r.get("team_members") if isinstance(r.get("team_members"), list) else []
                quest_votes = quest_votes_by_round.get(r["round_num"], {}) if should_reveal else {}
                quest_results.append({
                    "round": r["round_num"],
                    "team_size": len(team_members) if team_members else r.get("team_members") or 0,
                    "success": r.get("success"),
                    "fail_votes": r.get("fail_votes", 0),  # Fail vote count is always public
                    "team_members": team_members,
                    "quest_votes": quest_votes,
                })
        elif quest_votes_by_round:
            # Fallback: rebuild quest results from quest_vote actions
            approved_teams_by_round: Dict[int, List[int]] = {}
            for (round_num, attempt), votes in votes_by_round.items():
                approvals = sum(1 for v in votes.values() if v)
                if approvals > len(votes) // 2:
                    for action in vote_actions:
                        if action.get("round_num") == round_num and (action.get("vote_attempt") or 1) == attempt:
                            if action.get("proposed_team"):
                                approved_teams_by_round[round_num] = action.get("proposed_team")
                                break
            
            for round_num in sorted(quest_votes_by_round.keys()):
                votes = quest_votes_by_round[round_num]
                fail_votes = sum(1 for v in votes.values() if not v)
                success = fail_votes == 0
                team_members = approved_teams_by_round.get(round_num, list(votes.keys()))
                quest_results.append({
                    "round": round_num,
                    "team_size": len(votes),
                    "success": success,
                    "fail_votes": fail_votes,  # Fail vote count is always public
                    "team_members": team_members,
                    "quest_votes": votes if should_reveal else {},
                })
        
        # Find human player for visibility check
        human_seat = None
        for p in players:
            if p.get("is_human"):
                human_seat = p["seat"]
                break
        
        # Build players list with role visibility control
        players_response = []
        for p in sorted(players, key=lambda p: p["seat"]):
            player_data = {
                "seat": p["seat"],
                "name": p["name"],
                "is_human": p.get("is_human", False),
                "model_name": p.get("model_name"),
                "is_leader": False,
                "is_on_quest": False,
            }
            
            # Role visibility
            if should_reveal:
                player_data["role"] = p.get("role")
                player_data["team"] = self._get_team_for_role(p.get("role"))
            elif human_seat is not None and p["seat"] == human_seat:
                player_data["role"] = p.get("role")
                player_data["team"] = self._get_team_for_role(p.get("role"))
            else:
                player_data["role"] = None
                player_data["team"] = None
            
            players_response.append(player_data)
        
        # Build assassination discussion history from actions
        assassination_discussion_history = []
        for action in actions:
            if action.get("action_type") == "assassination_discussion":
                player_name = next(
                    (p["name"] for p in players if p["seat"] == action.get("player_seat")),
                    "Unknown"
                )
                assassination_discussion_history.append({
                    "seat": action.get("player_seat"),
                    "player_name": player_name,
                    "content": action.get("content") or "",
                    "timestamp": action["timestamp"].isoformat() if action.get("timestamp") else None,
                })
        
        # Extract assassinated player from assassination action
        assassinated_player = None
        for action in actions:
            if action.get("action_type") == "assassination":
                assassinated_player = action.get("target_seat")
                break
        
        # Get current round from quest results or default to 1
        current_round = len(quest_results) + 1 if quest_results else 1
        if game["status"] == "finished":
            current_round = len(quest_results) if quest_results else 1
        
        return {
            "id": game["_id"],
            "status": game["status"],
            "phase": game.get("phase"),
            "player_count": game["player_count"],
            "players": players_response,
            "current_round": current_round,
            "current_leader": 0,
            "vote_attempt": 1,
            "discussion_history": discussion_history,
            "vote_history": vote_history,
            "quest_results": quest_results,
            "assassination_discussion_history": assassination_discussion_history,
            "proposed_team": [],
            "winner": game.get("winner"),
            "assassinated_player": assassinated_player,
            "waiting_for_human": False,
            "human_action_type": None,
        }
    
    def _get_team_for_role(self, role: Optional[str]) -> Optional[str]:
        """Get the team for a role."""
        if not role:
            return None
        evil_roles = ["assassin", "morgana", "mordred", "oberon", "minion"]
        return "evil" if role in evil_roles else "good"
    
    async def get_game_for_restore(self, game_id: str) -> Optional[GameState]:
        """Get a game from database and restore it to a GameState object.
        
        This is used for resuming games after server restart.
        
        Args:
            game_id: The game ID
            
        Returns:
            A GameState object that can be used to create a GameEngine, or None if not found.
        """
        from server.game.roles import Role, Team
        from server.game.state import (
            GameState, Player, GamePhase as StateGamePhase, 
            GameStatus as StateGameStatus, QuestResult as StateQuestResult,
            VoteResult as StateVoteResult, DiscussionMessage as StateDiscussionMessage
        )
        
        db = get_db()
        game = await db.games.find_one({"_id": game_id})
        
        if not game:
            return None
        
        # Get actions
        actions = await db.actions.find({"game_id": game_id}).sort("timestamp", 1).to_list(None)
        
        # Create players
        players = []
        for p in sorted(game.get("players", []), key=lambda p: p["seat"]):
            role = None
            if p.get("role"):
                try:
                    role = Role(p["role"])
                except ValueError:
                    pass
            
            player = Player(
                seat=p["seat"],
                name=p["name"],
                role=role,
                is_human=p.get("is_human", False),
                model_name=p.get("model_name"),
                provider=p.get("provider"),
            )
            players.append(player)
        
        # Build discussion history
        discussion_history = []
        for action in actions:
            if action.get("action_type") == "discussion":
                player_name = next(
                    (p["name"] for p in game.get("players", []) if p["seat"] == action.get("player_seat")),
                    "Unknown"
                )
                discussion_history.append(StateDiscussionMessage(
                    seat=action.get("player_seat"),
                    player_name=player_name,
                    content=action.get("content") or "",
                    round=action.get("round_num"),
                    attempt=action.get("vote_attempt") or 1,
                    timestamp=action["timestamp"].isoformat() if action.get("timestamp") else "",
                ))
        
        # Build vote history
        vote_history = []
        vote_actions = [a for a in actions if a.get("action_type") == "team_vote"]
        votes_by_key: Dict[tuple, Dict[str, Any]] = {}
        for action in vote_actions:
            key = (action.get("round_num"), action.get("vote_attempt") or 1)
            if key not in votes_by_key:
                votes_by_key[key] = {
                    "votes": {},
                    "proposed_team": action.get("proposed_team") or [],
                }
            if action.get("vote") is not None:
                votes_by_key[key]["votes"][action.get("player_seat")] = action.get("vote")
        
        for (round_num, attempt), data in votes_by_key.items():
            votes = data["votes"]
            approvals = sum(1 for v in votes.values() if v)
            vote_history.append(StateVoteResult(
                round=round_num,
                attempt=attempt,
                votes=votes,
                approved=approvals > len(votes) // 2,
                proposed_team=data["proposed_team"],
                leader=0,
            ))
        
        # Build quest results
        quest_results = []
        quest_vote_actions = [a for a in actions if a.get("action_type") == "quest_vote"]
        quest_votes_by_round: Dict[int, Dict[int, bool]] = {}
        for action in quest_vote_actions:
            round_num = action.get("round_num")
            if round_num not in quest_votes_by_round:
                quest_votes_by_round[round_num] = {}
            if action.get("vote") is not None:
                quest_votes_by_round[round_num][action.get("player_seat")] = action.get("vote")
        
        rounds = game.get("rounds", [])
        if rounds:
            for r in sorted(rounds, key=lambda r: r["round_num"]):
                team_members = r.get("team_members") if isinstance(r.get("team_members"), list) else []
                quest_votes = quest_votes_by_round.get(r["round_num"], {})
                quest_results.append(StateQuestResult(
                    round=r["round_num"],
                    team_size=len(team_members) if team_members else 0,
                    success=r.get("success"),
                    fail_votes=r.get("fail_votes", 0),
                    team_members=team_members,
                    quest_votes=quest_votes,
                ))
        elif quest_votes_by_round:
            # Fallback
            approved_teams_by_round: Dict[int, List[int]] = {}
            for (round_num, attempt), data in votes_by_key.items():
                votes = data["votes"]
                approvals = sum(1 for v in votes.values() if v)
                if approvals > len(votes) // 2:
                    if data["proposed_team"]:
                        approved_teams_by_round[round_num] = data["proposed_team"]
            
            for round_num in sorted(quest_votes_by_round.keys()):
                votes = quest_votes_by_round[round_num]
                fail_votes = sum(1 for v in votes.values() if not v)
                success = fail_votes == 0
                team_members = approved_teams_by_round.get(round_num, list(votes.keys()))
                quest_results.append(StateQuestResult(
                    round=round_num,
                    team_size=len(votes),
                    success=success,
                    fail_votes=fail_votes,
                    team_members=team_members,
                    quest_votes=votes,
                ))
        
        # Determine current state
        current_round = len(quest_results) + 1 if quest_results else 1
        
        # Calculate current leader
        total_leader_rotations = 0
        for vr in vote_history:
            total_leader_rotations += 1
        current_leader = total_leader_rotations % game["player_count"]
        
        # Determine vote attempt for current round
        current_round_votes = [v for v in vote_history if v.round == current_round]
        vote_attempt = len(current_round_votes) + 1
        
        # Determine phase
        try:
            phase = StateGamePhase(game.get("phase", "role_assignment"))
        except ValueError:
            phase = StateGamePhase.ROLE_ASSIGNMENT
        
        try:
            status = StateGameStatus(game.get("status", "waiting"))
        except ValueError:
            status = StateGameStatus.WAITING
        
        # Determine winner
        winner = None
        if game.get("winner"):
            try:
                winner = Team(game["winner"])
            except ValueError:
                pass
        
        # Create GameState
        state = GameState(
            id=game["_id"],
            status=status,
            phase=phase,
            player_count=game["player_count"],
            players=players,
            current_round=current_round,
            current_leader=current_leader,
            vote_attempt=vote_attempt,
            quest_results=quest_results,
            vote_history=vote_history,
            discussion_history=discussion_history,
            proposed_team=[],
            current_votes={},
            current_quest_votes={},
            winner=winner,
            created_at=game["created_at"].isoformat() if game.get("created_at") else datetime.now().isoformat(),
            finished_at=game["finished_at"].isoformat() if game.get("finished_at") else None,
        )
        
        return state
