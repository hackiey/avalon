"""Training data exporter for RL training.

Exports complete game trajectories with all LLM decisions,
including inputs, outputs, and reasoning.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from game.roles import is_evil_role
from server.storage.repository import GameRepository
from server.models.database import get_db


@dataclass
class LLMDecision:
    """A single LLM decision in the game."""
    
    # Sequence info
    seq_num: int  # Order in the game (0-indexed)
    
    # Player info
    player_seat: int
    player_name: str
    player_role: str
    player_team: str  # "good" or "evil"
    model_name: str
    provider: str
    
    # Game state at decision time
    round_num: int
    vote_attempt: int
    
    # Decision type
    action_type: str  # discussion, team_vote, quest_vote, team_selection, assassination, assassination_discussion
    
    # LLM interaction - the core training data
    llm_input: Dict[str, Any]   # Messages sent to LLM
    llm_output: Dict[str, Any]  # Raw LLM response including reasoning_content
    
    # Extracted decision result
    content: Optional[str] = None       # For discussion
    vote: Optional[bool] = None         # For votes
    team: Optional[List[int]] = None    # For team selection
    target: Optional[int] = None        # For assassination
    
    # Reasoning (extracted from llm_output for convenience)
    reasoning: Optional[str] = None
    
    timestamp: str = ""


@dataclass
class GameTrajectory:
    """Complete trajectory of a game for training."""
    
    game_id: str
    batch_id: Optional[str]
    
    # Game setup
    player_count: int
    players: List[Dict[str, Any]]  # All players with roles, models
    
    # Game result
    winner: str  # "good" or "evil"
    good_won: bool
    evil_won: bool
    merlin_assassinated: bool
    
    # All decisions in execution order
    decisions: List[LLMDecision] = field(default_factory=list)
    
    # Quest results
    quest_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Vote history 
    vote_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    total_rounds: int = 0
    total_decisions: int = 0
    created_at: str = ""
    finished_at: str = ""


@dataclass
class ExportStats:
    """Statistics about the export."""
    total_games: int = 0
    total_decisions: int = 0
    good_wins: int = 0
    evil_wins: int = 0
    decisions_by_type: Dict[str, int] = field(default_factory=dict)


class TrainingDataExporter:
    """Exports game trajectories for RL training."""
    
    def __init__(self):
        self.repo = GameRepository()
    
    async def export_trajectories(
        self,
        output_path: str,
        batch_id: Optional[str] = None,
        tag: Optional[str] = None,
        game_ids: Optional[List[str]] = None,
        include_web: bool = False,
    ) -> ExportStats:
        """Export complete game trajectories from MongoDB.
        
        Each trajectory contains:
        - Game setup (players, roles, models)
        - All LLM decisions in execution order
        - Each decision includes: llm_input, llm_output, reasoning
        - Game outcome (for reward calculation)
        
        Args:
            output_path: Path to save the JSONL file
            batch_id: Filter by batch ID
            tag: Filter by experiment tag (e.g., "experiment_v1")
            game_ids: Specific game IDs to export
            include_web: If True, include web UI games (default: False, only batch data)
            
        Returns:
            ExportStats with export summary
        """
        from server.models.database import init_db
        await init_db()
        
        stats = ExportStats()
        
        # Get finished games (default: only batch rollout data)
        games = await self._get_games(batch_id, tag, game_ids, include_web)
        print(f"Found {len(games)} completed games to export")
        
        # Build and write trajectories
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for game in games:
                trajectory = await self._build_trajectory(game)
                self._write_trajectory(f, trajectory, stats)
        
        self._print_stats(stats, output_path)
        return stats

    async def export_from_memory(
        self,
        memory_repo,
        output_path: str,
        batch_id: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> ExportStats:
        """Export game trajectories directly from an InMemoryRepository.

        Used when MongoDB is not available (--no-mongo mode).

        Args:
            memory_repo: An InMemoryRepository instance with game data
            output_path: Path to save the JSONL file
            batch_id: Filter by batch ID
            tag: Filter by experiment tag

        Returns:
            ExportStats with export summary
        """
        stats = ExportStats()

        games = memory_repo.get_finished_games(batch_id=batch_id, tag=tag)
        print(f"Found {len(games)} completed games to export (from memory)")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for game in games:
                game_id = game["_id"]
                actions = memory_repo.get_game_actions(game_id)
                trajectory = self._build_trajectory_from_data(game, actions)
                self._write_trajectory(f, trajectory, stats)

        self._print_stats(stats, output_path)
        return stats

    @staticmethod
    def _write_trajectory(f, trajectory: GameTrajectory, stats: ExportStats):
        """Write a single trajectory to file and update stats."""
        stats.total_games += 1
        stats.total_decisions += len(trajectory.decisions)
        if trajectory.good_won:
            stats.good_wins += 1
        else:
            stats.evil_wins += 1

        for d in trajectory.decisions:
            stats.decisions_by_type[d.action_type] = \
                stats.decisions_by_type.get(d.action_type, 0) + 1

        record = asdict(trajectory)
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _print_stats(stats: ExportStats, output_path: str):
        """Print export summary."""
        print(f"\nExport completed:")
        print(f"  - Total games: {stats.total_games}")
        print(f"  - Total decisions: {stats.total_decisions}")
        print(f"  - Good wins: {stats.good_wins} ({stats.good_wins/max(1,stats.total_games)*100:.1f}%)")
        print(f"  - Evil wins: {stats.evil_wins} ({stats.evil_wins/max(1,stats.total_games)*100:.1f}%)")
        print(f"  - Decisions by type: {stats.decisions_by_type}")
        print(f"  - Output: {output_path}")

    async def _get_games(
        self,
        batch_id: Optional[str],
        tag: Optional[str],
        game_ids: Optional[List[str]],
        include_web: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get finished games matching criteria.
        
        By default, only returns batch rollout games (source="batch").
        Web UI games are excluded to avoid polluting training data.
        """
        db = get_db()
        
        query: Dict[str, Any] = {"status": "finished"}
        
        # Only include batch rollout data by default
        if not include_web:
            query["source"] = "batch"
        
        if batch_id:
            query["batch_id"] = batch_id
        
        if tag:
            query["batch_tag"] = tag
        
        if game_ids:
            query["_id"] = {"$in": game_ids}
        
        cursor = db.games.find(query).sort("created_at", 1)
        return await cursor.to_list(None)
    
    async def _build_trajectory(self, game: Dict[str, Any]) -> GameTrajectory:
        """Build a complete game trajectory from database."""
        db = get_db()
        game_id = game["_id"]
        
        # Get all actions in order
        actions = await db.actions.find(
            {"game_id": game_id}
        ).sort("timestamp", 1).to_list(None)

        return self._build_trajectory_from_data(game, actions)

    def _build_trajectory_from_data(
        self, game: Dict[str, Any], actions: List[Dict[str, Any]]
    ) -> GameTrajectory:
        """Build a complete game trajectory from game dict and action list.

        This is the shared core logic used by both MongoDB and in-memory export.
        """
        game_id = game["_id"]
        players = game.get("players", [])
        player_map = {p["seat"]: p for p in players}
        winner = game.get("winner", "")

        # Extract LLM decisions
        decisions = []
        seq_num = 0

        for action in actions:
            if not action.get("llm_input"):
                continue

            seat = action.get("player_seat")
            player = player_map.get(seat, {})
            role = player.get("role", "unknown")
            team = self._get_team(role)
            llm_output = action.get("llm_output", {})

            decision = LLMDecision(
                seq_num=seq_num,
                player_seat=seat,
                player_name=player.get("name", f"Player{seat+1}"),
                player_role=role,
                player_team=team,
                model_name=player.get("model_name", ""),
                provider=player.get("provider", ""),
                round_num=action.get("round_num", 1),
                vote_attempt=action.get("vote_attempt", 1),
                action_type=action.get("action_type", ""),
                llm_input=action.get("llm_input", {}),
                llm_output=llm_output,
                content=action.get("content"),
                vote=action.get("vote"),
                team=action.get("proposed_team"),
                target=action.get("target_seat"),
                reasoning=llm_output.get("reasoning_content"),
                timestamp=action["timestamp"].isoformat() if action.get("timestamp") else "",
            )
            decisions.append(decision)
            seq_num += 1

        # Build vote history from actions
        vote_actions = [a for a in actions if a.get("action_type") == "team_vote"]
        votes_by_round: Dict[tuple, Dict[int, bool]] = {}
        for action in vote_actions:
            key = (action.get("round_num"), action.get("vote_attempt") or 1)
            if key not in votes_by_round:
                votes_by_round[key] = {}
            if action.get("vote") is not None:
                votes_by_round[key][action.get("player_seat")] = action.get("vote")

        vote_history = []
        for (round_num, attempt), votes in votes_by_round.items():
            approvals = sum(1 for v in votes.values() if v)
            vote_history.append({
                "round": round_num,
                "attempt": attempt,
                "votes": votes,
                "approved": approvals > len(votes) // 2,
            })

        # Build quest results from actions
        quest_vote_actions = [a for a in actions if a.get("action_type") == "quest_vote"]
        quest_votes_by_round: Dict[int, Dict[int, bool]] = {}
        for action in quest_vote_actions:
            round_num = action.get("round_num")
            if round_num not in quest_votes_by_round:
                quest_votes_by_round[round_num] = {}
            if action.get("vote") is not None:
                quest_votes_by_round[round_num][action.get("player_seat")] = action.get("vote")

        quest_results = []
        rounds = game.get("rounds", [])
        if rounds:
            for r in sorted(rounds, key=lambda r: r["round_num"]):
                team_members = r.get("team_members") if isinstance(r.get("team_members"), list) else []
                quest_results.append({
                    "round": r["round_num"],
                    "team_size": len(team_members) if team_members else r.get("team_members") or 0,
                    "success": r.get("success"),
                    "fail_votes": r.get("fail_votes", 0),
                    "team_members": team_members,
                    "quest_votes": quest_votes_by_round.get(r["round_num"], {}),
                })
        elif quest_votes_by_round:
            # Fallback: rebuild from quest_vote actions
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
                    "fail_votes": fail_votes,
                    "team_members": team_members,
                    "quest_votes": votes,
                })

        # Check if Merlin was assassinated
        merlin_assassinated = False
        if winner == "evil":
            for action in actions:
                if action.get("action_type") == "assassination":
                    target_seat = action.get("target_seat")
                    target_player = player_map.get(target_seat, {})
                    if target_player.get("role") == "merlin":
                        merlin_assassinated = True
                    break

        trajectory = GameTrajectory(
            game_id=game_id,
            batch_id=game.get("batch_id"),
            player_count=game.get("player_count", 5),
            players=players,
            winner=winner,
            good_won=(winner == "good"),
            evil_won=(winner == "evil"),
            merlin_assassinated=merlin_assassinated,
            decisions=decisions,
            quest_results=quest_results,
            vote_history=vote_history,
            total_rounds=len(quest_results),
            total_decisions=len(decisions),
            created_at=game["created_at"].isoformat() if game.get("created_at") else "",
            finished_at=game["finished_at"].isoformat() if game.get("finished_at") else "",
        )

        return trajectory

    def _get_team(self, role: str) -> str:
        """Get team from role."""
        return "evil" if is_evil_role(role) else "good"


async def list_batches() -> List[Dict[str, Any]]:
    """List all batch runs with summary stats."""
    from server.models.database import init_db, get_db
    
    await init_db()
    db = get_db()
    
    # Aggregate by batch_id
    pipeline = [
        {"$match": {"batch_id": {"$exists": True, "$ne": None}}},
        {"$group": {
            "_id": "$batch_id",
            "total_games": {"$sum": 1},
            "completed": {"$sum": {"$cond": [{"$eq": ["$status", "finished"]}, 1, 0]}},
            "good_wins": {"$sum": {"$cond": [{"$eq": ["$winner", "good"]}, 1, 0]}},
            "evil_wins": {"$sum": {"$cond": [{"$eq": ["$winner", "evil"]}, 1, 0]}},
            "first_game": {"$min": "$created_at"},
            "last_game": {"$max": "$finished_at"},
        }},
        {"$sort": {"first_game": -1}},
    ]
    
    results = await db.games.aggregate(pipeline).to_list(None)
    
    batches = []
    for r in results:
        batches.append({
            "batch_id": r["_id"],
            "total_games": r["total_games"],
            "completed": r["completed"],
            "good_wins": r["good_wins"],
            "evil_wins": r["evil_wins"],
            "good_win_rate": f"{r['good_wins']/max(1,r['completed'])*100:.1f}%",
            "started_at": r["first_game"].isoformat() if r.get("first_game") else None,
            "finished_at": r["last_game"].isoformat() if r.get("last_game") else None,
        })
    
    return batches
