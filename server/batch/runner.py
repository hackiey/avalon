"""Batch game runner for RL training data collection.

This module provides a headless game runner that can run multiple games
using GameRollout + LLMPlayerManager, saving all LLM decisions and game data
to the database for later export and RL training.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field

from tqdm import tqdm

from game.rollout import GameRollout
from game.roles import is_evil
from server.llm.base import ToolCallParseError


@dataclass
class BatchConfig:
    """Configuration for batch game running."""

    num_games: int = 100
    player_count: int = 5

    # Model configuration for players
    # Each entry is (model_name, provider)
    models: List[tuple] = field(default_factory=list)

    # If True, rotate models among players for variety
    rotate_models: bool = True

    # Number of games to run in parallel
    parallel: int = 1

    # Callback for progress updates
    progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None

    # Tag for batch identification
    batch_tag: Optional[str] = None

    # Skip MongoDB, store everything in memory (for servers without MongoDB)
    no_mongo: bool = False

    # Use incremental (multi-turn) context instead of full-state prompts
    use_incremental_context: bool = False

    # Role-based model assignment: {role_value: (model_name, provider)}
    # When set, overrides model assignment after role allocation.
    # Roles not in the map use the first entry in `models` as default.
    role_model_map: Optional[Dict[str, Tuple[str, str]]] = None


@dataclass
class BatchResult:
    """Result of a batch run."""

    total_games: int = 0
    completed_games: int = 0
    failed_games: int = 0
    good_wins: int = 0
    evil_wins: int = 0
    game_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    batch_id: str = ""
    started_at: str = ""
    finished_at: str = ""


class BatchGameRunner:
    """Headless game runner for batch processing.

    Uses GameRollout for game lifecycle and LLMPlayerManager for LLM calls,
    replacing the previous GameManager + NullSocketIO approach.
    """

    def __init__(self, config: BatchConfig):
        self.config = config
        self._stop_requested = False

    def stop(self):
        """Request to stop the batch run."""
        self._stop_requested = True

    async def run(self) -> BatchResult:
        """Run a batch of games and return results."""
        if self.config.no_mongo:
            from server.storage.memory_repository import InMemoryRepository
            self._repo = InMemoryRepository()
            self._memory_repo = self._repo  # Keep reference for export
        else:
            from server.models.database import init_db
            from server.storage.repository import GameRepository
            await init_db()
            self._repo = GameRepository()
            self._memory_repo = None

        batch_id = str(uuid.uuid4())[:8]
        result = BatchResult(
            total_games=self.config.num_games,
            batch_id=batch_id,
            started_at=datetime.now().isoformat(),
        )

        self._result = result
        self._result_lock = asyncio.Lock()
        self._completed_count = 0

        print(f"\n{'='*60}")
        print(f"Starting batch run: {batch_id}")
        print(f"Total games: {self.config.num_games}")
        print(f"Player count: {self.config.player_count}")
        print(f"Parallel: {self.config.parallel}")
        print(f"Models: {self.config.models}")
        print(f"{'='*60}\n")

        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.config.parallel)

        pbar = tqdm(total=self.config.num_games, desc="Rollout", unit="game", dynamic_ncols=True)

        async def run_with_semaphore(game_index: int):
            async with semaphore:
                if self._stop_requested:
                    return
                await self._run_game_and_record(game_index, batch_id, pbar)

        # Create all tasks
        tasks = [run_with_semaphore(i) for i in range(self.config.num_games)]

        # Run all tasks concurrently (semaphore limits actual parallelism)
        await asyncio.gather(*tasks, return_exceptions=True)
        pbar.close()

        result.finished_at = datetime.now().isoformat()

        # Print summary
        print(f"\n{'='*60}")
        print(f"Batch run completed: {batch_id}")
        print(f"Completed: {result.completed_games}/{result.total_games}")
        print(f"Failed: {result.failed_games}")
        print(f"Good wins: {result.good_wins} ({result.good_wins/max(1,result.completed_games)*100:.1f}%)")
        print(f"Evil wins: {result.evil_wins} ({result.evil_wins/max(1,result.completed_games)*100:.1f}%)")
        print(f"{'='*60}\n")

        return result

    async def _run_game_and_record(self, game_index: int, batch_id: str, pbar: tqdm = None):
        """Run a single game and record the result."""
        try:
            game_id, winner = await self._run_single_game(game_index, batch_id)

            # Thread-safe update of results
            async with self._result_lock:
                if winner == "good":
                    self._result.good_wins += 1
                elif winner == "evil":
                    self._result.evil_wins += 1

                self._result.completed_games += 1
                self._result.game_ids.append(game_id)
                self._completed_count += 1
                completed = self._completed_count
                good_wins = self._result.good_wins
                evil_wins = self._result.evil_wins

            if self.config.progress_callback:
                self.config.progress_callback(completed, self.config.num_games, game_id)
            elif pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    good=f"{good_wins}({good_wins/completed*100:.0f}%)" if completed else "0",
                    evil=f"{evil_wins}({evil_wins/completed*100:.0f}%)" if completed else "0",
                    last=winner or "?",
                )

        except Exception as e:
            async with self._result_lock:
                self._result.failed_games += 1
                self._completed_count += 1
                error_msg = f"Game {game_index+1} failed: {str(e)}"
                self._result.errors.append(error_msg)
            if pbar is not None:
                pbar.update(1)
                pbar.write(f"ERROR: {error_msg}")
            else:
                print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()

    async def _run_single_game(self, game_index: int, batch_id: str) -> Tuple[str, Optional[str]]:
        """Run a single game using GameRollout + LLMPlayerManager.

        Returns:
            (game_id, winner) - winner is "good", "evil", or None
        """
        from server.models.schemas import GameCreate, PlayerConfig

        # 1. Create player configs (with model assignments)
        player_configs = self._create_player_configs(game_index)

        # 2. Create rollout with player model info
        rollout_player_configs = [
            {
                "seat": pc.seat,
                "name": pc.name,
                "is_human": pc.is_human,
                "model": pc.model,
                "provider": pc.provider,
            }
            for pc in player_configs
        ]

        rollout = GameRollout(player_count=self.config.player_count)
        first_seat, first_phase = rollout.create_and_start(rollout_player_configs)

        # 2.5 Role-based model override (for controlled A/B evaluation)
        if self.config.role_model_map:
            default_model, default_provider = self.config.models[0]
            for player in rollout.state.players:
                role_key = player.role.value if player.role else None
                if role_key and role_key in self.config.role_model_map:
                    player.model_name, player.provider = self.config.role_model_map[role_key]
                else:
                    player.model_name = default_model
                    player.provider = default_provider

        # 3. Create LLM player manager
        if self.config.use_incremental_context:
            from server.llm.player_v2 import LLMPlayerManagerV2
            llm_manager = LLMPlayerManagerV2()
        else:
            from server.llm.player import LLMPlayerManager
            llm_manager = LLMPlayerManager()
        for player in rollout.state.players:
            if not player.is_human:
                llm_manager.add_player(player)
        await llm_manager.initialize_all(rollout.state)

        # 4. Save game to storage (database or memory)
        game_config = GameCreate(
            player_count=self.config.player_count,
            players=player_configs,
        )
        game_response = await self._repo.create_game(game_config)
        game_id = game_response.id
        rollout.state.id = game_id

        # Mark as batch game
        if self.config.no_mongo:
            self._repo.set_batch_metadata(game_id, batch_id, self.config.batch_tag)
        else:
            from server.models.database import get_db
            db = get_db()
            await db.games.update_one(
                {"_id": game_id},
                {"$set": {
                    "source": "batch",
                    "batch_id": batch_id,
                    "batch_tag": self.config.batch_tag,
                }}
            )

        # 5. Game loop
        seat, phase = first_seat, first_phase
        error_terminated = False

        while not rollout.is_finished:
            llm_player = llm_manager.get_player(seat)
            if not llm_player:
                break

            try:
                await self._handle_turn(rollout, llm_player, seat, phase, game_id)
            except ToolCallParseError as e:
                print(
                    f"[WARN] Game {game_id}: Malformed tool call at "
                    f"round {rollout.state.current_round}, seat {seat}, "
                    f"phase {phase}. Stopping game."
                )
                error_terminated = True
                break

            if rollout.is_finished:
                break

            # Advance to next decision point
            next_result = rollout.advance_to_next_decision()
            seat, phase = next_result
            if seat is None:
                break

        # 6. Save final state to database
        await self._repo.update_game_state(rollout.state)

        winner = rollout.winner.value if rollout.winner else None
        if error_terminated:
            winner = "error"
        return game_id, winner

    async def _handle_turn(
        self,
        rollout: GameRollout,
        llm_player,
        seat: int,
        phase: str,
        game_id: str,
    ):
        """Handle a single turn: call LLM, execute action, save to DB.

        Raises ToolCallParseError if model generates malformed tool calls,
        after saving the failed action to the database.
        """
        state = rollout.state

        try:
            await self._handle_turn_inner(rollout, llm_player, seat, phase, game_id)
        except ToolCallParseError as e:
            await self._repo.save_action(
                game_id=game_id,
                round_num=state.current_round,
                action_type=phase,
                player_seat=seat,
                llm_input=e.llm_input,
                llm_output={
                    "error": "tool_call_parse_error",
                    "raw_content": e.raw_content,
                },
            )
            raise

    async def _handle_turn_inner(
        self,
        rollout: GameRollout,
        llm_player,
        seat: int,
        phase: str,
        game_id: str,
    ):
        """Inner turn handler (separated for ToolCallParseError wrapping)."""
        state = rollout.state

        if phase in ("leader_discussion", "discussion"):
            if phase == "leader_discussion":
                call_result = await llm_player.discuss_as_leader(state)
            else:
                call_result = await llm_player.discuss(state)

            content = call_result.result
            rollout.execute_structured_action(seat, phase, content=content)

            await self._repo.save_action(
                game_id=game_id,
                round_num=state.current_round,
                action_type="discussion",
                player_seat=seat,
                content=content,
                vote_attempt=state.vote_attempt,
                llm_input=call_result.llm_input,
                llm_output=call_result.llm_output,
            )

        elif phase == "team_selection":
            team, speech, llm_input, llm_output = await llm_player.select_team_final(state)

            rollout.execute_structured_action(
                seat, phase, team=team, content=speech,
            )

            # Save discussion (leader summary speech)
            if speech:
                await self._repo.save_action(
                    game_id=game_id,
                    round_num=state.current_round,
                    action_type="discussion",
                    player_seat=seat,
                    content=speech,
                    vote_attempt=state.vote_attempt,
                    llm_input=llm_input,
                    llm_output=llm_output,
                )

            # Save team selection
            await self._repo.save_action(
                game_id=game_id,
                round_num=state.current_round,
                action_type="team_selection",
                player_seat=seat,
                proposed_team=team,
                vote_attempt=state.vote_attempt,
            )

        elif phase == "team_vote":
            vote_result = await llm_player.vote(state)
            approve = vote_result.result

            rollout.execute_structured_action(seat, phase, approve=approve)

            await self._repo.save_action(
                game_id=game_id,
                round_num=state.current_round,
                action_type="team_vote",
                player_seat=seat,
                vote=approve,
                vote_attempt=state.vote_attempt,
                proposed_team=state.proposed_team,
                llm_input=vote_result.llm_input,
                llm_output=vote_result.llm_output,
            )

        elif phase == "quest_execution":
            player = state.get_player(seat)

            # Good players always succeed (no LLM call needed)
            if player and not is_evil(player.role):
                rollout.execute_structured_action(seat, phase, success=True)
                await self._repo.save_action(
                    game_id=game_id,
                    round_num=state.current_round,
                    action_type="quest_vote",
                    player_seat=seat,
                    vote=True,
                )
            else:
                quest_result = await llm_player.execute_quest(state)
                success = quest_result.result

                rollout.execute_structured_action(seat, phase, success=success)

                await self._repo.save_action(
                    game_id=game_id,
                    round_num=state.current_round,
                    action_type="quest_vote",
                    player_seat=seat,
                    vote=success,
                    llm_input=quest_result.llm_input,
                    llm_output=quest_result.llm_output,
                )

        elif phase == "assassination_discussion":
            call_result = await llm_player.discuss_assassination(state)
            content = call_result.result

            rollout.execute_structured_action(seat, phase, content=content)

            await self._repo.save_action(
                game_id=game_id,
                round_num=state.current_round,
                action_type="assassination_discussion",
                player_seat=seat,
                content=content,
                llm_input=call_result.llm_input,
                llm_output=call_result.llm_output,
            )

        elif phase == "assassination":
            assassinate_result = await llm_player.assassinate(state)
            target = assassinate_result.result

            rollout.execute_structured_action(seat, phase, target=target)

            await self._repo.save_action(
                game_id=game_id,
                round_num=state.current_round,
                action_type="assassination",
                player_seat=seat,
                target_seat=target,
                llm_input=assassinate_result.llm_input,
                llm_output=assassinate_result.llm_output,
            )

    def _create_player_configs(self, game_index: int) -> List:
        """Create player configurations for a game."""
        from server.models.schemas import PlayerConfig

        if not self.config.models:
            raise ValueError("No models configured. Please provide at least one model in BatchConfig.models")

        configs = []

        for seat in range(self.config.player_count):
            if self.config.rotate_models:
                # Rotate models based on game index and seat
                model_idx = (game_index + seat) % len(self.config.models)
            else:
                # Use seat-based model assignment
                model_idx = seat % len(self.config.models)

            model_name, provider = self.config.models[model_idx]

            configs.append(PlayerConfig(
                seat=seat,
                name=f"Player{seat + 1}",
                is_human=False,  # All players are LLM in batch mode
                model=model_name,
                provider=provider,
            ))

        return configs
