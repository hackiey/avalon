"""Batch game runner for RL training data collection.

This module provides a headless game runner that can run multiple games
without Socket.IO, saving all LLM decisions and game data to the database
for later export and RL training.
"""

import asyncio
import uuid
from datetime import datetime
from typing import List, Optional, Callable
from dataclasses import dataclass, field


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


class NullSocketIO:
    """A no-op Socket.IO adapter for headless game running.
    
    This allows us to reuse GameManager without actually sending
    any Socket.IO events.
    """
    
    async def emit(self, event: str, data: dict, room: str = None):
        """No-op emit - just ignore all socket events."""
        pass


class BatchGameRunner:
    """Headless game runner for batch processing.
    
    This runner reuses the existing GameManager logic but with a
    no-op Socket.IO adapter, avoiding any code duplication.
    """
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self._stop_requested = False
        self._null_sio = NullSocketIO()
    
    def stop(self):
        """Request to stop the batch run."""
        self._stop_requested = True
    
    async def run(self) -> BatchResult:
        """Run a batch of games and return results."""
        from server.models.database import init_db
        from server.storage.repository import GameRepository
        
        # Ensure database is initialized
        await init_db()
        
        batch_id = str(uuid.uuid4())[:8]
        result = BatchResult(
            total_games=self.config.num_games,
            batch_id=batch_id,
            started_at=datetime.now().isoformat(),
        )
        
        self._repo = GameRepository()
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
        
        async def run_with_semaphore(game_index: int):
            async with semaphore:
                if self._stop_requested:
                    return
                await self._run_game_and_record(game_index, batch_id)
        
        # Create all tasks
        tasks = [run_with_semaphore(i) for i in range(self.config.num_games)]
        
        # Run all tasks concurrently (semaphore limits actual parallelism)
        await asyncio.gather(*tasks, return_exceptions=True)
        
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
    
    async def _run_game_and_record(self, game_index: int, batch_id: str):
        """Run a single game and record the result."""
        try:
            game_id = await self._run_single_game(game_index, batch_id)
            
            # Get game result
            game = await self._repo.get_game(game_id, reveal_all=True)
            winner = game.get("winner") if game else None
            
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
            
            # Progress update
            if self.config.progress_callback:
                self.config.progress_callback(completed, self.config.num_games, game_id)
            else:
                print(f"Game {completed}/{self.config.num_games} completed: {game_id} (Winner: {winner})")
            
        except Exception as e:
            async with self._result_lock:
                self._result.failed_games += 1
                self._completed_count += 1
                error_msg = f"Game {game_index+1} failed: {str(e)}"
                self._result.errors.append(error_msg)
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
    
    async def _run_single_game(self, game_index: int, batch_id: str) -> str:
        """Run a single game and return the game ID.
        
        This method reuses the existing GameManager, just with a no-op Socket.IO.
        """
        from server.game.manager import GameManager
        from server.models.schemas import GameCreate, PlayerConfig
        from server.models.database import get_db
        
        # Create player configurations
        player_configs = self._create_player_configs(game_index)
        
        # Create game config
        game_config = GameCreate(
            player_count=self.config.player_count,
            players=player_configs,
        )
        
        # Use GameManager to create and run the game
        # Create a fresh manager instance with headless=True to skip UI delays
        manager = GameManager(headless=True)
        
        # Create game
        state = await manager.create_game(game_config)
        game_id = state.id
        
        # Mark as batch game (distinguish from web UI games)
        db = get_db()
        await db.games.update_one(
            {"_id": game_id},
            {"$set": {
                "source": "batch",  # Mark as batch rollout data
                "batch_id": batch_id,
                "batch_tag": self.config.batch_tag,
            }}
        )
        
        # Run the game using GameManager's existing logic
        # with our no-op Socket.IO adapter
        await manager.start_game(game_id, self._null_sio)
        
        return game_id
    
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
