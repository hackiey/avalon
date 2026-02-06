"""Game manager that coordinates game engine, LLM players, and Socket.IO."""

import asyncio
from datetime import datetime
from typing import Dict, Optional, Any
import socketio

from server.game.engine import GameEngine
from server.game.state import GameState, GamePhase, GameStatus, Player
from server.game.roles import Role, is_evil
from server.llm.player import LLMPlayerManager, LLMPlayer
from server.storage.repository import GameRepository
from server.models.schemas import GameCreate


class GameManager:
    """Singleton manager for all active games."""
    
    _instance: Optional["GameManager"] = None
    
    def __init__(self, headless: bool = False):
        self.games: Dict[str, GameEngine] = {}
        self.llm_managers: Dict[str, LLMPlayerManager] = {}
        self.repo = GameRepository()
        self.headless = headless  # Skip delays for batch running
    
    @classmethod
    def get_instance(cls) -> "GameManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def create_game(self, config: GameCreate) -> GameState:
        """Create a new game."""
        # Create in database
        game_response = await self.repo.create_game(config)
        
        # Create game engine
        player_configs = [
            {
                "seat": p.seat,
                "name": p.name,
                "is_human": p.is_human,
                "model": p.model,
                "provider": p.provider,
            }
            for p in config.players
        ]
        
        engine = GameEngine.create_game(config.player_count, player_configs)
        engine.state.id = game_response.id
        
        self.games[engine.state.id] = engine
        
        # Create LLM player manager
        llm_manager = LLMPlayerManager()
        for player in engine.state.players:
            if not player.is_human:
                llm_manager.add_player(player)
        
        self.llm_managers[engine.state.id] = llm_manager
        
        return engine.state
    
    def get_game(self, game_id: str) -> Optional[GameEngine]:
        """Get a game engine by ID."""
        return self.games.get(game_id)
    
    async def restore_game(self, game_id: str) -> Optional[GameEngine]:
        """Restore a game from database.
        
        This is used to restore games after server restart.
        Only restores games that are not finished.
        
        Returns:
            The restored GameEngine, or None if game not found or already finished.
        """
        # Check if already in memory
        if game_id in self.games:
            return self.games[game_id]
        
        # Try to restore from database
        state = await self.repo.get_game_for_restore(game_id)
        if not state:
            return None
        
        # Don't restore finished games - they can be viewed but not resumed
        if state.status == GameStatus.FINISHED:
            return None
        
        # Create engine from restored state
        engine = GameEngine(state)
        self.games[game_id] = engine
        
        # Recreate LLM player manager
        llm_manager = LLMPlayerManager()
        for player in state.players:
            if not player.is_human:
                llm_manager.add_player(player)
        
        self.llm_managers[game_id] = llm_manager
        
        # Initialize LLM players with game state
        await llm_manager.initialize_all(state)
        
        return engine
    
    async def start_game(self, game_id: str, sio: socketio.AsyncServer):
        """Start a game and run the game loop."""
        engine = self.games.get(game_id)
        if not engine:
            return
        
        llm_manager = self.llm_managers.get(game_id)
        if not llm_manager:
            return
        
        # Start the game
        engine.start_game()
        
        # Initialize LLM players
        await llm_manager.initialize_all(engine.state)
        
        # Emit initial state
        await self._emit_state(game_id, sio)
        
        # Short delay for night phase (skip in headless mode)
        if not self.headless:
            await asyncio.sleep(2)
        
        # Move to discussion phase (in new flow, discussion happens before team selection)
        engine.proceed_to_discussion()
        await self._emit_state(game_id, sio)
        
        # Update database
        await self.repo.update_game_state(engine.state)
        
        # Start game loop
        await self._run_game_loop(game_id, sio)
    
    async def _run_game_loop(self, game_id: str, sio: socketio.AsyncServer):
        """Main game loop."""
        engine = self.games.get(game_id)
        llm_manager = self.llm_managers.get(game_id)
        
        if not engine or not llm_manager:
            return
        
        while engine.state.status == GameStatus.IN_PROGRESS:
            # Check if waiting for human
            if engine.state.waiting_for_human:
                # Wait for human action (in headless mode, this shouldn't happen)
                if not self.headless:
                    await asyncio.sleep(0.5)
                continue
            
            phase = engine.state.phase
            
            if phase == GamePhase.TEAM_SELECTION:
                await self._handle_team_selection(game_id, sio)
            
            elif phase == GamePhase.DISCUSSION:
                await self._handle_discussion(game_id, sio)
            
            elif phase == GamePhase.TEAM_VOTE:
                await self._handle_team_vote(game_id, sio)
            
            elif phase == GamePhase.QUEST_EXECUTION:
                await self._handle_quest(game_id, sio)
            
            elif phase == GamePhase.ASSASSINATION_DISCUSSION:
                await self._handle_assassination_discussion(game_id, sio)
            
            elif phase == GamePhase.ASSASSINATION:
                await self._handle_assassination(game_id, sio)
            
            elif phase == GamePhase.GAME_OVER:
                break
            
            if not self.headless:
                await asyncio.sleep(0.1)
        
        # Game ended
        await self.repo.update_game_state(engine.state)
        await self._emit_state(game_id, sio)
    
    async def _handle_team_selection(self, game_id: str, sio: socketio.AsyncServer):
        """Handle team selection phase.
        
        This phase now happens AFTER discussion.
        The leader has already spoken during discussion, 
        now they just need to make the final team choice.
        """
        engine = self.games.get(game_id)
        llm_manager = self.llm_managers.get(game_id)
        
        if not engine or not llm_manager:
            return
        
        leader = engine.state.get_leader()
        
        if leader.is_human:
            engine.state.waiting_for_human = True
            engine.state.human_action_type = "team_selection"
            await self._emit_state(game_id, sio)
            return
        
        # LLM selects team with summary speech
        llm_player = llm_manager.get_player(leader.seat)
        if llm_player:
            team, speech, llm_input, llm_output = await llm_player.select_team_final(engine.state)
            
            # Add leader's summary speech to discussion history before selecting team
            if speech:
                engine.add_discussion(leader.seat, speech)
                
                # Save discussion to database
                await self.repo.save_action(
                    game_id=game_id,
                    round_num=engine.state.current_round,
                    action_type="discussion",
                    player_seat=leader.seat,
                    content=speech,
                    vote_attempt=engine.state.vote_attempt,
                    llm_input=llm_input,
                    llm_output=llm_output,
                )
                
                # Emit discussion message
                await sio.emit("game:discussion", {
                    "game_id": game_id,
                    "seat": leader.seat,
                    "player_name": leader.name,
                    "content": speech,
                    "round": engine.state.current_round,
                    "attempt": engine.state.vote_attempt,
                    "timestamp": datetime.now().isoformat(),
                }, room=f"game:{game_id}")
            
            engine.select_team(team)
            
            # Save team selection action to database
            await self.repo.save_action(
                game_id=game_id,
                round_num=engine.state.current_round,
                action_type="team_selection",
                player_seat=leader.seat,
                proposed_team=team,
                vote_attempt=engine.state.vote_attempt,
            )
            
            await self._emit_state(game_id, sio)
    
    async def _handle_discussion(self, game_id: str, sio: socketio.AsyncServer):
        """Handle discussion phase.
        
        In the new flow:
        1. Leader speaks first (proposes a team configuration)
        2. Other players discuss in order
        3. After all players speak, move to team selection (leader makes final choice)
        """
        engine = self.games.get(game_id)
        llm_manager = self.llm_managers.get(game_id)
        
        if not engine or not llm_manager:
            return
        
        while not engine.state.discussion_complete:
            seat = engine.next_discussion_speaker()
            if seat is None:
                break
            
            player = engine.state.get_player(seat)
            if not player:
                continue
            
            is_leader = seat == engine.state.current_leader
            
            if player.is_human:
                engine.state.waiting_for_human = True
                engine.state.human_action_type = "leader_discussion" if is_leader else "discussion"
                await self._emit_state(game_id, sio)
                return
            
            # LLM discusses
            llm_player = llm_manager.get_player(seat)
            if llm_player:
                if is_leader:
                    # Leader speaks first, proposes a team
                    call_result = await llm_player.discuss_as_leader(engine.state)
                else:
                    # Regular discussion
                    call_result = await llm_player.discuss(engine.state)
                
                content = call_result.result
                engine.add_discussion(seat, content)
                
                # Save to database with LLM call details
                await self.repo.save_action(
                    game_id=game_id,
                    round_num=engine.state.current_round,
                    action_type="discussion",
                    player_seat=seat,
                    content=content,
                    vote_attempt=engine.state.vote_attempt,
                    llm_input=call_result.llm_input,
                    llm_output=call_result.llm_output,
                )
                
                # Emit discussion message
                await sio.emit("game:discussion", {
                    "game_id": game_id,
                    "seat": seat,
                    "player_name": player.name,
                    "content": content,
                    "round": engine.state.current_round,
                    "attempt": engine.state.vote_attempt,
                    "timestamp": datetime.now().isoformat(),
                }, room=f"game:{game_id}")
                
                if not self.headless:
                    await asyncio.sleep(1)  # Pacing
        
        # Discussion complete, move to team selection (leader makes final choice)
        engine.proceed_to_team_selection()
        await self._emit_state(game_id, sio)
    
    async def _handle_team_vote(self, game_id: str, sio: socketio.AsyncServer):
        """Handle team voting phase."""
        engine = self.games.get(game_id)
        llm_manager = self.llm_managers.get(game_id)
        
        if not engine or not llm_manager:
            return
        
        # Collect votes from all players
        for player in engine.state.players:
            if player.seat in engine.state.current_votes:
                continue  # Already voted
            
            if player.is_human:
                engine.state.waiting_for_human = True
                engine.state.human_action_type = "vote"
                await self._emit_state(game_id, sio)
                return
            
            # LLM votes
            llm_player = llm_manager.get_player(player.seat)
            if llm_player:
                vote_result = await llm_player.vote(engine.state)
                vote = vote_result.result
                engine.cast_vote(player.seat, vote)
                
                # Save to database with LLM details
                await self.repo.save_action(
                    game_id=game_id,
                    round_num=engine.state.current_round,
                    action_type="team_vote",
                    player_seat=player.seat,
                    vote=vote,
                    vote_attempt=engine.state.vote_attempt,
                    proposed_team=engine.state.proposed_team,
                    llm_input=vote_result.llm_input,
                    llm_output=vote_result.llm_output,
                )
        
        # All votes cast
        if engine.all_votes_cast():
            approved = engine.resolve_vote()
            
            await sio.emit("game:vote_result", {
                "game_id": game_id,
                "votes": engine.state.vote_history[-1].votes if engine.state.vote_history else {},
                "approved": approved,
            }, room=f"game:{game_id}")
            
            await self._emit_state(game_id, sio)
            if not self.headless:
                await asyncio.sleep(2)  # Let players see result
    
    async def _handle_quest(self, game_id: str, sio: socketio.AsyncServer):
        """Handle quest execution phase."""
        engine = self.games.get(game_id)
        llm_manager = self.llm_managers.get(game_id)
        
        if not engine or not llm_manager:
            return
        
        # Collect quest votes from team members
        for seat in engine.state.proposed_team:
            if seat in engine.state.current_quest_votes:
                continue
            
            player = engine.state.get_player(seat)
            if not player:
                continue
            
            if player.is_human:
                engine.state.waiting_for_human = True
                engine.state.human_action_type = "quest"
                await self._emit_state(game_id, sio)
                return
            
            # LLM executes quest
            llm_player = llm_manager.get_player(seat)
            if llm_player:
                quest_result = await llm_player.execute_quest(engine.state)
                success = quest_result.result
                engine.cast_quest_vote(seat, success)
                
                # Save to database with LLM details
                await self.repo.save_action(
                    game_id=game_id,
                    round_num=engine.state.current_round,
                    action_type="quest_vote",
                    player_seat=seat,
                    vote=success,
                    vote_attempt=1,  # Quest votes don't have attempts
                    llm_input=quest_result.llm_input,
                    llm_output=quest_result.llm_output,
                )
        
        # All quest votes cast
        if engine.all_quest_votes_cast():
            success = engine.resolve_quest()
            
            # Save quest result to database
            if engine.state.quest_results:
                await self.repo.save_quest_result(game_id, engine.state.quest_results[-1])
            
            await sio.emit("game:quest_result", {
                "game_id": game_id,
                "round": engine.state.current_round - 1 if engine.state.phase != GamePhase.GAME_OVER else engine.state.current_round,
                "success": success,
                "fail_votes": engine.state.quest_results[-1].fail_votes if engine.state.quest_results else 0,
            }, room=f"game:{game_id}")
            
            await self._emit_state(game_id, sio)
            if not self.headless:
                await asyncio.sleep(2)
    
    async def _handle_assassination_discussion(self, game_id: str, sio: socketio.AsyncServer):
        """Handle assassination discussion phase - evil team discusses before assassination."""
        engine = self.games.get(game_id)
        llm_manager = self.llm_managers.get(game_id)
        
        if not engine or not llm_manager:
            return
        
        while not engine.state.assassination_discussion_complete:
            seat = engine.next_assassination_discussion_speaker()
            if seat is None:
                break
            
            player = engine.state.get_player(seat)
            if not player:
                continue
            
            if player.is_human:
                engine.state.waiting_for_human = True
                engine.state.human_action_type = "assassination_discussion"
                await self._emit_state(game_id, sio)
                return
            
            # LLM discusses assassination
            llm_player = llm_manager.get_player(seat)
            if llm_player:
                call_result = await llm_player.discuss_assassination(engine.state)
                content = call_result.result
                engine.add_assassination_discussion(seat, content)
                
                # Save to database
                await self.repo.save_action(
                    game_id=game_id,
                    round_num=engine.state.current_round,
                    action_type="assassination_discussion",
                    player_seat=seat,
                    content=content,
                    llm_input=call_result.llm_input,
                    llm_output=call_result.llm_output,
                )
                
                # Emit assassination discussion message
                await sio.emit("game:assassination_discussion", {
                    "game_id": game_id,
                    "seat": seat,
                    "player_name": player.name,
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                }, room=f"game:{game_id}")
                
                if not self.headless:
                    await asyncio.sleep(1)  # Pacing
        
        # Discussion complete, move to assassination phase
        engine.proceed_to_assassination()
        await self._emit_state(game_id, sio)
    
    async def _handle_assassination(self, game_id: str, sio: socketio.AsyncServer):
        """Handle assassination phase."""
        engine = self.games.get(game_id)
        llm_manager = self.llm_managers.get(game_id)
        
        if not engine or not llm_manager:
            return
        
        assassin_seat = engine.get_assassin_seat()
        if assassin_seat is None:
            return
        
        assassin = engine.state.get_player(assassin_seat)
        if not assassin:
            return
        
        if assassin.is_human:
            engine.state.waiting_for_human = True
            engine.state.human_action_type = "assassinate"
            await self._emit_state(game_id, sio)
            return
        
        # LLM assassinates
        llm_player = llm_manager.get_player(assassin_seat)
        if llm_player:
            assassinate_result = await llm_player.assassinate(engine.state)
            target = assassinate_result.result
            success = engine.assassinate(target)
            
            # Save to database with LLM details
            await self.repo.save_action(
                game_id=game_id,
                round_num=engine.state.current_round,
                action_type="assassination",
                player_seat=assassin_seat,
                target_seat=target,
                llm_input=assassinate_result.llm_input,
                llm_output=assassinate_result.llm_output,
            )
            
            target_player = engine.state.get_player(target)
            await sio.emit("game:assassination", {
                "game_id": game_id,
                "assassin": assassin_seat,
                "target": target,
                "target_name": target_player.name if target_player else "Unknown",
                "success": success,
                "target_was_merlin": target_player.role == Role.MERLIN if target_player else False,
            }, room=f"game:{game_id}")
            
            await self._emit_state(game_id, sio)
    
    async def _emit_state(self, game_id: str, sio: socketio.AsyncServer):
        """Emit current game state to all clients."""
        engine = self.games.get(game_id)
        if not engine:
            return
        
        await sio.emit("game:state", {
            "game_id": game_id,
            "state": engine.state.to_dict(),
        }, room=f"game:{game_id}")
    
    # Human action handlers
    async def handle_human_team_select(self, game_id: str, team: list, speech: str, sio: socketio.AsyncServer):
        """Handle human player team selection with optional summary speech."""
        engine = self.games.get(game_id)
        if not engine:
            return
        
        leader = engine.state.get_leader()
        
        # Add leader's summary speech if provided
        if speech and leader:
            engine.add_discussion(leader.seat, speech)
            
            await self.repo.save_action(
                game_id=game_id,
                round_num=engine.state.current_round,
                action_type="discussion",
                player_seat=leader.seat,
                content=speech,
                vote_attempt=engine.state.vote_attempt,
            )
            
            await sio.emit("game:discussion", {
                "game_id": game_id,
                "seat": leader.seat,
                "player_name": leader.name,
                "content": speech,
                "round": engine.state.current_round,
                "attempt": engine.state.vote_attempt,
                "timestamp": datetime.now().isoformat(),
            }, room=f"game:{game_id}")
        
        if engine.select_team(team):
            engine.state.waiting_for_human = False
            await self._emit_state(game_id, sio)
    
    async def handle_human_discussion(self, game_id: str, content: str, sio: socketio.AsyncServer):
        """Handle human player discussion."""
        engine = self.games.get(game_id)
        if not engine:
            return
        
        seat = engine.state.current_speaker_seat  # Use the tracked speaker seat
        
        # Handle case where human is the first speaker and loop hasn't started yet
        # This happens because proceed_to_discussion() calls _check_human_action() 
        # which sets waiting_for_human=True before the loop runs next_discussion_speaker()
        if seat is None and engine.state.phase == GamePhase.DISCUSSION:
            check_seat = engine.state.current_discussion_seat
            player = engine.state.get_player(check_seat)
            
            # Verify this is indeed the human player we're waiting for
            if player and player.is_human and engine.state.waiting_for_human:
                seat = check_seat
                # Manually advance the state since the loop didn't do it
                engine.state.current_discussion_seat = (engine.state.current_discussion_seat + 1) % engine.state.player_count
                engine.state.discussion_speakers_count += 1
                engine.state.current_speaker_seat = seat
        
        if seat is None:
            return
            
        player = engine.state.get_player(seat)
        
        if player:
            engine.add_discussion(seat, content)
            
            await self.repo.save_action(
                game_id=game_id,
                round_num=engine.state.current_round,
                action_type="discussion",
                player_seat=seat,
                content=content,
                vote_attempt=engine.state.vote_attempt,
            )
            
            await sio.emit("game:discussion", {
                "game_id": game_id,
                "seat": seat,
                "player_name": player.name,
                "content": content,
                "round": engine.state.current_round,
                "attempt": engine.state.vote_attempt,
                "timestamp": datetime.now().isoformat(),
            }, room=f"game:{game_id}")
        
        engine.state.waiting_for_human = False
        await self._emit_state(game_id, sio)
    
    async def handle_human_vote(self, game_id: str, approve: bool, sio: socketio.AsyncServer):
        """Handle human player vote."""
        engine = self.games.get(game_id)
        if not engine:
            return
        
        # Find the human player who hasn't voted
        for player in engine.state.players:
            if player.is_human and player.seat not in engine.state.current_votes:
                engine.cast_vote(player.seat, approve)
                
                await self.repo.save_action(
                    game_id=game_id,
                    round_num=engine.state.current_round,
                    action_type="team_vote",
                    player_seat=player.seat,
                    vote=approve,
                    vote_attempt=engine.state.vote_attempt,
                    proposed_team=engine.state.proposed_team,
                )
                break
        
        engine.state.waiting_for_human = False
        await self._emit_state(game_id, sio)
    
    async def handle_human_quest(self, game_id: str, success: bool, sio: socketio.AsyncServer):
        """Handle human player quest decision."""
        engine = self.games.get(game_id)
        if not engine:
            return
        
        # Find the human player on quest who hasn't voted
        for seat in engine.state.proposed_team:
            player = engine.state.get_player(seat)
            if player and player.is_human and seat not in engine.state.current_quest_votes:
                # Good players must succeed
                if not is_evil(player.role):
                    success = True
                
                engine.cast_quest_vote(seat, success)
                
                await self.repo.save_action(
                    game_id=game_id,
                    round_num=engine.state.current_round,
                    action_type="quest_vote",
                    player_seat=seat,
                    vote=success,
                )
                break
        
        engine.state.waiting_for_human = False
        await self._emit_state(game_id, sio)
    
    async def handle_human_assassination_discussion(self, game_id: str, content: str, sio: socketio.AsyncServer):
        """Handle human player assassination discussion."""
        engine = self.games.get(game_id)
        if not engine:
            return
        
        # Find the human evil player who hasn't spoken yet
        evil_seats = engine.get_evil_seats()
        for seat in evil_seats:
            if seat not in engine.state.assassination_discussion_speakers:
                player = engine.state.get_player(seat)
                if player and player.is_human:
                    engine.add_assassination_discussion(seat, content)
                    
                    await self.repo.save_action(
                        game_id=game_id,
                        round_num=engine.state.current_round,
                        action_type="assassination_discussion",
                        player_seat=seat,
                        content=content,
                    )
                    
                    await sio.emit("game:assassination_discussion", {
                        "game_id": game_id,
                        "seat": seat,
                        "player_name": player.name,
                        "content": content,
                        "timestamp": datetime.now().isoformat(),
                    }, room=f"game:{game_id}")
                    break
        
        engine.state.waiting_for_human = False
        await self._emit_state(game_id, sio)
    
    async def handle_human_assassinate(self, game_id: str, target: int, sio: socketio.AsyncServer):
        """Handle human player assassination."""
        engine = self.games.get(game_id)
        if not engine:
            return
        
        assassin_seat = engine.get_assassin_seat()
        success = engine.assassinate(target)
        
        await self.repo.save_action(
            game_id=game_id,
            round_num=engine.state.current_round,
            action_type="assassination",
            player_seat=assassin_seat,
            target_seat=target,
        )
        
        target_player = engine.state.get_player(target)
        await sio.emit("game:assassination", {
            "game_id": game_id,
            "assassin": assassin_seat,
            "target": target,
            "target_name": target_player.name if target_player else "Unknown",
            "success": success,
            "target_was_merlin": target_player.role == Role.MERLIN if target_player else False,
        }, room=f"game:{game_id}")
        
        engine.state.waiting_for_human = False
        await self._emit_state(game_id, sio)
