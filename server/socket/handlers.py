"""Socket.IO event handlers."""

import socketio
from typing import Dict, Any

# Store active game sessions
active_games: Dict[str, Any] = {}


def register_handlers(sio: socketio.AsyncServer):
    """Register all Socket.IO event handlers."""
    
    @sio.event
    async def connect(sid, environ):
        """Handle client connection."""
        print(f"Client connected: {sid}")
    
    @sio.event
    async def disconnect(sid):
        """Handle client disconnection."""
        print(f"Client disconnected: {sid}")
    
    @sio.event
    async def join_game(sid, data):
        """Join a game room."""
        game_id = data.get("game_id")
        if game_id:
            await sio.enter_room(sid, f"game:{game_id}")
            print(f"Client {sid} joined game {game_id}")
            
            # Try to send current game state if available
            from server.game.manager import GameManager
            manager = GameManager.get_instance()
            engine = manager.get_game(game_id)
            
            if engine:
                # Game is in memory, send current state
                await sio.emit("game:state", {
                    "game_id": game_id,
                    "state": engine.state.to_dict(),
                }, to=sid)
    
    @sio.event
    async def leave_game(sid, data):
        """Leave a game room."""
        game_id = data.get("game_id")
        if game_id:
            await sio.leave_room(sid, f"game:{game_id}")
            print(f"Client {sid} left game {game_id}")
    
    @sio.event
    async def game_start(sid, data):
        """Start a game."""
        game_id = data.get("game_id")
        if game_id:
            from server.game.manager import GameManager
            manager = GameManager.get_instance()
            
            # Check if game exists in memory
            engine = manager.get_game(game_id)
            
            if not engine:
                # Try to restore from database
                engine = await manager.restore_game(game_id)
                
                if engine:
                    # Game restored, emit current state and continue game loop
                    await sio.emit("game:state", {
                        "game_id": game_id,
                        "state": engine.state.to_dict(),
                    }, room=f"game:{game_id}")
                    
                    # If game is in progress, resume the game loop
                    if engine.state.status.value == "in_progress":
                        await manager._run_game_loop(game_id, sio)
                    return
            
            # Normal start for new games
            await manager.start_game(game_id, sio)
    
    @sio.event
    async def human_discussion(sid, data):
        """Handle human player discussion."""
        game_id = data.get("game_id")
        content = data.get("content")
        if game_id and content:
            from server.game.manager import GameManager
            manager = GameManager.get_instance()
            await manager.handle_human_discussion(game_id, content, sio)
    
    @sio.event
    async def human_vote(sid, data):
        """Handle human player vote."""
        game_id = data.get("game_id")
        approve = data.get("approve")
        if game_id is not None and approve is not None:
            from server.game.manager import GameManager
            manager = GameManager.get_instance()
            await manager.handle_human_vote(game_id, approve, sio)
    
    @sio.event
    async def human_quest(sid, data):
        """Handle human player quest decision."""
        game_id = data.get("game_id")
        success = data.get("success")
        if game_id is not None and success is not None:
            from server.game.manager import GameManager
            manager = GameManager.get_instance()
            await manager.handle_human_quest(game_id, success, sio)
    
    @sio.event
    async def human_team_select(sid, data):
        """Handle human player team selection with optional summary speech."""
        game_id = data.get("game_id")
        team = data.get("team")
        speech = data.get("speech", "")  # Optional summary speech
        if game_id and team:
            from server.game.manager import GameManager
            manager = GameManager.get_instance()
            await manager.handle_human_team_select(game_id, team, speech, sio)
    
    @sio.event
    async def human_assassination_discussion(sid, data):
        """Handle human player assassination discussion."""
        game_id = data.get("game_id")
        content = data.get("content")
        if game_id and content:
            from server.game.manager import GameManager
            manager = GameManager.get_instance()
            await manager.handle_human_assassination_discussion(game_id, content, sio)
    
    @sio.event
    async def human_assassinate(sid, data):
        """Handle human player assassination."""
        game_id = data.get("game_id")
        target = data.get("target")
        if game_id is not None and target is not None:
            from server.game.manager import GameManager
            manager = GameManager.get_instance()
            await manager.handle_human_assassinate(game_id, target, sio)
