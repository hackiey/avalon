"""Main entry point for the Avalon server."""

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.models.database import init_db

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*"
)

# Create FastAPI app
app = FastAPI(
    title="Avalon LLM Training Engine",
    description="An engine for training LLMs to play the Avalon game",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wrap with Socket.IO ASGI app
asgi_app = socketio.ASGIApp(sio, other_asgi_app=app)


@app.on_event("startup")
async def startup():
    """Initialize database on startup."""
    await init_db()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Avalon LLM Training Engine"}


@app.get("/api/health")
async def health_check():
    """API health check."""
    return {"status": "healthy"}


# Import and register routers after app creation to avoid circular imports
def register_routers():
    from server.api import games, stats, config as config_router, batch
    app.include_router(games.router, prefix="/api/games", tags=["games"])
    app.include_router(stats.router, prefix="/api/stats", tags=["stats"])
    app.include_router(config_router.router, prefix="/api/config", tags=["config"])
    app.include_router(batch.router, prefix="/api/batch", tags=["batch"])


def register_socket_handlers():
    from server.socket import handlers
    handlers.register_handlers(sio)


register_routers()
register_socket_handlers()
