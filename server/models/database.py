"""MongoDB database connection and collections."""

from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from server.config import settings


# MongoDB client and database
_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


async def init_db():
    """Initialize the MongoDB connection and create indexes."""
    global _client, _db
    
    _client = AsyncIOMotorClient(settings.mongodb_uri)
    _db = _client[settings.mongodb_database]
    
    # Create indexes for better query performance
    # Games collection
    await _db.games.create_index("status")
    await _db.games.create_index("created_at")
    
    # Actions collection
    await _db.actions.create_index("game_id")
    await _db.actions.create_index([("game_id", 1), ("round_num", 1)])
    await _db.actions.create_index([("game_id", 1), ("action_type", 1)])
    await _db.actions.create_index("timestamp")


def get_db() -> AsyncIOMotorDatabase:
    """Get the database instance."""
    if _db is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _db


async def close_db():
    """Close the MongoDB connection."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
