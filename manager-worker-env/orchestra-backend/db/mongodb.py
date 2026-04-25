"""
MongoDB connection and utilities.
"""

from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from config import settings


class MongoDB:
    """MongoDB connection manager."""
    
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None
    
    @classmethod
    async def connect(cls) -> None:
        """Connect to MongoDB."""
        if not settings.MONGODB_URL:
            raise ValueError("MONGODB_URL not configured")
        
        cls.client = AsyncIOMotorClient(settings.MONGODB_URL)
        cls.db = cls.client[settings.MONGODB_DB_NAME]
        
        # Create indexes
        await cls._create_indexes()
        
        print(f"✓ Connected to MongoDB: {settings.MONGODB_DB_NAME}")
    
    @classmethod
    async def disconnect(cls) -> None:
        """Disconnect from MongoDB."""
        if cls.client:
            cls.client.close()
            print("✓ Disconnected from MongoDB")
    
    @classmethod
    async def _create_indexes(cls) -> None:
        """Create database indexes."""
        if cls.db is None:
            return
        
        # Episodes collection indexes
        episodes = cls.db["episodes"]
        await episodes.create_index("episode_id", unique=True)
        await episodes.create_index("created_at")
        await episodes.create_index("is_active")
        
        # Metrics collection indexes
        metrics = cls.db["metrics"]
        await metrics.create_index("timestamp")
        await metrics.create_index("episode_id")
        
        # Training sessions collection indexes
        sessions = cls.db["training_sessions"]
        await sessions.create_index("session_id", unique=True)
        await sessions.create_index("created_at")
        
        print("✓ Database indexes created")
    
    @classmethod
    def get_db(cls) -> AsyncIOMotorDatabase:
        """Get database instance."""
        if cls.db is None:
            raise RuntimeError("Database not connected")
        return cls.db


def get_db() -> AsyncIOMotorDatabase:
    """Get database instance."""
    return MongoDB.get_db()
