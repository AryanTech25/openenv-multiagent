"""
Backend configuration settings.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Server
    APP_NAME: str = "Manager-Worker RL Environment API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # MongoDB
    MONGODB_URL: Optional[str] = None  # Will be provided by user
    MONGODB_DB_NAME: str = "manager_worker_rl"
    
    # Environment
    MAX_EPISODES: int = 100
    EPISODE_TIMEOUT: int = 3600  # 1 hour
    
    # WebSocket
    MAX_CONNECTIONS: int = 100
    MESSAGE_QUEUE_SIZE: int = 1000
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
