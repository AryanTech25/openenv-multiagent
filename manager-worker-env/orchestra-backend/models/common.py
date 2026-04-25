"""
Common models for API responses.
"""

from typing import Any, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    uptime_seconds: float = 0.0


class StatusResponse(BaseModel):
    """Server status response."""
    status: str = "running"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    uptime_seconds: float = 0.0


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str = "ppo_manager"
    model_version: str = "1.0.0"
    framework: str = "stable-baselines3"
    algorithm: str = "PPO"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CheckpointResponse(BaseModel):
    """Model checkpoint response."""
    checkpoint_id: str
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
