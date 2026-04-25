"""
Episode models for API and database.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class EpisodeStartRequest(BaseModel):
    """Request to start a new episode."""
    task_id: str = Field(..., description="Task ID")
    difficulty: int = Field(default=1, ge=1, le=5, description="Difficulty level")
    num_workers: int = Field(default=4, ge=1, le=10, description="Number of workers")
    budget: float = Field(default=1000.0, gt=0, description="Token budget")


class ActionRequest(BaseModel):
    """Request to execute an action."""
    action_id: int = Field(..., ge=0, le=6, description="Action ID")
    target_worker_id: Optional[int] = Field(None, description="Target worker ID")


class EpisodeStartResponse(BaseModel):
    """Response from episode start."""
    episode_id: str
    task_id: str
    status: str
    timestamp: datetime


class StepResponse(BaseModel):
    """Response from step action."""
    episode_id: str
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]
    timestamp: datetime


class EpisodeStateResponse(BaseModel):
    """Current episode state."""
    episode_id: str
    state: Dict[str, Any]
    timestamp: datetime


class EpisodeHistoryResponse(BaseModel):
    """Episode history."""
    episode_id: str
    history: List[Dict[str, Any]]
    timestamp: datetime


class EpisodeEndResponse(BaseModel):
    """Response from episode end."""
    episode_id: str
    status: str
    final_reward: float
    total_steps: int
    timestamp: datetime


class EpisodeListResponse(BaseModel):
    """List of episodes."""
    episodes: List[Dict[str, Any]] = []
    total_count: int = 0
    timestamp: Optional[datetime] = None


class EpisodeDocument(BaseModel):
    """Episode document for MongoDB."""
    episode_id: str
    task_id: str
    difficulty: int
    num_workers: int
    budget: float
    current_observation: Dict[str, Any]
    total_reward: float
    step_count: int
    is_active: bool
    created_at: datetime
    ended_at: Optional[datetime] = None
    episode_log: List[Dict[str, Any]] = []
    final_quality: Optional[float] = None
