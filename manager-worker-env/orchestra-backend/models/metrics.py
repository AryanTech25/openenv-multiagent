"""
Metrics models for API and database.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class MetricsResponse(BaseModel):
    """Training metrics response."""
    episode_count: int = 0
    total_steps: int = 0
    average_reward: float = 0.0
    average_episode_length: int = 0
    timestamp: Optional[datetime] = None


class MetricsHistoryResponse(BaseModel):
    """Historical metrics response."""
    history: List[Dict[str, Any]] = []
    count: int = 0
    timestamp: Optional[datetime] = None


class MetricsDocument(BaseModel):
    """Metrics document for MongoDB."""
    timestamp: datetime
    episode_id: Optional[str] = None
    total_timesteps: int
    mean_reward: float
    episode_count: int
    learning_rate: float
    hallucination_detection_rate: float
    average_episode_length: float
