"""
Pydantic models for API requests and responses.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# ============================================================================
# Request Models
# ============================================================================

class ActionRequest(BaseModel):
    """Request to execute an action in an episode."""
    action_id: int = Field(..., ge=0, le=6, description="Action ID (0-6)")


class EpisodeStartRequest(BaseModel):
    """Request to start a new episode."""
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional environment configuration"
    )


class ModelSaveRequest(BaseModel):
    """Request to save model to HuggingFace Hub."""
    repo_id: str = Field(..., description="HuggingFace repository ID")


# ============================================================================
# Response Models
# ============================================================================

class ObservationResponse(BaseModel):
    """Environment observation."""
    task_embedding: List[float]
    worker_states: List[List[float]]
    subtask_status: List[int]
    budget_remaining: float
    steps_remaining: float


class StepResponse(BaseModel):
    """Response from step action."""
    observation: ObservationResponse
    reward: float
    done: bool
    info: Dict[str, Any]
    step_number: int


class EpisodeStartResponse(BaseModel):
    """Response from episode start."""
    episode_id: str
    observation: ObservationResponse
    created_at: datetime


class EpisodeStateResponse(BaseModel):
    """Current episode state."""
    episode_id: str
    observation: ObservationResponse
    total_reward: float
    step_count: int
    is_active: bool
    created_at: datetime


class EpisodeHistoryResponse(BaseModel):
    """Episode history/log."""
    episode_id: str
    steps: List[Dict[str, Any]]
    total_reward: float
    final_quality: Optional[float]
    created_at: datetime
    ended_at: Optional[datetime]


class EpisodeEndResponse(BaseModel):
    """Response from episode end."""
    episode_id: str
    final_reward: float
    final_quality: float
    total_steps: int
    episode_statistics: Dict[str, Any]


class EpisodeListResponse(BaseModel):
    """List of active episodes."""
    episodes: List[Dict[str, Any]]
    total_count: int


class MetricsResponse(BaseModel):
    """Training metrics."""
    total_timesteps: int
    mean_reward: float
    episode_count: int
    learning_rate: float
    hallucination_detection_rate: float
    average_episode_length: float


class MetricsHistoryResponse(BaseModel):
    """Historical metrics."""
    metrics: List[Dict[str, Any]]
    total_records: int


class ModelInfoResponse(BaseModel):
    """Model information."""
    model_name: str
    training_timesteps: int
    hyperparameters: Dict[str, Any]
    created_at: datetime


class CheckpointResponse(BaseModel):
    """Model checkpoint information."""
    checkpoint_path: str
    timestamp: datetime
    model_size_kb: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    uptime_seconds: float
    active_episodes: int
    total_episodes: int
    version: str


class StatusResponse(BaseModel):
    """Server status response."""
    version: str
    status: str
    active_connections: int
    active_episodes: int
    uptime_seconds: float
    environment: Dict[str, Any]


# ============================================================================
# WebSocket Message Models
# ============================================================================

class StepUpdateMessage(BaseModel):
    """WebSocket message for step update."""
    type: str = "step_update"
    episode_id: str
    observation: ObservationResponse
    reward: float
    done: bool
    step_number: int
    timestamp: datetime


class WorkerUpdateMessage(BaseModel):
    """WebSocket message for worker update."""
    type: str = "worker_update"
    episode_id: str
    worker_id: int
    state: Dict[str, Any]
    failure_mode: Optional[str]
    quality_score: float
    timestamp: datetime


class BudgetUpdateMessage(BaseModel):
    """WebSocket message for budget update."""
    type: str = "budget_update"
    episode_id: str
    budget_remaining: float
    tokens_used: int
    budget_ratio: float
    timestamp: datetime


class EpisodeEndMessage(BaseModel):
    """WebSocket message for episode end."""
    type: str = "episode_end"
    episode_id: str
    final_reward: float
    final_quality: float
    episode_statistics: Dict[str, Any]
    timestamp: datetime


class ErrorMessage(BaseModel):
    """WebSocket error message."""
    type: str = "error"
    error_code: str
    message: str
    timestamp: datetime


# ============================================================================
# Error Response Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[str] = Field(None, description="Additional details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Database Models (for MongoDB)
# ============================================================================

class EpisodeDocument(BaseModel):
    """Episode document for MongoDB."""
    episode_id: str
    env_config: Dict[str, Any]
    current_observation: Dict[str, Any]
    total_reward: float
    step_count: int
    is_active: bool
    created_at: datetime
    ended_at: Optional[datetime]
    episode_log: List[Dict[str, Any]]
    final_quality: Optional[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "episode_id": "ep_123",
                "env_config": {"max_workers": 4},
                "current_observation": {},
                "total_reward": 50.0,
                "step_count": 10,
                "is_active": True,
                "created_at": "2026-04-22T10:00:00",
                "ended_at": None,
                "episode_log": [],
                "final_quality": None,
            }
        }


class MetricsDocument(BaseModel):
    """Metrics document for MongoDB."""
    timestamp: datetime
    episode_id: Optional[str]
    total_timesteps: int
    mean_reward: float
    episode_count: int
    learning_rate: float
    hallucination_detection_rate: float
    average_episode_length: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2026-04-22T10:00:00",
                "episode_id": "ep_123",
                "total_timesteps": 10000,
                "mean_reward": 52.67,
                "episode_count": 100,
                "learning_rate": 0.0003,
                "hallucination_detection_rate": 0.85,
                "average_episode_length": 12.0,
            }
        }


class TrainingSessionDocument(BaseModel):
    """Training session document for MongoDB."""
    session_id: str
    model_name: str
    training_timesteps: int
    hyperparameters: Dict[str, Any]
    created_at: datetime
    ended_at: Optional[datetime]
    final_metrics: Optional[Dict[str, Any]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_123",
                "model_name": "ppo_manager",
                "training_timesteps": 50000,
                "hyperparameters": {"learning_rate": 0.0003},
                "created_at": "2026-04-22T10:00:00",
                "ended_at": None,
                "final_metrics": None,
            }
        }
