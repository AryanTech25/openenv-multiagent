"""
Pydantic models for OpenEnv environment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from openenv.core import State, Observation, Action


class WorkerStateModel(BaseModel):
    """Represents the state of a single worker agent."""
    
    worker_id: int
    is_active: bool = False
    current_subtask_id: Optional[int] = None
    progress: float = 0.0  # 0.0 to 1.0
    hallucination_risk_score: float = 0.0  # 0.0 to 1.0
    output_quality_if_checked: float = 0.0  # 0.0 to 1.0
    tokens_consumed: int = 0
    failure_mode: Optional[str] = None  # 'hallucination', 'off_task', 'incomplete', 'stuck', None
    output_buffer: str = ""
    patience_counter: int = 0  # For detecting stuck loops
    skill_level: float = 0.5  # 0.3 to 1.0


class ManagerWorkerObservation(Observation):
    """Observation from the environment."""
    
    task_embedding: List[float] = Field(..., description="64-dimensional task embedding")
    worker_states: List[List[float]] = Field(..., description="4x5 worker states array")
    subtask_status: List[int] = Field(..., description="Binary array of subtask completion")
    budget_remaining: float = Field(..., description="Remaining budget as fraction [0, 1]")
    steps_remaining: float = Field(..., description="Remaining steps as fraction [0, 1]")
    episode_log: List[Dict[str, Any]] = Field(default_factory=list, description="Episode log")
    hallucination_catch_rate: float = Field(default=0.0, description="Hallucination detection rate")


class ManagerAction(Action):
    """Action from the Manager Agent."""
    
    action_id: int = Field(..., ge=0, le=6, description="Action ID (0-6)")
    target_worker_id: Optional[int] = Field(default=None, description="Target worker ID if applicable")


class ManagerWorkerState(State):
    """Internal state of the environment."""
    
    workers: List[WorkerStateModel] = Field(default_factory=list)
    task: Optional[Dict[str, Any]] = None
    budget_remaining: int = 1000
    step_counter: int = 0
    episode_log: List[Dict[str, Any]] = Field(default_factory=list)
    subtask_status: List[bool] = Field(default_factory=list)
    subtask_assignments: List[Optional[int]] = Field(default_factory=list)
    hallucination_catch_rate: float = 0.0
    hallucinations_detected: int = 0
    hallucinations_approved: int = 0
    false_positives: int = 0
    max_workers: int = 4
    max_steps: int = 50
    token_budget: int = 1000
    task_difficulty: int = 3
    failure_injection_rate: float = 0.6
