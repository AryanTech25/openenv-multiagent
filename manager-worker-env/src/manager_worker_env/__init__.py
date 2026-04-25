"""
Manager-Worker RL Environment - OpenEnv-compatible multi-agent environment.

This package provides a reinforcement learning environment where a Manager agent
coordinates multiple Worker agents to complete complex tasks under token budget constraints.
"""

from .environment import ManagerWorkerEnv
from .models import (
    ManagerWorkerObservation,
    ManagerAction,
    ManagerWorkerState,
    WorkerStateModel,
)

__version__ = "1.0.0"
__all__ = [
    "ManagerWorkerEnv",
    "ManagerWorkerObservation",
    "ManagerAction",
    "ManagerWorkerState",
    "WorkerStateModel",
]
