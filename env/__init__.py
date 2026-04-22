"""
Manager-Worker Multi-Agent RL Environment
OpenEnv-based environment for training a Manager Agent to coordinate Worker Agents
"""

from env.manager_worker_env import ManagerWorkerEnv
from env.task_library import TaskLibrary, Task, Subtask
from env.hallucination_engine import HallucinationEngine, FailureMode
from env.reward_calculator import RewardCalculator

__all__ = [
    'ManagerWorkerEnv',
    'TaskLibrary',
    'Task',
    'Subtask',
    'HallucinationEngine',
    'FailureMode',
    'RewardCalculator',
]
