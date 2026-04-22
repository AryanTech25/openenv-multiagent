"""
Training module for the Manager-Worker RL Environment.

This module provides utilities for training PPO agents on the ManagerWorkerEnv
with support for logging (TensorBoard + WandB) and model deployment to HuggingFace Hub.
"""

from training.train_manager import (
    TrainingConfig,
    EnvironmentWrapper,
    TrainingMetrics,
    create_environment,
    train_manager,
)

from training.logging_setup import (
    TrainingCallback,
    TensorBoardSetup,
    WandBSetup,
    LoggingManager,
    setup_logging,
)

from training.huggingface_integration import (
    HuggingFaceModelCard,
    HuggingFaceIntegration,
)

__all__ = [
    'TrainingConfig',
    'EnvironmentWrapper',
    'TrainingMetrics',
    'create_environment',
    'train_manager',
    'TrainingCallback',
    'TensorBoardSetup',
    'WandBSetup',
    'LoggingManager',
    'setup_logging',
    'HuggingFaceModelCard',
    'HuggingFaceIntegration',
]
