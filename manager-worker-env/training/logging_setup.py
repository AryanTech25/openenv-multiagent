#!/usr/bin/env python3
"""
Logging setup for training with TensorBoard and WandB.

This module provides callbacks and utilities for monitoring training progress
with both TensorBoard (local) and Weights & Biases (cloud).
"""

import os
from typing import Optional, Dict, Any
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class TrainingCallback(BaseCallback):
    """Custom callback for tracking training metrics."""
    
    def __init__(self, verbose: int = 0):
        """Initialize callback."""
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_qualities = []
        self.episode_tokens_used = []
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # This is called after each step
        return True
    
    def _on_training_end(self) -> None:
        """Called at end of training."""
        if self.verbose > 0:
            print("\nTraining completed!")


class TensorBoardSetup:
    """Setup TensorBoard logging."""
    
    def __init__(self, log_dir: str = 'logs/tensorboard'):
        """Initialize TensorBoard setup."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def get_log_dir(self) -> str:
        """Get TensorBoard log directory."""
        return self.log_dir
    
    def get_callback(self):
        """Get TensorBoard callback for stable-baselines3."""
        from stable_baselines3.common.callbacks import TensorBoardCallback
        return TensorBoardCallback()


class WandBSetup:
    """Setup Weights & Biases logging."""
    
    def __init__(
        self,
        project: str = 'manager-worker-rl',
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize WandB setup.
        
        Args:
            project: WandB project name
            entity: WandB entity (username or team)
            run_name: Name for this run
            config: Configuration dict to log
        """
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.config = config or {}
        self.run = None
    
    def initialize(self) -> None:
        """Initialize WandB run."""
        try:
            import wandb
            
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.run_name,
                config=self.config,
                reinit=True,
            )
            
            print(f"✓ WandB initialized: {self.run.get_url()}")
        except ImportError:
            print("⚠ WandB not installed. Install with: pip install wandb")
        except Exception as e:
            print(f"⚠ WandB initialization failed: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to WandB."""
        if self.run is None:
            return
        
        try:
            import wandb
            wandb.log(metrics, step=step)
        except Exception as e:
            print(f"⚠ Failed to log metrics to WandB: {e}")
    
    def log_model(self, model_path: str, model_name: str = 'ppo_manager') -> None:
        """Log model artifact to WandB."""
        if self.run is None:
            return
        
        try:
            import wandb
            artifact = wandb.Artifact(model_name, type='model')
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)
            print(f"✓ Model logged to WandB: {model_name}")
        except Exception as e:
            print(f"⚠ Failed to log model to WandB: {e}")
    
    def finish(self) -> None:
        """Finish WandB run."""
        if self.run is None:
            return
        
        try:
            import wandb
            self.run.finish()
            print("✓ WandB run finished")
        except Exception as e:
            print(f"⚠ Failed to finish WandB run: {e}")


class LoggingManager:
    """Manage all logging (TensorBoard + WandB)."""
    
    def __init__(
        self,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        tensorboard_log_dir: str = 'logs/tensorboard',
        wandb_project: str = 'manager-worker-rl',
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize logging manager.
        
        Args:
            use_tensorboard: Enable TensorBoard logging
            use_wandb: Enable WandB logging
            tensorboard_log_dir: TensorBoard log directory
            wandb_project: WandB project name
            wandb_entity: WandB entity
            wandb_run_name: WandB run name
            config: Configuration to log
        """
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        self.tensorboard = None
        self.wandb = None
        
        if use_tensorboard:
            self.tensorboard = TensorBoardSetup(tensorboard_log_dir)
        
        if use_wandb:
            self.wandb = WandBSetup(
                project=wandb_project,
                entity=wandb_entity,
                run_name=wandb_run_name,
                config=config,
            )
            self.wandb.initialize()
    
    def get_tensorboard_log_dir(self) -> Optional[str]:
        """Get TensorBoard log directory."""
        if self.tensorboard is None:
            return None
        return self.tensorboard.get_log_dir()
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to all enabled loggers."""
        if self.wandb is not None:
            self.wandb.log_metrics(metrics, step)
    
    def log_model(self, model_path: str, model_name: str = 'ppo_manager') -> None:
        """Log model to all enabled loggers."""
        if self.wandb is not None:
            self.wandb.log_model(model_path, model_name)
    
    def finish(self) -> None:
        """Finish all logging."""
        if self.wandb is not None:
            self.wandb.finish()


def setup_logging(
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> LoggingManager:
    """
    Setup logging for training.
    
    Args:
        use_tensorboard: Enable TensorBoard
        use_wandb: Enable WandB
        config: Configuration to log
    
    Returns:
        LoggingManager instance
    """
    return LoggingManager(
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        config=config,
    )
