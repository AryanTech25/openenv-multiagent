#!/usr/bin/env python3
"""
Training script for the Manager Agent using PPO (Proximal Policy Optimization).

This script trains a PPO agent to manage worker agents in the ManagerWorkerEnv.
The trained model is saved to HuggingFace Hub for deployment.

Usage:
    python training/train_manager.py --timesteps 50000 --learning-rate 3e-4
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import Optional, Dict, Any
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from env import ManagerWorkerEnv, ManagerAction


class TrainingConfig:
    """Configuration for PPO training."""
    
    def __init__(
        self,
        timesteps: int = 50000,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        seed: int = 42,
    ):
        """Initialize training configuration."""
        self.timesteps = timesteps
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.seed = seed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'timesteps': self.timesteps,
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm,
            'seed': self.seed,
        }


class EnvironmentWrapper(gym.Env):
    """Wrapper to make ManagerWorkerEnv compatible with stable-baselines3."""
    
    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        """Initialize environment wrapper."""
        super().__init__()
        self.env = ManagerWorkerEnv(env_config)
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
    
    def _get_observation_space(self):
        """Get observation space compatible with stable-baselines3."""
        
        # Create a dict space matching the observation structure
        return gym.spaces.Dict({
            'task_embedding': gym.spaces.Box(
                low=-1.0, high=1.0, shape=(64,), dtype=np.float32
            ),
            'worker_states': gym.spaces.Box(
                low=0.0, high=1.0, shape=(4, 5), dtype=np.float32
            ),
            'subtask_status': gym.spaces.MultiBinary(4),
            'budget_remaining': gym.spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            'steps_remaining': gym.spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
        })
    
    def _get_action_space(self):
        """Get action space compatible with stable-baselines3."""
        return gym.spaces.Discrete(7)
    
    def reset(self, seed=None, options=None):
        """Reset environment and return observation."""
        if seed is not None:
            np.random.seed(seed)
        obs = self.env.reset()
        return self._convert_observation(obs), {}
    
    def step(self, action: int):
        """Execute action and return observation, reward, done, info."""
        manager_action = ManagerAction(action_id=action)
        obs, reward, done, info = self.env.step(manager_action)
        # Convert to gymnasium format: obs, reward, terminated, truncated, info
        return self._convert_observation(obs), reward, done, False, info
    
    def _convert_observation(self, obs):
        """Convert ManagerWorkerObservation to dict format for stable-baselines3."""
        return {
            'task_embedding': np.array(obs.task_embedding, dtype=np.float32),
            'worker_states': np.array(obs.worker_states, dtype=np.float32),
            'subtask_status': np.array(obs.subtask_status, dtype=np.int8),
            'budget_remaining': np.array([obs.budget_remaining], dtype=np.float32),
            'steps_remaining': np.array([obs.steps_remaining], dtype=np.float32),
        }
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """Render environment."""
        return self.env.render(mode)
    
    def close(self):
        """Close environment."""
        pass


class TrainingMetrics:
    """Track training metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_qualities = []
        self.episode_tokens_used = []
    
    def add_episode(
        self,
        reward: float,
        length: int,
        quality: float,
        tokens_used: int,
    ):
        """Add episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_qualities.append(quality)
        self.episode_tokens_used.append(tokens_used)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        if not self.episode_rewards:
            return {}
        
        return {
            'mean_reward': np.mean(self.episode_rewards[-100:]),
            'mean_length': np.mean(self.episode_lengths[-100:]),
            'mean_quality': np.mean(self.episode_qualities[-100:]),
            'mean_tokens_used': np.mean(self.episode_tokens_used[-100:]),
        }


def create_environment(env_config: Optional[Dict[str, Any]] = None):
    """Create and wrap environment for training."""
    def make_env():
        return EnvironmentWrapper(env_config)
    
    return DummyVecEnv([make_env])


def train_manager(
    config: TrainingConfig,
    env_config: Optional[Dict[str, Any]] = None,
    model_path: str = 'models/ppo_manager',
    verbose: int = 1,
) -> PPO:
    """
    Train PPO agent on ManagerWorkerEnv.
    
    Args:
        config: Training configuration
        env_config: Environment configuration
        model_path: Path to save model
        verbose: Verbosity level
    
    Returns:
        Trained PPO model
    """
    print("=" * 80)
    print("Training Manager Agent with PPO")
    print("=" * 80)
    print(f"\nTraining Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Create environment
    print(f"\nCreating environment...")
    env = create_environment(env_config)
    
    # Create PPO model
    print(f"Creating PPO model...")
    model = PPO(
        policy='MultiInputPolicy',  # Correct policy for Dict observation spaces
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        seed=config.seed,
        verbose=verbose,
    )
    
    # Create model directory
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Train model
    print(f"\nTraining for {config.timesteps} timesteps...")
    print(f"Model will be saved to: {model_path}")
    print("-" * 80)
    
    model.learn(total_timesteps=config.timesteps)
    
    # Save model
    print(f"\nSaving model to {model_path}...")
    model.save(model_path)
    
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    return model


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description='Train PPO agent on ManagerWorkerEnv'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=50000,
        help='Number of timesteps to train (default: 50000)',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4)',
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=2048,
        help='Number of steps per update (default: 2048)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size (default: 64)',
    )
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=10,
        help='Number of epochs per update (default: 10)',
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/ppo_manager',
        help='Path to save model (default: models/ppo_manager)',
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level (default: 1)',
    )
    
    args = parser.parse_args()
    
    # Create training config
    config = TrainingConfig(
        timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
    )
    
    # Train model
    model = train_manager(
        config=config,
        model_path=args.model_path,
        verbose=args.verbose,
    )
    
    print(f"\n✓ Model saved to {args.model_path}")


if __name__ == '__main__':
    main()
