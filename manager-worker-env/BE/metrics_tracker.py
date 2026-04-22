"""
Training metrics tracking and management.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
import numpy as np


class MetricsTracker:
    """Tracks training metrics and statistics."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """Initialize metrics tracker."""
        self.db = db
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.total_timesteps = 0
        self.episode_count = 0
        self.learning_rate = 3e-4
        self.hallucination_detection_rate = 0.0
    
    async def record_episode(
        self,
        episode_id: str,
        total_reward: float,
        episode_length: int,
        final_quality: float,
    ) -> None:
        """Record metrics for a completed episode."""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.total_timesteps += episode_length
        self.episode_count += 1
        
        # Store in MongoDB
        metrics_doc = {
            'timestamp': datetime.utcnow(),
            'episode_id': episode_id,
            'total_timesteps': self.total_timesteps,
            'mean_reward': float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0,
            'episode_count': self.episode_count,
            'learning_rate': self.learning_rate,
            'hallucination_detection_rate': self.hallucination_detection_rate,
            'average_episode_length': float(np.mean(self.episode_lengths[-100:])) if self.episode_lengths else 0.0,
            'episode_reward': total_reward,
            'episode_length': episode_length,
            'final_quality': final_quality,
        }
        
        await self.db['metrics'].insert_one(metrics_doc)
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        mean_reward = float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0
        avg_length = float(np.mean(self.episode_lengths[-100:])) if self.episode_lengths else 0.0
        
        return {
            'total_timesteps': self.total_timesteps,
            'mean_reward': mean_reward,
            'episode_count': self.episode_count,
            'learning_rate': self.learning_rate,
            'hallucination_detection_rate': self.hallucination_detection_rate,
            'average_episode_length': avg_length,
            'timestamp': datetime.utcnow().isoformat(),
        }
    
    async def get_metrics_history(self, limit: int = 100) -> Dict[str, Any]:
        """Get historical metrics."""
        cursor = self.db['metrics'].find().sort('timestamp', -1).limit(limit)
        metrics = await cursor.to_list(length=limit)
        
        # Convert ObjectId to string and datetime to ISO format
        for metric in metrics:
            metric['_id'] = str(metric['_id'])
            if isinstance(metric.get('timestamp'), datetime):
                metric['timestamp'] = metric['timestamp'].isoformat()
        
        return {
            'metrics': list(reversed(metrics)),
            'total_records': len(metrics),
        }
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': 'ppo_manager',
            'training_timesteps': self.total_timesteps,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
            },
            'created_at': datetime.utcnow().isoformat(),
        }
    
    async def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get checkpoint information."""
        return {
            'checkpoint_path': 'models/ppo_manager.zip',
            'timestamp': datetime.utcnow().isoformat(),
            'model_size_kb': 270.5,
            'total_timesteps': self.total_timesteps,
        }
    
    def update_learning_rate(self, learning_rate: float) -> None:
        """Update learning rate."""
        self.learning_rate = learning_rate
    
    def update_hallucination_detection_rate(self, rate: float) -> None:
        """Update hallucination detection rate."""
        self.hallucination_detection_rate = rate
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        if not self.episode_rewards:
            return {
                'total_episodes': 0,
                'total_timesteps': 0,
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'min_reward': 0.0,
                'max_reward': 0.0,
                'mean_episode_length': 0.0,
            }
        
        return {
            'total_episodes': self.episode_count,
            'total_timesteps': self.total_timesteps,
            'mean_reward': float(np.mean(self.episode_rewards)),
            'std_reward': float(np.std(self.episode_rewards)),
            'min_reward': float(np.min(self.episode_rewards)),
            'max_reward': float(np.max(self.episode_rewards)),
            'mean_episode_length': float(np.mean(self.episode_lengths)),
        }
