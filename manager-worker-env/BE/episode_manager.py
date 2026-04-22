"""
Episode management - handles episode lifecycle and state.
"""

import uuid
import sys
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase

# Add parent directory to path to import env
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import ManagerWorkerEnv, ManagerAction


class Episode:
    """Represents a single episode."""
    
    def __init__(self, episode_id: str, env_config: Optional[Dict[str, Any]] = None):
        """Initialize episode."""
        self.episode_id = episode_id
        self.env_config = env_config or {}
        self.env = ManagerWorkerEnv(self.env_config)
        self.current_observation = None
        self.total_reward = 0.0
        self.step_count = 0
        self.is_active = True
        self.created_at = datetime.utcnow()
        self.ended_at = None
        self.episode_log: List[Dict[str, Any]] = []
        self.final_quality = None
    
    async def reset(self) -> Dict[str, Any]:
        """Reset episode and return initial observation."""
        self.current_observation = self.env.reset()
        self.total_reward = 0.0
        self.step_count = 0
        self.is_active = True
        self.episode_log = []
        self.final_quality = None
        
        return self._observation_to_dict(self.current_observation)
    
    async def step(self, action_id: int) -> tuple:
        """Execute action and return (observation, reward, done, info)."""
        if not self.is_active:
            raise ValueError("Episode is not active")
        
        if action_id < 0 or action_id > 6:
            raise ValueError(f"Invalid action_id: {action_id}")
        
        # Execute action
        action = ManagerAction(action_id=action_id)
        obs, reward, done, info = self.env.step(action)
        
        # Update state
        self.current_observation = obs
        self.total_reward += reward
        self.step_count += 1
        
        # Log step
        self.episode_log.append({
            'step': self.step_count,
            'action_id': action_id,
            'action_name': self.env.ACTION_NAMES[action_id],
            'reward': reward,
            'done': done,
            'timestamp': datetime.utcnow().isoformat(),
        })
        
        # Check termination
        if done:
            self.is_active = False
            self.ended_at = datetime.utcnow()
            self.final_quality = info.get('final_quality', 0.0)
        
        return (
            self._observation_to_dict(obs),
            reward,
            done,
            info,
        )
    
    async def end(self) -> Dict[str, Any]:
        """End episode and return final statistics."""
        if not self.is_active:
            raise ValueError("Episode is already ended")
        
        self.is_active = False
        self.ended_at = datetime.utcnow()
        
        # Calculate final quality if not already set
        if self.final_quality is None:
            self.final_quality = self.env._compute_final_quality()
        
        return {
            'episode_id': self.episode_id,
            'final_reward': self.total_reward,
            'final_quality': self.final_quality,
            'total_steps': self.step_count,
            'episode_statistics': {
                'total_reward': self.total_reward,
                'average_reward_per_step': self.total_reward / self.step_count if self.step_count > 0 else 0,
                'total_steps': self.step_count,
                'duration_seconds': (self.ended_at - self.created_at).total_seconds(),
            }
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current episode state."""
        return {
            'episode_id': self.episode_id,
            'observation': self._observation_to_dict(self.current_observation),
            'total_reward': self.total_reward,
            'step_count': self.step_count,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
        }
    
    def get_history(self) -> Dict[str, Any]:
        """Get episode history."""
        return {
            'episode_id': self.episode_id,
            'steps': self.episode_log,
            'total_reward': self.total_reward,
            'final_quality': self.final_quality,
            'created_at': self.created_at.isoformat(),
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary for storage."""
        return {
            'episode_id': self.episode_id,
            'env_config': self.env_config,
            'current_observation': self._observation_to_dict(self.current_observation),
            'total_reward': self.total_reward,
            'step_count': self.step_count,
            'is_active': self.is_active,
            'created_at': self.created_at,
            'ended_at': self.ended_at,
            'episode_log': self.episode_log,
            'final_quality': self.final_quality,
        }
    
    @staticmethod
    def _observation_to_dict(obs) -> Dict[str, Any]:
        """Convert observation to dictionary."""
        if obs is None:
            return {}
        
        return {
            'task_embedding': obs.task_embedding,
            'worker_states': obs.worker_states,
            'subtask_status': obs.subtask_status,
            'budget_remaining': obs.budget_remaining,
            'steps_remaining': obs.steps_remaining,
        }


class EpisodeManager:
    """Manages all episodes."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """Initialize episode manager."""
        self.db = db
        self.episodes: Dict[str, Episode] = {}
    
    async def create_episode(self, config: Optional[Dict[str, Any]] = None) -> tuple:
        """Create new episode and return (episode_id, initial_observation)."""
        episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        episode = Episode(episode_id, config)
        
        # Reset to get initial observation
        initial_obs = await episode.reset()
        
        # Store in memory
        self.episodes[episode_id] = episode
        
        # Store in MongoDB
        await self.db['episodes'].insert_one(episode.to_dict())
        
        return episode_id, initial_obs
    
    async def step_episode(self, episode_id: str, action_id: int) -> tuple:
        """Execute step in episode."""
        episode = self._get_episode(episode_id)
        obs, reward, done, info = await episode.step(action_id)
        
        # Update in MongoDB
        await self.db['episodes'].update_one(
            {'episode_id': episode_id},
            {'$set': {
                'current_observation': episode.current_observation.dict() if episode.current_observation else {},
                'total_reward': episode.total_reward,
                'step_count': episode.step_count,
                'is_active': episode.is_active,
                'ended_at': episode.ended_at,
                'episode_log': episode.episode_log,
                'final_quality': episode.final_quality,
            }}
        )
        
        return obs, reward, done, info, episode.step_count
    
    async def get_episode_state(self, episode_id: str) -> Dict[str, Any]:
        """Get current episode state."""
        episode = self._get_episode(episode_id)
        return episode.get_state()
    
    async def get_episode_history(self, episode_id: str) -> Dict[str, Any]:
        """Get episode history."""
        episode = self._get_episode(episode_id)
        return episode.get_history()
    
    async def reset_episode(self, episode_id: str) -> Dict[str, Any]:
        """Reset episode to initial state."""
        episode = self._get_episode(episode_id)
        initial_obs = await episode.reset()
        
        # Update in MongoDB
        await self.db['episodes'].update_one(
            {'episode_id': episode_id},
            {'$set': {
                'current_observation': episode.current_observation.dict() if episode.current_observation else {},
                'total_reward': episode.total_reward,
                'step_count': episode.step_count,
                'is_active': episode.is_active,
                'episode_log': episode.episode_log,
            }}
        )
        
        return initial_obs
    
    async def end_episode(self, episode_id: str) -> Dict[str, Any]:
        """End episode and return final statistics."""
        episode = self._get_episode(episode_id)
        result = await episode.end()
        
        # Update in MongoDB
        await self.db['episodes'].update_one(
            {'episode_id': episode_id},
            {'$set': {
                'is_active': episode.is_active,
                'ended_at': episode.ended_at,
                'final_quality': episode.final_quality,
            }}
        )
        
        return result
    
    async def list_episodes(self) -> Dict[str, Any]:
        """List all active episodes."""
        active_episodes = [
            {
                'episode_id': ep.episode_id,
                'total_reward': ep.total_reward,
                'step_count': ep.step_count,
                'is_active': ep.is_active,
                'created_at': ep.created_at.isoformat(),
            }
            for ep in self.episodes.values()
        ]
        
        return {
            'episodes': active_episodes,
            'total_count': len(active_episodes),
        }
    
    def _get_episode(self, episode_id: str) -> Episode:
        """Get episode by ID or raise error."""
        if episode_id not in self.episodes:
            raise ValueError(f"Episode not found: {episode_id}")
        return self.episodes[episode_id]
    
    async def cleanup_inactive_episodes(self) -> int:
        """Remove inactive episodes from memory."""
        inactive_ids = [
            ep_id for ep_id, ep in self.episodes.items()
            if not ep.is_active
        ]
        
        for ep_id in inactive_ids:
            del self.episodes[ep_id]
        
        return len(inactive_ids)
