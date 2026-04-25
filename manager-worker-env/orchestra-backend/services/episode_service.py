"""
Episode business logic service.
"""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
from motor.motor_asyncio import AsyncIOMotorDatabase
from db.repositories.episode_repo import EpisodeRepository


class EpisodeService:
    """Service for episode management."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """Initialize service."""
        self.db = db
        self.repo = EpisodeRepository(db)
        self.active_episodes: Dict[str, Any] = {}
    
    async def start_episode(
        self,
        task_id: str,
        difficulty: int,
        num_workers: int,
        budget: float,
    ) -> Dict[str, Any]:
        """Start a new episode."""
        episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        
        episode_data = {
            "episode_id": episode_id,
            "task_id": task_id,
            "difficulty": difficulty,
            "num_workers": num_workers,
            "budget": budget,
            "current_observation": {},
            "total_reward": 0.0,
            "step_count": 0,
            "is_active": True,
            "created_at": datetime.utcnow(),
            "ended_at": None,
            "episode_log": [],
            "final_quality": None,
        }
        
        await self.repo.create(episode_data)
        self.active_episodes[episode_id] = episode_data
        
        return episode_data
    
    async def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Get episode by ID."""
        return await self.repo.get_by_id(episode_id)
    
    async def step(
        self,
        episode_id: str,
        action_id: int,
        target_worker_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute action in episode."""
        episode = await self.get_episode(episode_id)
        if not episode:
            raise ValueError(f"Episode not found: {episode_id}")
        
        if not episode["is_active"]:
            raise ValueError(f"Episode not active: {episode_id}")
        
        # Simulate step
        reward = 1.0
        done = False
        observation = {"action": action_id, "worker": target_worker_id}
        
        # Update episode
        episode["step_count"] += 1
        episode["total_reward"] += reward
        episode["current_observation"] = observation
        episode["episode_log"].append({
            "step": episode["step_count"],
            "action": action_id,
            "reward": reward,
            "observation": observation,
        })
        
        await self.repo.update(episode_id, {
            "step_count": episode["step_count"],
            "total_reward": episode["total_reward"],
            "current_observation": observation,
            "episode_log": episode["episode_log"],
        })
        
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": {"step": episode["step_count"]},
        }
    
    async def end_episode(self, episode_id: str) -> Dict[str, Any]:
        """End an episode."""
        episode = await self.get_episode(episode_id)
        if not episode:
            raise ValueError(f"Episode not found: {episode_id}")
        
        episode["is_active"] = False
        episode["ended_at"] = datetime.utcnow()
        episode["final_quality"] = episode["total_reward"] / max(episode["step_count"], 1)
        
        await self.repo.update(episode_id, {
            "is_active": False,
            "ended_at": episode["ended_at"],
            "final_quality": episode["final_quality"],
        })
        
        if episode_id in self.active_episodes:
            del self.active_episodes[episode_id]
        
        return {
            "final_reward": episode["total_reward"],
            "total_steps": episode["step_count"],
        }
    
    async def list_episodes(self, limit: int = 10, skip: int = 0) -> Dict[str, Any]:
        """List episodes."""
        episodes = await self.repo.list_all(limit=limit, skip=skip)
        total = await self.repo.count_all()
        
        return {
            "episodes": episodes,
            "total_count": total,
        }
    
    async def get_episode_history(self, episode_id: str) -> Dict[str, Any]:
        """Get episode history."""
        episode = await self.get_episode(episode_id)
        if not episode:
            raise ValueError(f"Episode not found: {episode_id}")
        
        return episode.get("episode_log", [])
