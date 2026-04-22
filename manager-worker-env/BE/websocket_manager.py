"""
WebSocket connection management and broadcasting.
"""

from typing import Set, Dict, Any, List
from datetime import datetime
from fastapi import WebSocket
import json
import asyncio


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Set[WebSocket] = set()
        self.episode_subscriptions: Dict[str, Set[WebSocket]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
    
    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"✓ WebSocket connected. Total connections: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """Disconnect and unregister a WebSocket connection."""
        self.active_connections.discard(websocket)
        
        # Remove from all episode subscriptions
        for subscribers in self.episode_subscriptions.values():
            subscribers.discard(websocket)
        
        print(f"✓ WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def subscribe_to_episode(self, websocket: WebSocket, episode_id: str) -> None:
        """Subscribe a connection to episode updates."""
        if episode_id not in self.episode_subscriptions:
            self.episode_subscriptions[episode_id] = set()
        
        self.episode_subscriptions[episode_id].add(websocket)
        print(f"✓ Subscribed to episode {episode_id}")
    
    async def unsubscribe_from_episode(self, websocket: WebSocket, episode_id: str) -> None:
        """Unsubscribe a connection from episode updates."""
        if episode_id in self.episode_subscriptions:
            self.episode_subscriptions[episode_id].discard(websocket)
    
    async def broadcast_step_update(
        self,
        episode_id: str,
        observation: Dict[str, Any],
        reward: float,
        done: bool,
        step_number: int,
    ) -> None:
        """Broadcast step update to all subscribers."""
        message = {
            'type': 'step_update',
            'episode_id': episode_id,
            'observation': observation,
            'reward': reward,
            'done': done,
            'step_number': step_number,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        await self._broadcast_to_episode(episode_id, message)
    
    async def broadcast_worker_update(
        self,
        episode_id: str,
        worker_id: int,
        state: Dict[str, Any],
        failure_mode: str,
        quality_score: float,
    ) -> None:
        """Broadcast worker update to all subscribers."""
        message = {
            'type': 'worker_update',
            'episode_id': episode_id,
            'worker_id': worker_id,
            'state': state,
            'failure_mode': failure_mode,
            'quality_score': quality_score,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        await self._broadcast_to_episode(episode_id, message)
    
    async def broadcast_budget_update(
        self,
        episode_id: str,
        budget_remaining: float,
        tokens_used: int,
        budget_ratio: float,
    ) -> None:
        """Broadcast budget update to all subscribers."""
        message = {
            'type': 'budget_update',
            'episode_id': episode_id,
            'budget_remaining': budget_remaining,
            'tokens_used': tokens_used,
            'budget_ratio': budget_ratio,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        await self._broadcast_to_episode(episode_id, message)
    
    async def broadcast_episode_end(
        self,
        episode_id: str,
        final_reward: float,
        final_quality: float,
        episode_statistics: Dict[str, Any],
    ) -> None:
        """Broadcast episode end to all subscribers."""
        message = {
            'type': 'episode_end',
            'episode_id': episode_id,
            'final_reward': final_reward,
            'final_quality': final_quality,
            'episode_statistics': episode_statistics,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        await self._broadcast_to_episode(episode_id, message)
    
    async def broadcast_error(
        self,
        episode_id: str,
        error_code: str,
        message: str,
    ) -> None:
        """Broadcast error to all subscribers."""
        error_message = {
            'type': 'error',
            'episode_id': episode_id,
            'error_code': error_code,
            'message': message,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        await self._broadcast_to_episode(episode_id, error_message)
    
    async def _broadcast_to_episode(self, episode_id: str, message: Dict[str, Any]) -> None:
        """Broadcast message to all subscribers of an episode."""
        if episode_id not in self.episode_subscriptions:
            return
        
        subscribers = self.episode_subscriptions[episode_id].copy()
        disconnected = []
        
        for websocket in subscribers:
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"✗ Error sending message: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            await self.disconnect(websocket)
    
    async def broadcast_to_all(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all connected clients."""
        disconnected = []
        
        for websocket in self.active_connections.copy():
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"✗ Error sending message: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            await self.disconnect(websocket)
    
    def get_active_connections_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)
    
    def get_episode_subscribers_count(self, episode_id: str) -> int:
        """Get number of subscribers for an episode."""
        return len(self.episode_subscriptions.get(episode_id, set()))


# Global connection manager instance
connection_manager = ConnectionManager()
