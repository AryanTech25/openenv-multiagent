#!/usr/bin/env python3
"""
OpenEnv-compatible server entry point for Manager-Worker RL Environment.

This module provides the server application for multi-mode deployment.
It wraps the FastAPI backend and provides the required entry point for OpenEnv.

Usage:
    python -m server.app
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import backend components
try:
    from BE.config import settings
    from BE.database import MongoDB, get_db
    from BE.episode_manager import EpisodeManager
    from BE.websocket_manager import connection_manager
    from BE.metrics_tracker import MetricsTracker
    from BE.models import (
        ActionRequest, EpisodeStartRequest, ModelSaveRequest,
        StepResponse, EpisodeStartResponse, EpisodeStateResponse,
        EpisodeHistoryResponse, EpisodeEndResponse, EpisodeListResponse,
        MetricsResponse, MetricsHistoryResponse, ModelInfoResponse,
        CheckpointResponse, HealthResponse, StatusResponse, ErrorResponse,
    )
    from BE.errors import (
        EpisodeNotFoundError, EpisodeNotActiveError, InvalidActionError,
        DatabaseError, WebSocketError,
    )
except ImportError as e:
    logger.error(f"Failed to import backend components: {e}")
    raise


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting Manager-Worker RL Environment Server...")
    try:
        await MongoDB.connect()
        logger.info("✓ Database connected")
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await MongoDB.disconnect()
    logger.info("✓ Database disconnected")


# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API for Manager-Worker RL Environment",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Global managers
episode_manager: EpisodeManager = None
metrics_tracker: MetricsTracker = None
startup_time = time.time()


# ============================================================================
# Dependency Injection
# ============================================================================

async def get_episode_manager() -> EpisodeManager:
    """Get episode manager instance."""
    global episode_manager
    if episode_manager is None:
        db = get_db()
        episode_manager = EpisodeManager(db)
    return episode_manager


async def get_metrics_tracker() -> MetricsTracker:
    """Get metrics tracker instance."""
    global metrics_tracker
    if metrics_tracker is None:
        db = get_db()
        metrics_tracker = MetricsTracker(db)
    return metrics_tracker


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        uptime_seconds=time.time() - startup_time,
    )


@app.get("/status", response_model=StatusResponse, tags=["Status"])
async def get_status():
    """Get server status."""
    return StatusResponse(
        status="running",
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow(),
        uptime_seconds=time.time() - startup_time,
    )


# ============================================================================
# Episode Management Endpoints
# ============================================================================

@app.post("/episode/start", response_model=EpisodeStartResponse, tags=["Episodes"])
async def start_episode(
    request: EpisodeStartRequest,
    manager: EpisodeManager = Depends(get_episode_manager),
):
    """Start a new episode."""
    try:
        episode = await manager.start_episode(
            task_id=request.task_id,
            difficulty=request.difficulty,
            num_workers=request.num_workers,
            budget=request.budget,
        )
        return EpisodeStartResponse(
            episode_id=episode["episode_id"],
            task_id=episode["task_id"],
            status="started",
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Error starting episode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/episode/{episode_id}", response_model=EpisodeStateResponse, tags=["Episodes"])
async def get_episode_state(
    episode_id: str,
    manager: EpisodeManager = Depends(get_episode_manager),
):
    """Get current episode state."""
    try:
        episode = await manager.get_episode(episode_id)
        if not episode:
            raise HTTPException(status_code=404, detail="Episode not found")
        return EpisodeStateResponse(
            episode_id=episode_id,
            state=episode.get("state", {}),
            timestamp=datetime.utcnow(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting episode state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/episode/{episode_id}/step", response_model=StepResponse, tags=["Episodes"])
async def step_episode(
    episode_id: str,
    request: ActionRequest,
    manager: EpisodeManager = Depends(get_episode_manager),
):
    """Execute action in episode."""
    try:
        result = await manager.step(
            episode_id=episode_id,
            action_id=request.action_id,
            target_worker_id=request.target_worker_id,
        )
        return StepResponse(
            episode_id=episode_id,
            observation=result.get("observation", {}),
            reward=result.get("reward", 0.0),
            done=result.get("done", False),
            info=result.get("info", {}),
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Error stepping episode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/episode/{episode_id}/end", response_model=EpisodeEndResponse, tags=["Episodes"])
async def end_episode(
    episode_id: str,
    manager: EpisodeManager = Depends(get_episode_manager),
):
    """End an episode."""
    try:
        result = await manager.end_episode(episode_id)
        return EpisodeEndResponse(
            episode_id=episode_id,
            status="ended",
            final_reward=result.get("final_reward", 0.0),
            total_steps=result.get("total_steps", 0),
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Error ending episode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/episodes", response_model=EpisodeListResponse, tags=["Episodes"])
async def list_episodes(
    skip: int = 0,
    limit: int = 10,
    manager: EpisodeManager = Depends(get_episode_manager),
):
    """List all episodes."""
    try:
        result = await manager.list_episodes()
        return EpisodeListResponse(
            episodes=result.get('episodes', []),
            total_count=result.get('total_count', 0),
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Error listing episodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/episode/{episode_id}/history", response_model=EpisodeHistoryResponse, tags=["Episodes"])
async def get_episode_history(
    episode_id: str,
    manager: EpisodeManager = Depends(get_episode_manager),
):
    """Get episode history."""
    try:
        history = await manager.get_episode_history(episode_id)
        return EpisodeHistoryResponse(
            episode_id=episode_id,
            history=history,
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Error getting episode history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Metrics Endpoints
# ============================================================================

@app.get("/training/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics(
    tracker: MetricsTracker = Depends(get_metrics_tracker),
):
    """Get current training metrics."""
    try:
        metrics = await tracker.get_current_metrics()
        return MetricsResponse(
            episode_count=metrics.get("episode_count", 0),
            total_steps=metrics.get("total_steps", 0),
            average_reward=metrics.get("average_reward", 0.0),
            average_episode_length=metrics.get("average_episode_length", 0),
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/metrics/history", response_model=MetricsHistoryResponse, tags=["Metrics"])
async def get_metrics_history(
    limit: int = 100,
    tracker: MetricsTracker = Depends(get_metrics_tracker),
):
    """Get metrics history."""
    try:
        history = await tracker.get_metrics_history(limit=limit)
        return MetricsHistoryResponse(
            history=history,
            count=len(history),
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Error getting metrics history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Model Endpoints
# ============================================================================

@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get model information."""
    return ModelInfoResponse(
        model_name="ppo_manager",
        model_version=settings.APP_VERSION,
        framework="stable-baselines3",
        algorithm="PPO",
        timestamp=datetime.utcnow(),
    )


@app.post("/model/checkpoint", response_model=CheckpointResponse, tags=["Model"])
async def save_checkpoint(request: ModelSaveRequest):
    """Save model checkpoint."""
    try:
        # Implementation would save the model
        return CheckpointResponse(
            checkpoint_id=f"checkpoint_{int(time.time())}",
            status="saved",
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back or process data
            await connection_manager.broadcast(f"Message: {data}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connection_manager.disconnect(websocket)


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Manager-Worker RL Environment API",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the server."""
    import uvicorn
    
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Server: {settings.SERVER_HOST}:{settings.SERVER_PORT}")
    
    uvicorn.run(
        app,
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
