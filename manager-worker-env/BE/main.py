"""
FastAPI backend server for Manager-Worker RL Environment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import time

from config import settings
from database import MongoDB, get_db
from episode_manager import EpisodeManager
from websocket_manager import connection_manager
from metrics_tracker import MetricsTracker
from models import (
    ActionRequest, EpisodeStartRequest, ModelSaveRequest,
    StepResponse, EpisodeStartResponse, EpisodeStateResponse,
    EpisodeHistoryResponse, EpisodeEndResponse, EpisodeListResponse,
    MetricsResponse, MetricsHistoryResponse, ModelInfoResponse,
    CheckpointResponse, HealthResponse, StatusResponse, ErrorResponse,
)
from errors import (
    EpisodeNotFoundError, EpisodeNotActiveError, InvalidActionError,
    DatabaseError, WebSocketError,
)


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    print("Starting up...")
    try:
        await MongoDB.connect()
        print("✓ Database connected")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        raise
    
    yield
    
    # Shutdown
    print("Shutting down...")
    await MongoDB.disconnect()
    print("✓ Database disconnected")


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

@app.get("/health", response_model=HealthResponse)
async def health_check(
    em: EpisodeManager = Depends(get_episode_manager),
) -> HealthResponse:
    """Health check endpoint."""
    uptime = time.time() - startup_time
    episodes = await em.list_episodes()
    
    return HealthResponse(
        status="healthy",
        uptime_seconds=uptime,
        active_episodes=episodes['total_count'],
        total_episodes=episodes['total_count'],
        version=settings.APP_VERSION,
    )


@app.get("/status", response_model=StatusResponse)
async def get_status(
    em: EpisodeManager = Depends(get_episode_manager),
) -> StatusResponse:
    """Get server status."""
    uptime = time.time() - startup_time
    episodes = await em.list_episodes()
    
    return StatusResponse(
        version=settings.APP_VERSION,
        status="running",
        active_connections=connection_manager.get_active_connections_count(),
        active_episodes=episodes['total_count'],
        uptime_seconds=uptime,
        environment={
            'debug': settings.DEBUG,
            'max_episodes': settings.MAX_EPISODES,
            'max_connections': settings.MAX_CONNECTIONS,
        },
    )


# ============================================================================
# Episode Management Endpoints
# ============================================================================

@app.post("/episode/start", response_model=EpisodeStartResponse)
async def start_episode(
    request: EpisodeStartRequest,
    em: EpisodeManager = Depends(get_episode_manager),
) -> EpisodeStartResponse:
    """Start a new episode."""
    try:
        episode_id, initial_obs = await em.create_episode(request.config)
        
        return EpisodeStartResponse(
            episode_id=episode_id,
            observation=initial_obs,
            created_at=datetime.utcnow(),
        )
    except Exception as e:
        raise DatabaseError(str(e))


@app.post("/episode/{episode_id}/step", response_model=StepResponse)
async def step_episode(
    episode_id: str,
    request: ActionRequest,
    em: EpisodeManager = Depends(get_episode_manager),
) -> StepResponse:
    """Execute action in episode."""
    try:
        # Validate action
        if request.action_id < 0 or request.action_id > 6:
            raise InvalidActionError(request.action_id)
        
        # Execute step
        obs, reward, done, info, step_number = await em.step_episode(
            episode_id, request.action_id
        )
        
        # Broadcast update via WebSocket
        await connection_manager.broadcast_step_update(
            episode_id=episode_id,
            observation=obs,
            reward=reward,
            done=done,
            step_number=step_number,
        )
        
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
            step_number=step_number,
        )
    except ValueError as e:
        if "not found" in str(e).lower():
            raise EpisodeNotFoundError(episode_id)
        elif "not active" in str(e).lower():
            raise EpisodeNotActiveError(episode_id)
        else:
            raise InvalidActionError(request.action_id)
    except Exception as e:
        raise DatabaseError(str(e))


@app.get("/episode/{episode_id}/state", response_model=EpisodeStateResponse)
async def get_episode_state(
    episode_id: str,
    em: EpisodeManager = Depends(get_episode_manager),
) -> EpisodeStateResponse:
    """Get current episode state."""
    try:
        state = await em.get_episode_state(episode_id)
        return EpisodeStateResponse(**state)
    except ValueError:
        raise EpisodeNotFoundError(episode_id)
    except Exception as e:
        raise DatabaseError(str(e))


@app.get("/episode/{episode_id}/history", response_model=EpisodeHistoryResponse)
async def get_episode_history(
    episode_id: str,
    em: EpisodeManager = Depends(get_episode_manager),
) -> EpisodeHistoryResponse:
    """Get episode history."""
    try:
        history = await em.get_episode_history(episode_id)
        return EpisodeHistoryResponse(**history)
    except ValueError:
        raise EpisodeNotFoundError(episode_id)
    except Exception as e:
        raise DatabaseError(str(e))


@app.post("/episode/{episode_id}/reset")
async def reset_episode(
    episode_id: str,
    em: EpisodeManager = Depends(get_episode_manager),
):
    """Reset episode to initial state."""
    try:
        initial_obs = await em.reset_episode(episode_id)
        return {
            'episode_id': episode_id,
            'observation': initial_obs,
            'reset_at': datetime.utcnow().isoformat(),
        }
    except ValueError:
        raise EpisodeNotFoundError(episode_id)
    except Exception as e:
        raise DatabaseError(str(e))


@app.post("/episode/{episode_id}/end", response_model=EpisodeEndResponse)
async def end_episode(
    episode_id: str,
    em: EpisodeManager = Depends(get_episode_manager),
    mt: MetricsTracker = Depends(get_metrics_tracker),
) -> EpisodeEndResponse:
    """End episode and return final statistics."""
    try:
        result = await em.end_episode(episode_id)
        
        # Record metrics
        await mt.record_episode(
            episode_id=episode_id,
            total_reward=result['final_reward'],
            episode_length=result['total_steps'],
            final_quality=result['final_quality'],
        )
        
        # Broadcast episode end
        await connection_manager.broadcast_episode_end(
            episode_id=episode_id,
            final_reward=result['final_reward'],
            final_quality=result['final_quality'],
            episode_statistics=result['episode_statistics'],
        )
        
        return EpisodeEndResponse(**result)
    except ValueError:
        raise EpisodeNotFoundError(episode_id)
    except Exception as e:
        raise DatabaseError(str(e))


@app.get("/episode/list", response_model=EpisodeListResponse)
async def list_episodes(
    em: EpisodeManager = Depends(get_episode_manager),
) -> EpisodeListResponse:
    """List all active episodes."""
    try:
        episodes = await em.list_episodes()
        return EpisodeListResponse(**episodes)
    except Exception as e:
        raise DatabaseError(str(e))


# ============================================================================
# Training Metrics Endpoints
# ============================================================================

@app.get("/training/metrics", response_model=MetricsResponse)
async def get_metrics(
    mt: MetricsTracker = Depends(get_metrics_tracker),
) -> MetricsResponse:
    """Get current training metrics."""
    try:
        metrics = await mt.get_current_metrics()
        return MetricsResponse(**metrics)
    except Exception as e:
        raise DatabaseError(str(e))


@app.get("/training/metrics/history", response_model=MetricsHistoryResponse)
async def get_metrics_history(
    limit: int = 100,
    mt: MetricsTracker = Depends(get_metrics_tracker),
) -> MetricsHistoryResponse:
    """Get historical metrics."""
    try:
        history = await mt.get_metrics_history(limit)
        return MetricsHistoryResponse(**history)
    except Exception as e:
        raise DatabaseError(str(e))


@app.get("/training/model/info", response_model=ModelInfoResponse)
async def get_model_info(
    mt: MetricsTracker = Depends(get_metrics_tracker),
) -> ModelInfoResponse:
    """Get model information."""
    try:
        info = await mt.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        raise DatabaseError(str(e))


@app.get("/training/model/checkpoint", response_model=CheckpointResponse)
async def get_checkpoint(
    mt: MetricsTracker = Depends(get_metrics_tracker),
) -> CheckpointResponse:
    """Get checkpoint information."""
    try:
        checkpoint = await mt.get_checkpoint_info()
        return CheckpointResponse(**checkpoint)
    except Exception as e:
        raise DatabaseError(str(e))


@app.post("/training/model/save")
async def save_model(
    request: ModelSaveRequest,
    mt: MetricsTracker = Depends(get_metrics_tracker),
):
    """Save model to HuggingFace Hub."""
    try:
        # TODO: Implement HuggingFace Hub integration
        return {
            'status': 'success',
            'message': 'Model save initiated',
            'repo_id': request.repo_id,
            'timestamp': datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise DatabaseError(str(e))


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    try:
        await connection_manager.connect(websocket)
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Handle subscription requests
            if data.get('type') == 'subscribe':
                episode_id = data.get('episode_id')
                if episode_id:
                    await connection_manager.subscribe_to_episode(websocket, episode_id)
                    await websocket.send_json({
                        'type': 'subscribed',
                        'episode_id': episode_id,
                        'timestamp': datetime.utcnow().isoformat(),
                    })
            
            elif data.get('type') == 'unsubscribe':
                episode_id = data.get('episode_id')
                if episode_id:
                    await connection_manager.unsubscribe_from_episode(websocket, episode_id)
                    await websocket.send_json({
                        'type': 'unsubscribed',
                        'episode_id': episode_id,
                        'timestamp': datetime.utcnow().isoformat(),
                    })
    
    except Exception as e:
        print(f"✗ WebSocket error: {e}")
        await connection_manager.disconnect(websocket)


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        'name': settings.APP_NAME,
        'version': settings.APP_VERSION,
        'status': 'running',
        'docs': '/docs',
        'openapi': '/openapi.json',
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
