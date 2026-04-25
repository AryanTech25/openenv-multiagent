"""
Metrics API routes.
"""

from fastapi import APIRouter, Depends
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from db.mongodb import get_db
from services.metrics_service import MetricsService
from models.metrics import MetricsResponse, MetricsHistoryResponse

router = APIRouter(prefix="/training/metrics", tags=["Metrics"])


async def get_metrics_service(db: AsyncIOMotorDatabase = Depends(get_db)) -> MetricsService:
    """Get metrics service."""
    return MetricsService(db)


@router.get("", response_model=MetricsResponse)
async def get_metrics(service: MetricsService = Depends(get_metrics_service)):
    """Get current training metrics."""
    metrics = await service.get_current_metrics()
    return MetricsResponse(
        episode_count=metrics.get("episode_count", 0),
        total_steps=metrics.get("total_steps", 0),
        average_reward=metrics.get("average_reward", 0.0),
        average_episode_length=metrics.get("average_episode_length", 0),
        timestamp=datetime.utcnow(),
    )


@router.get("/history", response_model=MetricsHistoryResponse)
async def get_metrics_history(
    limit: int = 100,
    service: MetricsService = Depends(get_metrics_service),
):
    """Get metrics history."""
    history = await service.get_metrics_history(limit=limit)
    return MetricsHistoryResponse(
        history=history,
        count=len(history),
        timestamp=datetime.utcnow(),
    )
