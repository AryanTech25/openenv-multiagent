#!/usr/bin/env python3
"""
OrchestraAI Backend - FastAPI entry point.
"""

import sys
import os
import logging
import time
from contextlib import asynccontextmanager

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from config import settings
from db.mongodb import MongoDB
from api.routes import episode, metrics, model, websocket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
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
    description="OrchestraAI Backend API",
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

# Track startup time
startup_time = time.time()


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": time.time() - startup_time,
    }


@app.get("/status")
async def get_status():
    """Get server status."""
    return {
        "status": "running",
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": time.time() - startup_time,
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "OrchestraAI Backend API",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================================
# Include Routers
# ============================================================================

app.include_router(episode.router)
app.include_router(metrics.router)
app.include_router(model.router)
app.include_router(websocket.router)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the server."""
    import uvicorn
    
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Server: {settings.SERVER_HOST}:{settings.SERVER_PORT}")
    
    uvicorn.run(
        "main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    main()
