"""
Server module for Manager-Worker RL Environment.

This module provides the FastAPI server application for multi-mode deployment.
"""

from server.app import app, main

__all__ = ["app", "main"]
