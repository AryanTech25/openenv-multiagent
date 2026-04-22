"""
Error handling and custom exceptions.
"""

from fastapi import HTTPException, status
from datetime import datetime
from typing import Optional


class APIError(HTTPException):
    """Base API error."""
    
    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        details: Optional[str] = None,
    ):
        """Initialize API error."""
        self.error_code = error_code
        self.message = message
        self.details = details
        
        detail = {
            'error': error_code,
            'message': message,
            'details': details,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        super().__init__(status_code=status_code, detail=detail)


class BadRequestError(APIError):
    """400 Bad Request error."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code='BAD_REQUEST',
            message=message,
            details=details,
        )


class NotFoundError(APIError):
    """404 Not Found error."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code='NOT_FOUND',
            message=message,
            details=details,
        )


class ConflictError(APIError):
    """409 Conflict error."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            error_code='CONFLICT',
            message=message,
            details=details,
        )


class ValidationError(APIError):
    """422 Unprocessable Entity error."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code='VALIDATION_ERROR',
            message=message,
            details=details,
        )


class InternalServerError(APIError):
    """500 Internal Server Error."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code='INTERNAL_SERVER_ERROR',
            message=message,
            details=details,
        )


# Specific error classes

class EpisodeNotFoundError(NotFoundError):
    """Episode not found."""
    
    def __init__(self, episode_id: str):
        super().__init__(
            message=f"Episode not found: {episode_id}",
            details=f"No episode with ID '{episode_id}' exists",
        )


class EpisodeNotActiveError(ConflictError):
    """Episode is not active."""
    
    def __init__(self, episode_id: str):
        super().__init__(
            message=f"Episode is not active: {episode_id}",
            details=f"Episode '{episode_id}' has already ended",
        )


class InvalidActionError(ValidationError):
    """Invalid action ID."""
    
    def __init__(self, action_id: int):
        super().__init__(
            message=f"Invalid action ID: {action_id}",
            details=f"Action ID must be between 0 and 6, got {action_id}",
        )


class InvalidConfigError(ValidationError):
    """Invalid environment configuration."""
    
    def __init__(self, message: str):
        super().__init__(
            message=f"Invalid configuration: {message}",
            details="Check environment configuration parameters",
        )


class DatabaseError(InternalServerError):
    """Database operation error."""
    
    def __init__(self, message: str):
        super().__init__(
            message=f"Database error: {message}",
            details="An error occurred while accessing the database",
        )


class WebSocketError(InternalServerError):
    """WebSocket error."""
    
    def __init__(self, message: str):
        super().__init__(
            message=f"WebSocket error: {message}",
            details="An error occurred with the WebSocket connection",
        )
