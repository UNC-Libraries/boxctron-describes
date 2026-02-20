"""Health check endpoint."""
from fastapi import APIRouter, Depends

from app.config import settings
from app.dependencies import verify_auth

router = APIRouter(tags=["health"])


@router.get("/health", dependencies=[Depends(verify_auth)])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version
    }
