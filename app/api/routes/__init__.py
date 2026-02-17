"""Routes package."""
from app.api.routes.describe import router as describe_router
from app.api.routes.health import router as health_router

__all__ = ["describe_router", "health_router"]
