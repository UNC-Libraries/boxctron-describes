"""
Main application entry point for boxctron-describes.

A FastAPI microservice for generating descriptive information from images.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.litellm_config import configure_litellm
from app.logging_config import setup_logging
from app.api.routes import describe_router, health_router
from app.exception_handlers import register_exception_handlers

# Initialize logging
setup_logging(settings)
logger = logging.getLogger(__name__)

configure_litellm(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")

    yield

    # Shutdown
    logger.info(f"Shutting down {settings.app_name}")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="A microservice for generating descriptive information from images using AI vision models",
        lifespan=lifespan,
        debug=settings.debug,
        root_path=settings.root_path
    )

    # Configure CORS - minimal config since primary usage is server-to-server
    # CORS is only enforced by browsers; HTTP clients ignore it
    # Only needed if the upload form or API is accessed directly via browser
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows browser access from any origin
        allow_credentials=False,  # Must be False when allow_origins=["*"]
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(describe_router)
    app.include_router(health_router)

    # Register exception handlers
    register_exception_handlers(app)

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
