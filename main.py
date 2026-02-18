"""
Main application entry point for boxctron-describes.

A FastAPI microservice for generating descriptive information from images.
"""
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import litellm

from app.config import settings
from app.logging_config import setup_logging
from app.api.routes import describe_router, health_router

# Initialize logging
setup_logging(settings)
logger = logging.getLogger(__name__)

# Configure LiteLLM
litellm.drop_params = settings.litellm_drop_params

# Set environment variables that LiteLLM expects for Azure OpenAI
if settings.azure_openai_api_key:
    os.environ["AZURE_API_KEY"] = settings.azure_openai_api_key
if settings.azure_openai_endpoint:
    os.environ["AZURE_API_BASE"] = settings.azure_openai_endpoint
if settings.azure_openai_api_version:
    os.environ["AZURE_API_VERSION"] = settings.azure_openai_api_version

# Set environment variables for other providers
if settings.google_api_key:
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key
if settings.anthropic_api_key:
    os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key


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
        debug=settings.debug
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(describe_router)
    app.include_router(health_router)

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
