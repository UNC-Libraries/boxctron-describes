"""
Global exception handlers for the FastAPI application.
"""
import logging
import traceback
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI):
    """
    Register global exception handlers for the application.
    """
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle ValueError exceptions (e.g., invalid image data)."""
        logger.error(f"ValueError on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)}
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        """Handle FileNotFoundError exceptions."""
        logger.error(f"FileNotFoundError on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "Requested file not found"}
        )

    @app.exception_handler(IOError)
    async def io_error_handler(request: Request, exc: IOError):
        """Handle IOError exceptions (e.g., file read/write failures)."""
        logger.error(f"IOError on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "File operation failed"}
        )

    @app.exception_handler(OSError)
    async def os_error_handler(request: Request, exc: OSError):
        """Handle OSError exceptions."""
        logger.error(f"OSError on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "System operation failed"}
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle any unhandled exceptions."""
        logger.error(
            f"Unhandled exception on {request.url.path}: {exc}\n"
            f"Traceback: {traceback.format_exc()}"
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )
