"""Logging configuration for the application."""
import logging
import logging.handlers
import sys
from pathlib import Path

from app.config import Settings


def setup_logging(settings: Settings) -> None:
    """
    Configure application logging based on settings.

    Args:
        settings: Application settings containing logging configuration

    Environment variable examples:
        # Development (console only, debug level)
        LOG_LEVEL=DEBUG
        LOG_OUTPUT=console

        # Production (file with rotation)
        LOG_LEVEL=INFO
        LOG_OUTPUT=file
        LOG_FILE_PATH=/var/log/boxctron-describes/app.log
        LOG_MAX_BYTES=10485760
        LOG_BACKUP_COUNT=5

        # Both console and file
        LOG_OUTPUT=both
        LOG_FILE_PATH=/var/log/boxctron-describes/app.log
    """
    # Parse log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Choose format
    if settings.log_format == "json":
        log_format = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
    else:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(log_format)

    # Add console handler if requested
    if settings.log_output in ["console", "both"]:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if requested
    if settings.log_output in ["file", "both"]:
        if not settings.log_file_path:
            raise ValueError("log_file_path must be set when log_output is 'file' or 'both'")

        # Ensure log directory exists
        log_file = Path(settings.log_file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=settings.log_file_path,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Log initial configuration
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: level={settings.log_level}, output={settings.log_output}, "
        f"format={settings.log_format}"
    )

    # Configure uvicorn/FastAPI loggers to use our handlers
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"]:
        logger_obj = logging.getLogger(logger_name)
        logger_obj.setLevel(log_level)
        logger_obj.handlers = []  # Remove default handlers
        # Add our handlers
        for handler in root_logger.handlers:
            logger_obj.addHandler(handler)
        logger_obj.propagate = False  # Don't propagate to root to avoid duplicates

    # Set log levels for noisy third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
