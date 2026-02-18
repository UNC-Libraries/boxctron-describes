"""
Application configuration management using pydantic-settings.
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    app_name: str = "boxctron-describes"
    app_version: str = "0.1.0"
    debug: bool = False

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # LLM Provider API Keys
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: Optional[str] = "2024-02-15-preview"

    google_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # LiteLLM Configuration
    litellm_full_desc_model: str = "azure/gpt-4o"
    litellm_full_desc_temperature: float = 0.7
    litellm_full_desc_max_tokens: int = 1000
    litellm_full_desc_reasoning_effort: str = "low"
    litellm_num_retries: int = 3

    # File Upload Configuration
    max_upload_size: int = 10 * 1024 * 1024  # 10 MB

    # Image Processing Configuration
    image_max_dimension: int = 1600

    # Logging Configuration
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_format: str = "default"  # default or json
    log_output: str = "console"  # console, file, or both
    log_file_path: Optional[str] = None  # Path to log file (required if log_output includes 'file')
    log_max_bytes: int = 10 * 1024 * 1024  # 10 MB per log file
    log_backup_count: int = 5  # Number of backup log files to keep

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
settings = Settings()
