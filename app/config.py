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
    litellm_model: str = "azure/gpt-4o"
    litellm_temperature: float = 0.7
    litellm_max_tokens: int = 1000

    # File Upload Configuration
    max_upload_size: int = 10 * 1024 * 1024  # 10 MB

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
settings = Settings()
