"""Shared LiteLLM runtime configuration."""
import os

import litellm

from app.config import Settings


def configure_litellm(settings: Settings) -> None:
    """Configure LiteLLM from application settings for service and script entry points."""
    litellm.drop_params = settings.litellm_drop_params

    if settings.azure_openai_api_key:
        os.environ["AZURE_API_KEY"] = settings.azure_openai_api_key
    if settings.azure_openai_endpoint:
        os.environ["AZURE_API_BASE"] = settings.azure_openai_endpoint
    if settings.azure_openai_api_version:
        os.environ["AZURE_API_VERSION"] = settings.azure_openai_api_version

    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    if settings.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
