"""Tests for shared LiteLLM runtime configuration."""
import os
from unittest.mock import Mock, patch

from app.config import Settings
from app.litellm_config import configure_litellm


@patch("app.litellm_config.litellm")
def test_configure_litellm_sets_provider_environment(mock_litellm, monkeypatch):
    settings = Mock(spec=Settings)
    settings.litellm_drop_params = True
    settings.azure_openai_api_key = "azure-key"
    settings.azure_openai_endpoint = "https://example.openai.azure.com"
    settings.azure_openai_api_version = "2024-02-15-preview"
    settings.google_api_key = "google-key"
    settings.anthropic_api_key = "anthropic-key"

    for variable in (
        "AZURE_API_KEY",
        "AZURE_API_BASE",
        "AZURE_API_VERSION",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)

    configure_litellm(settings)

    assert mock_litellm.drop_params is True
    assert os.environ["AZURE_API_KEY"] == "azure-key"
    assert os.environ["AZURE_API_BASE"] == "https://example.openai.azure.com"
    assert os.environ["AZURE_API_VERSION"] == "2024-02-15-preview"
    assert os.environ["GOOGLE_API_KEY"] == "google-key"
    assert os.environ["ANTHROPIC_API_KEY"] == "anthropic-key"
