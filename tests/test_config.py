"""Tests for application configuration."""
import os
import pytest
from app.config import Settings


def test_settings_defaults(monkeypatch):
    """Test that settings have expected default values."""
    # Clear all environment variables that could affect settings
    # Get all field names from Settings and convert to uppercase env var format
    for field_name in Settings.model_fields.keys():
        env_var_name = field_name.upper()
        monkeypatch.delenv(env_var_name, raising=False)

    # Create settings without loading .env file
    settings = Settings(_env_file=None)

    assert settings.app_name == "boxctron-describes"
    assert settings.app_version == "0.1.0"
    assert settings.debug is False
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.litellm_full_desc_model == "azure/gpt-4o"
    assert settings.max_upload_size == 10 * 1024 * 1024
