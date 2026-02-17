"""Tests for application configuration."""
from app.config import Settings


def test_settings_defaults():
    """Test that settings have expected default values."""
    settings = Settings()

    assert settings.app_name == "boxctron-describes"
    assert settings.app_version == "0.1.0"
    assert settings.debug is False
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.litellm_model == "azure/gpt-4o"
    assert settings.max_upload_size == 10 * 1024 * 1024
