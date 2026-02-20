"""Unit tests for AuthenticationService."""
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials

from app.services.authentication_service import AuthenticationService
from app.config import Settings


# =============================================================================
# Initialization and Configuration Validation Tests
# =============================================================================

def test_init_with_auth_disabled_no_credentials():
    """Test that service initializes successfully when auth is disabled and no credentials."""
    settings = Settings(
        auth_enabled=False,
        api_keys=None,
        auth_username=None,
        auth_password=None
    )
    # Should not raise an error
    service = AuthenticationService(settings)
    assert service._has_api_keys is False
    assert service._has_basic_auth is False


def test_init_with_auth_enabled_and_api_keys():
    """Test that service initializes successfully with auth enabled and API keys configured."""
    settings = Settings(
        auth_enabled=True,
        api_keys="key1,key2,key3",
        auth_username=None,
        auth_password=None
    )
    service = AuthenticationService(settings)
    assert service._has_api_keys is True
    assert service._has_basic_auth is False
    assert len(service._valid_api_keys) == 3


def test_init_with_auth_enabled_and_basic_auth():
    """Test that service initializes successfully with auth enabled and basic auth configured."""
    settings = Settings(
        auth_enabled=True,
        api_keys=None,
        auth_username="testuser",
        auth_password="testpass"
    )
    service = AuthenticationService(settings)
    assert service._has_api_keys is False
    assert service._has_basic_auth is True


def test_init_with_auth_enabled_and_both_methods():
    """Test that service initializes successfully with both auth methods configured."""
    settings = Settings(
        auth_enabled=True,
        api_keys="key1",
        auth_username="testuser",
        auth_password="testpass"
    )
    service = AuthenticationService(settings)
    assert service._has_api_keys is True
    assert service._has_basic_auth is True


def test_init_fails_with_auth_enabled_no_credentials():
    """Test that service raises ValueError when auth is enabled but no credentials configured."""
    settings = Settings(
        auth_enabled=True,
        api_keys=None,
        auth_username=None,
        auth_password=None
    )
    with pytest.raises(ValueError) as exc_info:
        AuthenticationService(settings)

    assert "Authentication is enabled" in str(exc_info.value)
    assert "no credentials are configured" in str(exc_info.value)


def test_init_fails_with_auth_enabled_only_username():
    """Test that service raises ValueError when only username is configured."""
    settings = Settings(
        auth_enabled=True,
        api_keys=None,
        auth_username="testuser",
        auth_password=None
    )
    with pytest.raises(ValueError) as exc_info:
        AuthenticationService(settings)

    assert "no credentials are configured" in str(exc_info.value)


def test_init_fails_with_auth_enabled_only_password():
    """Test that service raises ValueError when only password is configured."""
    settings = Settings(
        auth_enabled=True,
        api_keys=None,
        auth_username=None,
        auth_password="testpass"
    )
    with pytest.raises(ValueError) as exc_info:
        AuthenticationService(settings)

    assert "no credentials are configured" in str(exc_info.value)


def test_init_fails_with_whitespace_only_api_keys():
    """Test that whitespace-only API keys are treated as invalid configuration."""
    settings = Settings(
        auth_enabled=True,
        api_keys="  ,  , ",  # Only whitespace and commas
        auth_username=None,
        auth_password=None
    )
    with pytest.raises(ValueError) as exc_info:
        AuthenticationService(settings)

    assert "no credentials are configured" in str(exc_info.value)


def test_init_fails_with_empty_string_api_keys():
    """Test that empty string API keys are treated as invalid configuration."""
    settings = Settings(
        auth_enabled=True,
        api_keys="",
        auth_username=None,
        auth_password=None
    )
    with pytest.raises(ValueError) as exc_info:
        AuthenticationService(settings)

    assert "no credentials are configured" in str(exc_info.value)


def test_init_fails_with_single_space_api_keys():
    """Test that single space API keys are treated as invalid configuration."""
    settings = Settings(
        auth_enabled=True,
        api_keys=" ",
        auth_username=None,
        auth_password=None
    )
    with pytest.raises(ValueError) as exc_info:
        AuthenticationService(settings)

    assert "no credentials are configured" in str(exc_info.value)


def test_init_with_whitespace_api_keys_but_valid_basic_auth():
    """Test that service initializes when API keys are whitespace but basic auth is configured."""
    settings = Settings(
        auth_enabled=True,
        api_keys="  ,  , ",  # Only whitespace
        auth_username="testuser",
        auth_password="testpass"
    )
    service = AuthenticationService(settings)
    assert service._has_api_keys is False  # Whitespace keys filtered out
    assert service._has_basic_auth is True


def test_api_keys_are_trimmed():
    """Test that API keys are trimmed of whitespace."""
    settings = Settings(
        auth_enabled=True,
        api_keys=" key1 , key2 , key3 ",
        auth_username=None,
        auth_password=None
    )
    service = AuthenticationService(settings)
    assert "key1" in service._valid_api_keys
    assert "key2" in service._valid_api_keys
    assert "key3" in service._valid_api_keys
    assert " key1 " not in service._valid_api_keys  # Spaces trimmed


# =============================================================================
# Authentication Verification Tests
# =============================================================================

def test_verify_with_auth_disabled():
    """Test that verification succeeds when auth is disabled."""
    settings = Settings(
        auth_enabled=False,
        api_keys=None,
        auth_username=None,
        auth_password=None
    )
    service = AuthenticationService(settings)

    # Should return True regardless of credentials
    assert service.verify_authentication() is True
    assert service.verify_authentication(api_key="anything") is True


def test_verify_with_valid_api_key():
    """Test that valid API key authenticates successfully."""
    settings = Settings(
        auth_enabled=True,
        api_keys="valid-key-1,valid-key-2",
        auth_username=None,
        auth_password=None
    )
    service = AuthenticationService(settings)

    assert service.verify_authentication(api_key="valid-key-1") is True
    assert service.verify_authentication(api_key="valid-key-2") is True


def test_verify_with_invalid_api_key():
    """Test that invalid API key raises HTTPException."""
    settings = Settings(
        auth_enabled=True,
        api_keys="valid-key",
        auth_username=None,
        auth_password=None
    )
    service = AuthenticationService(settings)

    with pytest.raises(HTTPException) as exc_info:
        service.verify_authentication(api_key="invalid-key")

    assert exc_info.value.status_code == 401


def test_verify_with_valid_basic_auth():
    """Test that valid basic auth credentials authenticate successfully."""
    settings = Settings(
        auth_enabled=True,
        api_keys=None,
        auth_username="testuser",
        auth_password="testpass"
    )
    service = AuthenticationService(settings)

    credentials = HTTPBasicCredentials(username="testuser", password="testpass")
    assert service.verify_authentication(credentials=credentials) is True


def test_verify_with_invalid_basic_auth_username():
    """Test that invalid username raises HTTPException."""
    settings = Settings(
        auth_enabled=True,
        api_keys=None,
        auth_username="testuser",
        auth_password="testpass"
    )
    service = AuthenticationService(settings)

    credentials = HTTPBasicCredentials(username="wronguser", password="testpass")
    with pytest.raises(HTTPException) as exc_info:
        service.verify_authentication(credentials=credentials)

    assert exc_info.value.status_code == 401


def test_verify_with_invalid_basic_auth_password():
    """Test that invalid password raises HTTPException."""
    settings = Settings(
        auth_enabled=True,
        api_keys=None,
        auth_username="testuser",
        auth_password="testpass"
    )
    service = AuthenticationService(settings)

    credentials = HTTPBasicCredentials(username="testuser", password="wrongpass")
    with pytest.raises(HTTPException) as exc_info:
        service.verify_authentication(credentials=credentials)

    assert exc_info.value.status_code == 401


def test_verify_with_no_credentials_provided():
    """Test that missing credentials raises HTTPException."""
    settings = Settings(
        auth_enabled=True,
        api_keys="valid-key",
        auth_username="testuser",
        auth_password="testpass"
    )
    service = AuthenticationService(settings)

    with pytest.raises(HTTPException) as exc_info:
        service.verify_authentication()  # No credentials at all

    assert exc_info.value.status_code == 401


def test_verify_api_key_takes_precedence():
    """Test that when both auth methods provided, API key is checked first."""
    settings = Settings(
        auth_enabled=True,
        api_keys="valid-api-key",
        auth_username="testuser",
        auth_password="testpass"
    )
    service = AuthenticationService(settings)

    # Valid API key with invalid basic auth should succeed
    invalid_credentials = HTTPBasicCredentials(username="wrong", password="wrong")
    assert service.verify_authentication(
        api_key="valid-api-key",
        credentials=invalid_credentials
    ) is True


def test_verify_falls_back_to_basic_auth():
    """Test that basic auth works when API key is invalid."""
    settings = Settings(
        auth_enabled=True,
        api_keys="valid-api-key",
        auth_username="testuser",
        auth_password="testpass"
    )
    service = AuthenticationService(settings)

    # Invalid API key but valid basic auth should succeed
    valid_credentials = HTTPBasicCredentials(username="testuser", password="testpass")
    assert service.verify_authentication(
        api_key="invalid-key",
        credentials=valid_credentials
    ) is True


def test_verify_with_whitespace_api_keys_falls_back_to_basic_auth():
    """Test that when API keys config is whitespace, basic auth works as fallback."""
    settings = Settings(
        auth_enabled=True,
        api_keys="  ,  ",  # Whitespace only - no valid keys
        auth_username="testuser",
        auth_password="testpass"
    )
    service = AuthenticationService(settings)

    # User provides some API key value but it won't match (no valid keys configured)
    # Should fall back to basic auth
    valid_credentials = HTTPBasicCredentials(username="testuser", password="testpass")
    assert service.verify_authentication(
        api_key="some-key",
        credentials=valid_credentials
    ) is True


    assert service.verify_api_key("") is False


def test_verify_basic_auth_method():
    """Test the verify_basic_auth helper method."""
    settings = Settings(
        auth_enabled=True,
        api_keys=None,
        auth_username="user",
        auth_password="pass"
    )
    service = AuthenticationService(settings)

    valid_creds = HTTPBasicCredentials(username="user", password="pass")
    invalid_user_creds = HTTPBasicCredentials(username="wrong", password="pass")
    invalid_pass_creds = HTTPBasicCredentials(username="user", password="wrong")

    assert service.verify_basic_auth(valid_creds) is True
    assert service.verify_basic_auth(invalid_user_creds) is False
    assert service.verify_basic_auth(invalid_pass_creds) is False
    assert service.verify_basic_auth(None) is False
