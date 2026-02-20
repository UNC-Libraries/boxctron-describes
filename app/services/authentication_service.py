"""Authentication service for API key and HTTP Basic authentication."""
import logging
import secrets
from typing import Optional
from fastapi import HTTPException, status
from fastapi.security import HTTPBasicCredentials

from app.config import Settings

logger = logging.getLogger(__name__)


class AuthenticationService:
    """Service for handling API key and HTTP Basic authentication."""

    def __init__(self, settings: Settings):
        """
        Initialize the AuthenticationService.

        Args:
            settings: Application settings containing authentication configuration

        Raises:
            ValueError: If authentication is enabled but no credentials are configured
        """
        self.settings = settings
        self._valid_api_keys = None

        # Parse and cache API keys on initialization
        if self.settings.api_keys:
            self._valid_api_keys = set(key.strip() for key in self.settings.api_keys.split(",") if key.strip())

        # Cache authentication configuration status
        self._has_api_keys = bool(self._valid_api_keys)
        self._has_basic_auth = bool(self.settings.auth_username and self.settings.auth_password)

        # Validate configuration at startup - fail fast if misconfigured
        if self.settings.auth_enabled and not self._has_api_keys and not self._has_basic_auth:
            error_msg = (
                "Authentication is enabled (auth_enabled=True) but no credentials are configured. "
                "Please set either API_KEYS or both AUTH_USERNAME and AUTH_PASSWORD in your environment."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def verify_api_key(self, api_key: Optional[str]) -> bool:
        """
        Verify if the provided API key is valid.

        Args:
            api_key: The API key to verify

        Returns:
            True if the API key is valid, False otherwise
        """
        if not api_key:
            return False

        if not self._valid_api_keys:
            return False

        return api_key in self._valid_api_keys

    def verify_basic_auth(self, credentials: Optional[HTTPBasicCredentials]) -> bool:
        """
        Verify if the provided HTTP Basic authentication credentials are valid.

        Args:
            credentials: The HTTP Basic credentials to verify

        Returns:
            True if the credentials are valid, False otherwise
        """
        if not credentials:
            return False

        if not self.settings.auth_username or not self.settings.auth_password:
            return False

        # Use constant-time comparison to prevent timing attacks
        correct_username = secrets.compare_digest(
            credentials.username,
            self.settings.auth_username
        )
        correct_password = secrets.compare_digest(
            credentials.password,
            self.settings.auth_password
        )

        return correct_username and correct_password

    def verify_authentication(
        self,
        api_key: Optional[str] = None,
        credentials: Optional[HTTPBasicCredentials] = None
    ) -> bool:
        """
        Verify authentication using either API key or HTTP Basic authentication.

        This method supports a hybrid authentication approach, accepting either
        an API key or HTTP Basic credentials.

        Args:
            api_key: Optional API key header value
            credentials: Optional HTTP Basic credentials

        Returns:
            True if authentication succeeds

        Raises:
            HTTPException: If authentication fails
        """
        # Skip authentication if disabled (for dev/test environments)
        if not self.settings.auth_enabled:
            logger.debug("Authentication is disabled")
            return True

        # Note: Configuration validity is already checked in __init__
        # If we reach here with auth_enabled=True, at least one auth method is configured

        # Try API key authentication first
        if api_key and self._has_api_keys:
            if self.verify_api_key(api_key):
                logger.debug("Successfully authenticated via API key")
                return True
            logger.warning("Invalid API key provided")

        # Try HTTP Basic authentication
        if credentials and self._has_basic_auth:
            if self.verify_basic_auth(credentials):
                logger.debug(f"Successfully authenticated user: {credentials.username}")
                return True
            logger.warning(f"Invalid credentials for user: {credentials.username}")

        # If we reach here, authentication failed
        logger.warning("Authentication failed: invalid or missing credentials")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
