"""Dependency injection providers for the application."""
from typing import Optional
from fastapi import Depends, Security
from fastapi.security import APIKeyHeader, HTTPBasic, HTTPBasicCredentials

from app.config import settings
from app.services import (
    ImageNormalizer,
    DescribeImageWorkflow,
    ImageDescriptionService,
    AltTextGenerationService,
    ReviewAssessmentService,
    AuthenticationService
)

# Security schemes for authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
http_basic = HTTPBasic(auto_error=False)


def get_image_normalizer() -> ImageNormalizer:
    """
    Provide an ImageNormalizer instance.

    Returns:
        ImageNormalizer: Configured image normalizer service
    """
    return ImageNormalizer(settings)

def get_image_description_service() -> ImageDescriptionService:
    """
    Provide a ImageDescriptionService instance with dependencies.

    Returns:
        ImageDescriptionService: Configured description service
    """
    return ImageDescriptionService(settings)

def get_alt_text_generation_service() -> AltTextGenerationService:
    """
    Provide an AltTextGenerationService instance.

    Returns:
        AltTextGenerationService: Configured alt text generation service
    """
    return AltTextGenerationService(settings)

def get_review_assessment_service() -> ReviewAssessmentService:
    """
    Provide a ReviewAssessmentService instance.

    Returns:
        ReviewAssessmentService: Configured review assessment service
    """
    return ReviewAssessmentService(settings)

def get_describe_workflow(
    normalizer: ImageNormalizer = Depends(get_image_normalizer),
    image_description_service: ImageDescriptionService = Depends(get_image_description_service),
    alt_text_service: AltTextGenerationService = Depends(get_alt_text_generation_service),
    review_service: ReviewAssessmentService = Depends(get_review_assessment_service)
) -> DescribeImageWorkflow:
    """
    Provide a DescribeImageWorkflow instance with dependencies.

    Args:
        normalizer: Image normalizer service
        image_description_service: Image description service
        alt_text_service: Alt text generation service
        review_service: Review assessment service

    Returns:
        DescribeImageWorkflow: Configured workflow service
    """
    return DescribeImageWorkflow(settings, normalizer, image_description_service, alt_text_service, review_service)


def get_authentication_service() -> AuthenticationService:
    """
    Provide an AuthenticationService instance.

    Returns:
        AuthenticationService: Configured authentication service
    """
    return AuthenticationService(settings)


async def verify_auth(
    api_key: Optional[str] = Security(api_key_header),
    credentials: Optional[HTTPBasicCredentials] = Depends(http_basic),
    auth_service: AuthenticationService = Depends(get_authentication_service)
) -> bool:
    """
    Verify authentication using either API key or HTTP Basic authentication.

    This dependency supports hybrid authentication, accepting either:
    - API key via X-API-Key header
    - HTTP Basic authentication credentials

    Args:
        api_key: Optional API key from X-API-Key header
        credentials: Optional HTTP Basic credentials
        auth_service: Authentication service instance

    Returns:
        True if authentication succeeds

    Raises:
        HTTPException: If authentication fails or is misconfigured
    """
    return auth_service.verify_authentication(api_key=api_key, credentials=credentials)