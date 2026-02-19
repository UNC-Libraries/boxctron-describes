"""Dependency injection providers for the application."""
from fastapi import Depends

from app.config import settings
from app.services import (
    ImageNormalizer,
    DescribeImageWorkflow,
    ImageDescriptionService,
    AltTextGenerationService,
    ReviewAssessmentService
)


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