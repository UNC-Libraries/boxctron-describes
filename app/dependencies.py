"""Dependency injection providers for the application."""
from fastapi import Depends

from app.config import settings
from app.services import ImageNormalizer, DescribeImageWorkflow, ImageDescriptionService


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

def get_describe_workflow(
    normalizer: ImageNormalizer = Depends(get_image_normalizer)
) -> DescribeImageWorkflow:
    """
    Provide a DescribeImageWorkflow instance with dependencies.

    Args:
        normalizer: Image normalizer service (injected)

    Returns:
        DescribeImageWorkflow: Configured workflow service
    """
    return DescribeImageWorkflow(settings, normalizer)