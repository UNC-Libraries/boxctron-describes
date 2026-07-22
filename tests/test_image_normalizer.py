"""Tests for ImageNormalizer."""
from PIL import Image

from app.config import Settings
from app.services.image_normalizer import ImageNormalizer


def test_init_sets_pil_max_image_pixels():
    """Test the normalizer raises PIL's pixel limit from settings."""
    original_max_pixels = Image.MAX_IMAGE_PIXELS
    try:
        settings = Settings()
        settings.image_max_pixels = 500_000_000

        normalizer = ImageNormalizer(settings)

        assert normalizer.max_pixels == 500_000_000
        assert Image.MAX_IMAGE_PIXELS == 500_000_000
    finally:
        Image.MAX_IMAGE_PIXELS = original_max_pixels


def test_init_allows_disabling_pil_max_image_pixels():
    """Test the normalizer can disable PIL's decompression-bomb pixel limit."""
    original_max_pixels = Image.MAX_IMAGE_PIXELS
    try:
        settings = Settings()
        settings.image_max_pixels = None

        normalizer = ImageNormalizer(settings)

        assert normalizer.max_pixels is None
        assert Image.MAX_IMAGE_PIXELS is None
    finally:
        Image.MAX_IMAGE_PIXELS = original_max_pixels
