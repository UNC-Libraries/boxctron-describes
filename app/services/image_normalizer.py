"""Image normalization service."""
import base64
import logging
from pathlib import Path
from PIL import Image, ImageOps
import io

from app.config import Settings

logger = logging.getLogger(__name__)


class ImageNormalizer:
    """Service for normalizing images to standard dimensions."""

    def __init__(self, settings: Settings):
        """
        Initialize the ImageNormalizer.

        Args:
            settings: Application settings containing normalization config
        """
        self.settings = settings
        self.max_dimension = settings.image_max_dimension

    def normalize_image(self, image_path: Path) -> str:
        """
        Normalize an image to a maximum dimension and return as base64 data URL.

        Args:
            image_path: Path to the image file to normalize

        Returns:
            Base64-encoded data URL suitable for LLM APIs (e.g., "data:image/jpeg;base64,...")

        Raises:
            IOError: If the image cannot be read or processed
            OSError: If the image file cannot be opened
        """
        logger.debug(f'Normalizing {image_path}')
        # Load the image and normalize it
        with Image.open(image_path) as image:
            try:
                jpeg_bytes = self.normalize_pillow(image)
            except TypeError:
                # Some TIFF files have malformed EXIF/XMP metadata that causes PIL to fail during load/resize
                # Strip the problematic metadata and retry
                logger.info(f'Stripping metadata from {image_path} due to transformation error, retrying')
                image.info.pop('xmp', None)
                image.info.pop('icc_profile', None)
                image.info.pop('exif', None)
                jpeg_bytes = self.normalize_pillow(image)

            # Convert to base64 data URL
            base64_str = base64.b64encode(jpeg_bytes).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_str}"

    def normalize_pillow(self, image: Image.Image) -> bytes:
        """
        Normalize a PIL Image and return as JPEG bytes.

        Args:
            image: PIL Image object to normalize

        Returns:
            JPEG-encoded image bytes

        Raises:
            TypeError: If image metadata causes processing errors (caller should strip metadata and retry)
        """
        image = ImageOps.exif_transpose(image)

        # Resize the image
        width, height = image.size
        if width > height:
            new_width = self.max_dimension
            new_height = int(height * (new_width / width))
        else:
            new_height = self.max_dimension
            new_width = int(width * (new_height / height))
        image = image.resize((new_width, new_height))

        image = image.convert("RGB")

        # Save the image to a BytesIO object
        image_data = io.BytesIO()
        image.save(image_data, format="JPEG")
        return image_data.getvalue()
