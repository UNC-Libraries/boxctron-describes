"""Test configuration and fixtures."""
import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def sample_image_data():
    """Provide sample image data for testing."""
    # Simple 1x1 PNG image (red pixel)
    return (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf'
        b'\xc0\x00\x00\x00\x03\x00\x01\x8f\x1c\x18p\x00\x00\x00\x00IEND\xaeB`\x82'
    )
