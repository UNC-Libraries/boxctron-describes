"""Tests for the describe endpoint."""
import io
from fastapi import status


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "version" in data


def test_describe_with_uploaded_file(client, sample_image_data):
    """Test describe endpoint with an uploaded file."""
    files = {"file": ("test.png", io.BytesIO(sample_image_data), "image/png")}
    data = {
        "filename": "test.png",
        "mimetype": "image/png",
        "context": "Test image"
    }
    
    response = client.post("/api/v1/describe", files=files, data=data)
    assert response.status_code == status.HTTP_200_OK
    
    result = response.json()
    assert result["success"] is True
    assert result["filename"] == "test.png"
    assert result["result"] is not None


def test_describe_with_file_uri(client):
    """Test describe endpoint with a file URI."""
    data = {
        "file_uri": "https://example.com/image.jpg",
        "filename": "image.jpg",
        "mimetype": "image/jpeg",
        "context": "Test image from URI"
    }
    
    response = client.post("/api/v1/describe", data=data)
    assert response.status_code == status.HTTP_200_OK
    
    result = response.json()
    assert result["success"] is True
    assert result["filename"] == "image.jpg"


def test_describe_without_file_or_uri(client):
    """Test describe endpoint without file or URI (should fail)."""
    data = {
        "filename": "test.jpg",
        "mimetype": "image/jpeg"
    }
    
    response = client.post("/api/v1/describe", data=data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_describe_with_invalid_mimetype(client):
    """Test describe endpoint with invalid MIME type."""
    data = {
        "file_uri": "https://example.com/document.pdf",
        "filename": "document.pdf",
        "mimetype": "application/pdf"
    }
    
    response = client.post("/api/v1/describe", data=data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_describe_with_unsupported_mimetype(client):
    """Test describe endpoint with unsupported image MIME type."""
    data = {
        "file_uri": "https://example.com/image.svg",
        "filename": "image.svg",
        "mimetype": "image/svg+xml"
    }
    
    response = client.post("/api/v1/describe", data=data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_describe_with_both_file_and_uri(client, sample_image_data):
    """Test describe endpoint with both file and URI provided."""
    files = {"file": ("test.png", io.BytesIO(sample_image_data), "image/png")}
    data = {
        "file_uri": "https://example.com/image.jpg",
        "filename": "test.png",
        "mimetype": "image/png"
    }
    
    response = client.post("/api/v1/describe", files=files, data=data)
    # Should succeed - having both is allowed
    assert response.status_code == status.HTTP_200_OK
