"""Tests for the describe endpoints."""
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


def test_describe_upload_with_file(client, sample_image_data):
    """Test describe/upload endpoint with an uploaded file."""
    files = {"file": ("test.png", io.BytesIO(sample_image_data), "image/png")}
    data = {
        "filename": "test.png",
        "mimetype": "image/png",
        "context": "Test image"
    }
    
    response = client.post("/api/v1/describe/upload", files=files, data=data)
    assert response.status_code == status.HTTP_200_OK
    
    result = response.json()
    assert result["success"] is True
    assert result["filename"] == "test.png"
    assert result["result"] is not None
    assert result["result"]["metadata"]["source"] == "upload"


def test_describe_upload_without_file(client):
    """Test describe/upload endpoint without file (should fail)."""
    data = {
        "filename": "test.jpg",
        "mimetype": "image/jpeg"
    }
    
    response = client.post("/api/v1/describe/upload", data=data)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_describe_upload_with_invalid_mimetype(client, sample_image_data):
    """Test describe/upload endpoint with invalid MIME type."""
    files = {"file": ("document.pdf", io.BytesIO(sample_image_data), "application/pdf")}
    data = {
        "filename": "document.pdf",
        "mimetype": "application/pdf"
    }
    
    response = client.post("/api/v1/describe/upload", files=files, data=data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_describe_upload_with_unsupported_mimetype(client, sample_image_data):
    """Test describe/upload endpoint with unsupported image MIME type."""
    files = {"file": ("image.svg", io.BytesIO(sample_image_data), "image/svg+xml")}
    data = {
        "filename": "image.svg",
        "mimetype": "image/svg+xml"
    }
    
    response = client.post("/api/v1/describe/upload", files=files, data=data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_describe_uri_with_valid_request(client):
    """Test describe/uri endpoint with valid JSON request."""
    payload = {
        "uri": "https://example.com/image.jpg",
        "filename": "image.jpg",
        "mimetype": "image/jpeg",
        "context": "Test image from URI"
    }
    
    response = client.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_200_OK
    
    result = response.json()
    assert result["success"] is True
    assert result["filename"] == "image.jpg"
    assert result["result"] is not None
    assert result["result"]["metadata"]["source"] == "uri"
    assert result["result"]["metadata"]["uri"] == "https://example.com/image.jpg"


def test_describe_uri_without_uri(client):
    """Test describe/uri endpoint without URI (should fail)."""
    payload = {
        "filename": "image.jpg",
        "mimetype": "image/jpeg"
    }
    
    response = client.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_describe_uri_with_invalid_mimetype(client):
    """Test describe/uri endpoint with invalid MIME type."""
    payload = {
        "uri": "https://example.com/document.pdf",
        "filename": "document.pdf",
        "mimetype": "application/pdf"
    }
    
    response = client.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_describe_uri_with_unsupported_mimetype(client):
    """Test describe/uri endpoint with unsupported image MIME type."""
    payload = {
        "uri": "https://example.com/image.svg",
        "filename": "image.svg",
        "mimetype": "image/svg+xml"
    }
    
    response = client.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_describe_upload_without_context(client, sample_image_data):
    """Test describe/upload endpoint without optional context field."""
    files = {"file": ("test.png", io.BytesIO(sample_image_data), "image/png")}
    data = {
        "filename": "test.png",
        "mimetype": "image/png"
    }
    
    response = client.post("/api/v1/describe/upload", files=files, data=data)
    assert response.status_code == status.HTTP_200_OK


def test_describe_uri_without_context(client):
    """Test describe/uri endpoint without optional context field."""
    payload = {
        "uri": "https://example.com/image.jpg",
        "filename": "image.jpg",
        "mimetype": "image/jpeg"
    }
    
    response = client.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_200_OK
