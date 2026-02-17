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

    # Verify full result structure
    result_data = result["result"]
    assert "full_description" in result_data
    assert "alt_text" in result_data
    assert "transcript" in result_data
    assert "safety_assessment" in result_data
    assert "review_assessment" in result_data
    assert "version" in result_data

    # Verify safety_assessment structure and default values
    safety = result_data["safety_assessment"]
    assert safety["people_visible"] == "NO"
    assert safety["violent_content"] == "NONE"
    assert safety["confidence"] == "LOW"
    assert safety["symbols_present"]["types"] == ["NONE"]
    assert safety["symbols_present"]["names"] == []
    assert safety["text_characteristics"]["text_present"] == "NO"
    assert safety["text_characteristics"]["text_type"] == "N/A"

    # Verify review_assessment structure and default values
    review = result_data["review_assessment"]
    assert review["biased_language"] == "NO"
    assert review["stereotyping"] == "NO"
    assert review["offensive_language"] == "NO"
    assert review["safety_assessment_consistency"] == "CONSISTENT"
    assert review["concerns_for_review"] == []
    assert isinstance(review["concerns_for_review"], list)

    # Verify version structure
    version = result_data["version"]
    assert "version" in version
    assert "models" in version
    assert "timestamp" in version
    assert isinstance(version["models"], list)
    assert len(version["models"]) > 0


def test_describe_upload_without_file(client):
    """Test describe/upload endpoint without file (should fail)."""
    data = {
        "filename": "test.jpg",
        "mimetype": "image/jpeg"
    }

    response = client.post("/api/v1/describe/upload", data=data)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_describe_upload_with_invalid_mimetype(client, sample_image_data):
    """Test describe/upload endpoint with invalid MIME type."""
    files = {"file": ("document.pdf", io.BytesIO(sample_image_data), "application/pdf")}
    data = {
        "filename": "document.pdf",
        "mimetype": "application/pdf"
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

    # Verify full result structure
    result_data = result["result"]
    assert "full_description" in result_data
    assert "alt_text" in result_data
    assert "transcript" in result_data
    assert "safety_assessment" in result_data
    assert "review_assessment" in result_data
    assert "version" in result_data

    # Verify safety_assessment structure and default values
    safety = result_data["safety_assessment"]
    assert safety["people_visible"] == "NO"
    assert safety["violent_content"] == "NONE"
    assert safety["confidence"] == "LOW"
    assert safety["symbols_present"]["types"] == ["NONE"]
    assert safety["symbols_present"]["names"] == []
    assert safety["text_characteristics"]["text_present"] == "NO"
    assert safety["text_characteristics"]["text_type"] == "N/A"

    # Verify review_assessment structure and default values
    review = result_data["review_assessment"]
    assert review["biased_language"] == "NO"
    assert review["stereotyping"] == "NO"
    assert review["offensive_language"] == "NO"
    assert review["safety_assessment_consistency"] == "CONSISTENT"
    assert review["concerns_for_review"] == []
    assert isinstance(review["concerns_for_review"], list)

    # Verify version structure
    version = result_data["version"]
    assert "version" in version
    assert "models" in version
    assert "timestamp" in version
    assert isinstance(version["models"], list)
    assert len(version["models"]) > 0


def test_describe_uri_without_uri(client):
    """Test describe/uri endpoint without URI (should fail)."""
    payload = {
        "filename": "image.jpg",
        "mimetype": "image/jpeg"
    }

    response = client.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_describe_uri_with_invalid_mimetype(client):
    """Test describe/uri endpoint with invalid MIME type."""
    payload = {
        "uri": "https://example.com/document.pdf",
        "filename": "document.pdf",
        "mimetype": "application/pdf"
    }

    response = client.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


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


def test_describe_uri_without_scheme(client):
    """Test describe/uri endpoint with URI missing scheme."""
    payload = {
        "uri": "example.com/image.jpg",
        "filename": "image.jpg",
        "mimetype": "image/jpeg"
    }

    response = client.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "scheme" in data["detail"].lower()


def test_describe_uri_with_unsupported_scheme(client):
    """Test describe/uri endpoint with unsupported URI scheme."""
    payload = {
        "uri": "ftp://example.com/image.jpg",
        "filename": "image.jpg",
        "mimetype": "image/jpeg"
    }

    response = client.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "scheme" in data["detail"].lower()


def test_describe_uri_without_path(client):
    """Test describe/uri endpoint with URI missing path."""
    payload = {
        "uri": "https://example.com",
        "filename": "image.jpg",
        "mimetype": "image/jpeg"
    }

    response = client.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "path" in data["detail"].lower()

def test_describe_uri_with_slash_path(client):
    """Test describe/uri endpoint with URI missing path."""
    payload = {
        "uri": "https://example.com/",
        "filename": "image.jpg",
        "mimetype": "image/jpeg"
    }

    response = client.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "path" in data["detail"].lower()


def test_describe_uri_http_without_domain(client):
    """Test describe/uri endpoint with http URI missing domain."""
    payload = {
        "uri": "http:///image.jpg",
        "filename": "image.jpg",
        "mimetype": "image/jpeg"
    }

    response = client.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "domain" in data["detail"].lower()


def test_describe_uri_https_without_domain(client):
    """Test describe/uri endpoint with https URI missing domain."""
    payload = {
        "uri": "https:///image.jpg",
        "filename": "image.jpg",
        "mimetype": "image/jpeg"
    }

    response = client.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "domain" in data["detail"].lower()


def test_describe_uri_with_file_scheme(client):
    """Test describe/uri endpoint with file URI scheme."""
    payload = {
        "uri": "file:///path/to/image.jpg",
        "filename": "image.jpg",
        "mimetype": "image/jpeg"
    }

    response = client.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_200_OK

def test_describe_uri_with_file_scheme_with_no_path(client):
    """Test describe/uri endpoint with file URI scheme."""
    payload = {
        "uri": "file:///",
        "filename": "image.jpg",
        "mimetype": "image/jpeg"
    }

    response = client.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "path" in data["detail"].lower()

def test_describe_uri_with_file_scheme_with_only_protocol(client):
    """Test describe/uri endpoint with file URI scheme."""
    payload = {
        "uri": "file:/",
        "filename": "image.jpg",
        "mimetype": "image/jpeg"
    }

    response = client.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "path" in data["detail"].lower()

def test_describe_uri_with_http_scheme(client):
    """Test describe/uri endpoint with http URI scheme."""
    payload = {
        "uri": "http://example.com/image.jpg",
        "filename": "image.jpg",
        "mimetype": "image/jpeg"
    }

    response = client.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_200_OK
