"""Integration tests for authentication on describe endpoints.

These tests verify that authentication is properly enforced on all describe endpoints
and that both API key and HTTP Basic authentication work correctly.
"""
import io
import json
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from main import app
from app.config import Settings


@pytest.fixture
def auth_settings():
    """Create settings with authentication enabled."""
    return Settings(
        auth_enabled=True,
        api_keys="test-api-key-1,test-api-key-2",
        auth_username="testuser",
        auth_password="testpass123"
    )


@pytest.fixture
def client_with_auth(auth_settings):
    """Create a test client with authentication enabled."""
    # Override the settings
    from app import config
    original_settings = config.settings
    config.settings = auth_settings

    # Also need to update the settings reference in dependencies
    from app import dependencies
    dependencies.settings = auth_settings

    client = TestClient(app)

    yield client

    # Restore original settings
    config.settings = original_settings
    dependencies.settings = original_settings


@pytest.fixture
def blurry_owl_path():
    """Path to the real test image."""
    return Path(__file__).parent / "fixtures" / "blurry_owl.jpg"


@pytest.fixture
def blurry_owl_data(blurry_owl_path):
    """Read the real test image as bytes."""
    with open(blurry_owl_path, "rb") as f:
        return f.read()


@pytest.fixture
def mock_llm_responses():
    """Mock all LLM completion calls to avoid actual API calls."""
    def completion_side_effect(*args, **kwargs):
        response_format = kwargs.get("response_format", {})
        schema_name = ""
        if isinstance(response_format, dict):
            json_schema = response_format.get("json_schema", {})
            if isinstance(json_schema, dict):
                schema_name = json_schema.get("name", "")

        if schema_name == "image_analysis":
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json.dumps({
                "FULL_DESCRIPTION": "A test image description",
                "TRANSCRIPT": "No text visible.",
                "SAFETY_ASSESSMENT_FORM": {
                    "people_visible": "NO",
                    "demographics_described": "NO",
                    "misidentification_risk_people": "LOW",
                    "minors_present": "NO",
                    "named_individuals_claimed": "NO",
                    "violent_content": "NONE",
                    "racial_violence_oppression": "NONE",
                    "nudity": "NONE",
                    "sexual_content": "NONE",
                    "symbols_present": {
                        "types": ["NONE"],
                        "names": [],
                        "misidentification_risk": "LOW"
                    },
                    "stereotyping_present": "NO",
                    "atrocities_depicted": "NO",
                    "text_characteristics": {
                        "text_present": "NO",
                        "text_type": "N/A",
                        "legibility": "N/A"
                    },
                    "confidence": "MEDIUM"
                },
                "SAFETY_ASSESSMENT_REASONING": "Test reasoning"
            })
            return mock_response

        if schema_name == "review_assessment":
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json.dumps({
                "biased_language": "NO",
                "stereotyping": "NO",
                "value_judgments": "NO",
                "contradictions_between_texts": "NO",
                "contradictions_within_description": "NO",
                "offensive_language": "NO",
                "inconsistent_demographics": "NO",
                "euphemistic_language": "NO",
                "people_first_language": "N/A",
                "unsupported_inferential_claims": "NO",
                "safety_assessment_consistency": "CONSISTENT",
                "concerns_for_review": []
            })
            return mock_response

        if not response_format:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test alt text"
            return mock_response

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        return mock_response

    with patch("app.services.image_description_service.completion", side_effect=completion_side_effect), \
         patch("app.services.alt_text_generation_service.completion", side_effect=completion_side_effect), \
         patch("app.services.review_assessment_service.completion", side_effect=completion_side_effect):
        yield


# =============================================================================
# Upload Form Endpoint Tests
# =============================================================================

def test_upload_form_no_auth(client_with_auth):
    """Test that upload form endpoint requires authentication."""
    response = client_with_auth.get("/api/v1/describe/upload/form")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_upload_form_with_api_key(client_with_auth):
    """Test that upload form endpoint works with valid API key."""
    headers = {"X-API-Key": "test-api-key-1"}
    response = client_with_auth.get("/api/v1/describe/upload/form", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    assert "text/html" in response.headers["content-type"]


def test_upload_form_with_invalid_api_key(client_with_auth):
    """Test that upload form endpoint rejects invalid API key."""
    headers = {"X-API-Key": "invalid-key"}
    response = client_with_auth.get("/api/v1/describe/upload/form", headers=headers)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_upload_form_with_basic_auth(client_with_auth):
    """Test that upload form endpoint works with valid HTTP Basic auth."""
    response = client_with_auth.get(
        "/api/v1/describe/upload/form",
        auth=("testuser", "testpass123")
    )
    assert response.status_code == status.HTTP_200_OK
    assert "text/html" in response.headers["content-type"]


def test_upload_form_with_invalid_basic_auth(client_with_auth):
    """Test that upload form endpoint rejects invalid HTTP Basic auth."""
    response = client_with_auth.get(
        "/api/v1/describe/upload/form",
        auth=("testuser", "wrongpassword")
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_upload_form_with_second_api_key(client_with_auth):
    """Test that upload form works with the second configured API key."""
    headers = {"X-API-Key": "test-api-key-2"}
    response = client_with_auth.get("/api/v1/describe/upload/form", headers=headers)
    assert response.status_code == status.HTTP_200_OK


# =============================================================================
# Upload Endpoint Tests
# =============================================================================

def test_upload_no_auth(client_with_auth, blurry_owl_data):
    """Test that upload endpoint requires authentication."""
    files = {"file": ("test.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}
    response = client_with_auth.post("/api/v1/describe/upload", files=files)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_upload_with_api_key(client_with_auth, blurry_owl_data, mock_llm_responses):
    """Test that upload endpoint works with valid API key."""
    headers = {"X-API-Key": "test-api-key-1"}
    files = {"file": ("test.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}

    response = client_with_auth.post(
        "/api/v1/describe/upload",
        files=files,
        headers=headers
    )

    assert response.status_code == status.HTTP_200_OK
    result = response.json()
    assert result["success"] is True
    assert result["filename"] == "test.jpg"


def test_upload_with_invalid_api_key(client_with_auth, blurry_owl_data):
    """Test that upload endpoint rejects invalid API key."""
    headers = {"X-API-Key": "invalid-key"}
    files = {"file": ("test.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}

    response = client_with_auth.post(
        "/api/v1/describe/upload",
        files=files,
        headers=headers
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_upload_with_basic_auth(client_with_auth, blurry_owl_data, mock_llm_responses):
    """Test that upload endpoint works with valid HTTP Basic auth."""
    files = {"file": ("test.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}

    response = client_with_auth.post(
        "/api/v1/describe/upload",
        files=files,
        auth=("testuser", "testpass123")
    )

    assert response.status_code == status.HTTP_200_OK
    result = response.json()
    assert result["success"] is True
    assert result["filename"] == "test.jpg"


def test_upload_with_invalid_basic_auth(client_with_auth, blurry_owl_data):
    """Test that upload endpoint rejects invalid HTTP Basic auth."""
    files = {"file": ("test.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}

    response = client_with_auth.post(
        "/api/v1/describe/upload",
        files=files,
        auth=("testuser", "wrongpassword")
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_upload_with_context_and_api_key(client_with_auth, blurry_owl_data, mock_llm_responses):
    """Test that upload endpoint with context works with API key."""
    headers = {"X-API-Key": "test-api-key-1"}
    files = {"file": ("test.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}
    data = {"context": "Wildlife photography"}

    response = client_with_auth.post(
        "/api/v1/describe/upload",
        files=files,
        data=data,
        headers=headers
    )

    assert response.status_code == status.HTTP_200_OK
    result = response.json()
    assert result["success"] is True


# =============================================================================
# URI Endpoint Tests
# =============================================================================

def test_uri_no_auth(client_with_auth, blurry_owl_path):
    """Test that URI endpoint requires authentication."""
    payload = {
        "uri": f"file://{blurry_owl_path}",
        "filename": "test.jpg",
        "mimetype": "image/jpeg"
    }

    response = client_with_auth.post("/api/v1/describe/uri", json=payload)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_uri_with_api_key(client_with_auth, blurry_owl_path, mock_llm_responses):
    """Test that URI endpoint works with valid API key."""
    headers = {"X-API-Key": "test-api-key-1"}
    payload = {
        "uri": f"file://{blurry_owl_path}",
        "filename": "test.jpg",
        "mimetype": "image/jpeg"
    }

    response = client_with_auth.post(
        "/api/v1/describe/uri",
        json=payload,
        headers=headers
    )

    assert response.status_code == status.HTTP_200_OK
    result = response.json()
    assert result["success"] is True
    assert result["filename"] == "test.jpg"


def test_uri_with_invalid_api_key(client_with_auth, blurry_owl_path):
    """Test that URI endpoint rejects invalid API key."""
    headers = {"X-API-Key": "invalid-key"}
    payload = {
        "uri": f"file://{blurry_owl_path}",
        "filename": "test.jpg",
        "mimetype": "image/jpeg"
    }

    response = client_with_auth.post(
        "/api/v1/describe/uri",
        json=payload,
        headers=headers
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_uri_with_basic_auth(client_with_auth, blurry_owl_path, mock_llm_responses):
    """Test that URI endpoint works with valid HTTP Basic auth."""
    payload = {
        "uri": f"file://{blurry_owl_path}",
        "filename": "test.jpg",
        "mimetype": "image/jpeg"
    }

    response = client_with_auth.post(
        "/api/v1/describe/uri",
        json=payload,
        auth=("testuser", "testpass123")
    )

    assert response.status_code == status.HTTP_200_OK
    result = response.json()
    assert result["success"] is True
    assert result["filename"] == "test.jpg"


def test_uri_with_invalid_basic_auth(client_with_auth, blurry_owl_path):
    """Test that URI endpoint rejects invalid HTTP Basic auth."""
    payload = {
        "uri": f"file://{blurry_owl_path}",
        "filename": "test.jpg",
        "mimetype": "image/jpeg"
    }

    response = client_with_auth.post(
        "/api/v1/describe/uri",
        json=payload,
        auth=("testuser", "wrongpassword")
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_uri_with_context_and_basic_auth(client_with_auth, blurry_owl_path, mock_llm_responses):
    """Test that URI endpoint with context works with HTTP Basic auth."""
    payload = {
        "uri": f"file://{blurry_owl_path}",
        "filename": "test.jpg",
        "mimetype": "image/jpeg",
        "context": "Test context"
    }

    response = client_with_auth.post(
        "/api/v1/describe/uri",
        json=payload,
        auth=("testuser", "testpass123")
    )

    assert response.status_code == status.HTTP_200_OK
    result = response.json()
    assert result["success"] is True


# =============================================================================
# Edge Cases and Security Tests
# =============================================================================

def test_empty_api_key_header(client_with_auth, blurry_owl_data):
    """Test that empty API key header is rejected."""
    headers = {"X-API-Key": ""}
    files = {"file": ("test.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}

    response = client_with_auth.post(
        "/api/v1/describe/upload",
        files=files,
        headers=headers
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_wrong_api_key_header_name(client_with_auth, blurry_owl_data):
    """Test that wrong header name is rejected."""
    headers = {"Authorization": "Bearer test-api-key-1"}  # Wrong header
    files = {"file": ("test.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}

    response = client_with_auth.post(
        "/api/v1/describe/upload",
        files=files,
        headers=headers
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_empty_username_basic_auth(client_with_auth, blurry_owl_data):
    """Test that empty username in Basic auth is rejected."""
    files = {"file": ("test.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}

    response = client_with_auth.post(
        "/api/v1/describe/upload",
        files=files,
        auth=("", "testpass123")
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_empty_password_basic_auth(client_with_auth, blurry_owl_data):
    """Test that empty password in Basic auth is rejected."""
    files = {"file": ("test.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}

    response = client_with_auth.post(
        "/api/v1/describe/upload",
        files=files,
        auth=("testuser", "")
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_api_key_with_whitespace(client_with_auth, blurry_owl_data):
    """Test that API key with extra whitespace is rejected."""
    headers = {"X-API-Key": " test-api-key-1 "}  # Spaces around key
    files = {"file": ("test.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}

    response = client_with_auth.post(
        "/api/v1/describe/upload",
        files=files,
        headers=headers
    )

    # Should be rejected since our config trims keys but header value isn't trimmed
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_case_sensitive_api_key(client_with_auth, blurry_owl_data):
    """Test that API key is case-sensitive."""
    headers = {"X-API-Key": "TEST-API-KEY-1"}  # Wrong case
    files = {"file": ("test.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}

    response = client_with_auth.post(
        "/api/v1/describe/upload",
        files=files,
        headers=headers
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_both_auth_methods_api_key_valid(client_with_auth, blurry_owl_data, mock_llm_responses):
    """Test that when both auth methods provided, API key is checked first and works."""
    headers = {"X-API-Key": "test-api-key-1"}
    files = {"file": ("test.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}

    # Provide valid API key but invalid basic auth
    response = client_with_auth.post(
        "/api/v1/describe/upload",
        files=files,
        headers=headers,
        auth=("wronguser", "wrongpass")
    )

    # Should succeed because API key is valid and checked first
    assert response.status_code == status.HTTP_200_OK


def test_both_auth_methods_basic_auth_valid(client_with_auth, blurry_owl_data, mock_llm_responses):
    """Test that when API key invalid but basic auth valid, basic auth works."""
    headers = {"X-API-Key": "invalid-key"}
    files = {"file": ("test.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}

    # Provide invalid API key but valid basic auth
    response = client_with_auth.post(
        "/api/v1/describe/upload",
        files=files,
        headers=headers,
        auth=("testuser", "testpass123")
    )

    # Should succeed because basic auth is valid
    assert response.status_code == status.HTTP_200_OK


# =============================================================================
# Health Endpoint Tests
# =============================================================================

def test_health_check_with_auth_enabled(client_with_auth):
    """Test that health check requires authentication when auth is enabled."""
    response = client_with_auth.get("/health")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_health_check_with_api_key(client_with_auth):
    """Test that health check works with valid API key."""
    headers = {"X-API-Key": "test-api-key-1"}
    response = client_with_auth.get("/health", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    result = response.json()
    assert result["status"] == "healthy"


def test_health_check_with_basic_auth(client_with_auth):
    """Test that health check works with valid HTTP Basic auth."""
    response = client_with_auth.get("/health", auth=("testuser", "testpass123"))
    assert response.status_code == status.HTTP_200_OK
    result = response.json()
    assert result["status"] == "healthy"
