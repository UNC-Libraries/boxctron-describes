"""Integration tests for the describe endpoints.

These tests use real code throughout the stack, only mocking LLM API calls
to avoid costs. All image processing, file handling, and service orchestration
use real implementations.
"""
import io
import json
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import respx
import httpx
from fastapi import status
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """Create a test client for integration tests."""
    return TestClient(app)


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
    """
    Mock all LLM completion calls with realistic responses.

    Patches completion at the import locations in each service module
    to avoid actual API calls.
    """
    def completion_side_effect(*args, **kwargs):
        # Check response_format to distinguish between services
        response_format = kwargs.get("response_format", {})
        schema_name = ""
        if isinstance(response_format, dict):
            json_schema = response_format.get("json_schema", {})
            if isinstance(json_schema, dict):
                schema_name = json_schema.get("name", "")

        # Image description generation (has schema name "image_analysis")
        if schema_name == "image_analysis":
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json.dumps({
                "FULL_DESCRIPTION": "A detailed photograph showing a blurry owl perched on a branch in low light conditions. The owl appears to be a species with mottled brown and white plumage.",
                "TRANSCRIPT": "No visible text in image.",
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
                "SAFETY_ASSESSMENT_REASONING": "This is a nature photograph with no concerning content. The blurriness affects image quality but not safety assessment."
            })
            return mock_response

        # Review assessment (has schema name "review_assessment")
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

        # Alt text generation (no response_format, returns plain text)
        if not response_format:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "A blurry owl perched on a branch in low light"
            return mock_response

        # Default fallback
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        return mock_response

    # Patch at the import locations in each service module
    with patch("app.services.image_description_service.completion", side_effect=completion_side_effect) as mock1, \
         patch("app.services.alt_text_generation_service.completion", side_effect=completion_side_effect) as mock2, \
         patch("app.services.review_assessment_service.completion", side_effect=completion_side_effect) as mock3:
        # Yield all three mocks so tests can check individual or total call counts
        yield (mock1, mock2, mock3)


def test_integration_upload_real_image(client, blurry_owl_data, mock_llm_responses):
    """
    Integration test: Upload a real image file and process through full workflow.

    Tests:
    - File upload handling
    - Image normalization with Pillow
    - Full workflow orchestration
    - All services working together
    """
    files = {"file": ("blurry_owl.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}
    data = {
        "filename": "blurry_owl.jpg",
        "mimetype": "image/jpeg",
        "context": "Wildlife photography"
    }

    response = client.post("/api/v1/describe/upload", files=files, data=data)

    assert response.status_code == status.HTTP_200_OK

    result = response.json()
    assert result["success"] is True
    assert result["filename"] == "blurry_owl.jpg"
    assert result["result"] is not None

    # Verify complete result structure with real data
    result_data = result["result"]
    assert "blurry owl" in result_data["full_description"].lower()
    assert "blurry owl" in result_data["alt_text"].lower()
    assert result_data["transcript"] == "No visible text in image."

    # Verify safety assessment was processed
    safety = result_data["safety_assessment"]
    assert safety["people_visible"] == "NO"
    assert safety["confidence"] == "MEDIUM"

    # Verify review assessment was processed
    review = result_data["review_assessment"]
    assert review["safety_assessment_consistency"] == "CONSISTENT"
    assert isinstance(review["concerns_for_review"], list)

    # Verify version info
    version = result_data["version"]
    assert "version" in version
    assert "timestamp" in version

    # Verify LLM mocks were called
    image_desc_llm_mock, alt_text_llm_mock, review_llm_mock = mock_llm_responses
    assert image_desc_llm_mock.call_count == 1
    assert alt_text_llm_mock.call_count == 1
    assert review_llm_mock.call_count == 1


def test_integration_file_uri(client, blurry_owl_path, mock_llm_responses):
    """
    Integration test: Process image from file:// URI.

    Tests:
    - File URI handling
    - Local file access
    - Same full workflow as upload
    """
    payload = {
        "uri": f"file://{blurry_owl_path}",
        "filename": "blurry_owl.jpg",
        "mimetype": "image/jpeg",
        "context": "Local test image"
    }

    response = client.post("/api/v1/describe/uri", json=payload)

    assert response.status_code == status.HTTP_200_OK

    result = response.json()
    assert result["success"] is True
    assert "blurry owl" in result["result"]["full_description"].lower()
    assert "blurry owl" in result["result"]["alt_text"].lower()

    # Verify LLM mocks were called
    image_desc_llm_mock, alt_text_llm_mock, review_llm_mock = mock_llm_responses
    assert image_desc_llm_mock.call_count == 1
    assert alt_text_llm_mock.call_count == 1
    assert review_llm_mock.call_count == 1


@respx.mock
def test_integration_http_uri(client, blurry_owl_data, mock_llm_responses):
    """
    Integration test: Download image from HTTP URI and process.

    Tests:
    - HTTP download with httpx
    - respx mocking for external requests
    - Full processing pipeline
    """
    # Mock the HTTP download
    respx.get("https://example.com/owl.jpg").mock(
        return_value=httpx.Response(200, content=blurry_owl_data)
    )

    payload = {
        "uri": "https://example.com/owl.jpg",
        "filename": "owl.jpg",
        "mimetype": "image/jpeg",
        "context": "Downloaded wildlife photo"
    }

    response = client.post("/api/v1/describe/uri", json=payload)

    assert response.status_code == status.HTTP_200_OK

    result = response.json()
    assert result["success"] is True
    assert result["filename"] == "owl.jpg"

    # Verify the image was actually processed
    assert "owl" in result["result"]["full_description"].lower()
    assert len(result["result"]["alt_text"]) > 0

    # Verify HTTP download happened
    assert respx.calls.call_count == 1

    # Verify LLM mocks were called
    image_desc_llm_mock, alt_text_llm_mock, review_llm_mock = mock_llm_responses
    assert image_desc_llm_mock.call_count == 1
    assert alt_text_llm_mock.call_count == 1
    assert review_llm_mock.call_count == 1


def test_integration_no_context(client, blurry_owl_data, mock_llm_responses):
    """
    Integration test: Process image without optional context.

    Tests:
    - Workflow works without context parameter
    - All services handle None context appropriately
    """
    files = {"file": ("owl.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}
    data = {
        "filename": "owl.jpg",
        "mimetype": "image/jpeg"
        # No context provided
    }

    response = client.post("/api/v1/describe/upload", files=files, data=data)

    assert response.status_code == status.HTTP_200_OK
    result = response.json()
    assert result["success"] is True

    # Should still produce full results
    assert len(result["result"]["full_description"]) > 0
    assert len(result["result"]["alt_text"]) > 0


def test_integration_large_context(client, blurry_owl_data, mock_llm_responses):
    """
    Integration test: Process with substantial context information.

    Tests:
    - Context is properly passed through workflow
    - Large context strings are handled
    """
    large_context = (
        "This is a wildlife photograph taken during a nocturnal bird survey "
        "in the South east. The image was captured using a trail camera "
        "with infrared capabilities. The owl species is believed to be a Northern "
        "Spotted Owl, which is a threatened species. This image is part of a "
        "conservation monitoring project documenting owl populations in old-growth forests."
    )

    files = {"file": ("owl.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}
    data = {
        "filename": "owl.jpg",
        "mimetype": "image/jpeg",
        "context": large_context
    }

    response = client.post("/api/v1/describe/upload", files=files, data=data)

    assert response.status_code == status.HTTP_200_OK
    result = response.json()
    assert result["success"] is True


@respx.mock
def test_integration_http_download_failure(client, mock_llm_responses):
    """
    Integration test: Handle HTTP download failures gracefully.

    Tests:
    - Error handling for failed downloads
    - Appropriate error responses
    """
    # Mock a failed HTTP response
    respx.get("https://example.com/missing.jpg").mock(
        return_value=httpx.Response(404)
    )

    payload = {
        "uri": "https://example.com/missing.jpg",
        "filename": "missing.jpg",
        "mimetype": "image/jpeg"
    }

    response = client.post("/api/v1/describe/uri", json=payload)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "error" in response.json()["detail"].lower() or "failed" in response.json()["detail"].lower()


def test_integration_invalid_image_format(client, mock_llm_responses):
    """
    Integration test: Handle corrupted/invalid image data.

    Tests:
    - Pillow error handling
    - Graceful failure for bad image data
    """
    # Create invalid image data
    invalid_data = b"This is not a valid image file"

    files = {"file": ("fake.jpg", io.BytesIO(invalid_data), "image/jpeg")}
    data = {
        "filename": "fake.jpg",
        "mimetype": "image/jpeg"
    }

    response = client.post("/api/v1/describe/upload", files=files, data=data)

    # Should return error (400 or 500 depending on where error is caught)
    print(f"STATUS WAS: {response.status_code}")
    assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_500_INTERNAL_SERVER_ERROR]


def test_integration_all_result_fields_populated(client, blurry_owl_data, mock_llm_responses):
    """
    Integration test: Verify all expected fields are populated in result.

    Tests:
    - Complete DescriptionResult structure
    - All nested objects present
    - No None/null values where data should exist
    """
    files = {"file": ("owl.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}
    data = {
        "filename": "owl.jpg",
        "mimetype": "image/jpeg",
        "context": "Test"
    }

    response = client.post("/api/v1/describe/upload", files=files, data=data)

    assert response.status_code == status.HTTP_200_OK
    result = response.json()["result"]

    # Top-level fields
    assert result["full_description"] is not None
    assert len(result["full_description"]) > 0
    assert result["alt_text"] is not None
    assert len(result["alt_text"]) > 0
    assert result["transcript"] is not None

    # Safety assessment fields
    safety = result["safety_assessment"]
    assert safety["people_visible"] in ["YES", "NO"]
    assert safety["confidence"] in ["LOW", "MEDIUM", "HIGH"]
    assert safety["reasoning"] is not None
    assert isinstance(safety["symbols_present"]["types"], list)
    assert isinstance(safety["symbols_present"]["names"], list)
    assert safety["text_characteristics"]["text_present"] in ["YES", "NO"]

    # Review assessment fields
    review = result["review_assessment"]
    assert review["biased_language"] in ["NO", "POSSIBLY", "YES"]
    assert review["safety_assessment_consistency"] in ["CONSISTENT", "INCONSISTENT"]
    assert isinstance(review["concerns_for_review"], list)

    # Version info
    version = result["version"]
    assert version["version"] is not None
    assert isinstance(version["models"], list)
    assert len(version["models"]) > 0
    assert version["timestamp"] is not None


def test_integration_file_cleanup(client, blurry_owl_data, mock_llm_responses, tmp_path):
    """
    Integration test: Verify temporary files are cleaned up.

    Tests:
    - Temp files created during processing are deleted
    - No file leaks on success or failure
    """
    import tempfile
    original_tempdir = tempfile.gettempdir()
    temp_files_before = set(Path(original_tempdir).glob("tmp*"))

    files = {"file": ("owl.jpg", io.BytesIO(blurry_owl_data), "image/jpeg")}
    data = {
        "filename": "owl.jpg",
        "mimetype": "image/jpeg"
    }

    response = client.post("/api/v1/describe/upload", files=files, data=data)
    assert response.status_code == status.HTTP_200_OK

    # Check that no new temp files are left behind
    temp_files_after = set(Path(original_tempdir).glob("tmp*"))
    new_files = temp_files_after - temp_files_before

    # Some system temp files might exist, but shouldn't be from our process
    # This is a basic check - in production you'd track specific file handles
    assert len(new_files) == 0 or all(not f.name.startswith("tmp") for f in new_files)
