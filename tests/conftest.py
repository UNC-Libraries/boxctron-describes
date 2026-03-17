"""Test configuration and fixtures."""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

# Disable authentication for all tests in this conftest by default. Override in tests that need it
from app.config import settings as _app_settings
_app_settings.auth_enabled = False

from main import app
from app.dependencies import get_describe_workflow
from app.models import (
    DescriptionResult,
    SafetyAssessment,
    ReviewAssessment,
    SymbolsPresent,
    TextCharacteristics,
    VersionInfo
)


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


@pytest.fixture
def mock_description_result():
    """Provide a mock DescriptionResult for testing."""
    return DescriptionResult(
        full_description="A test image description",
        alt_text="Test alt text",
        transcript="Test transcript",
        safety_assessment=SafetyAssessment(
            people_visible="NO",
            demographics_described="NO",
            misidentification_risk_people="LOW",
            minors_present="NO",
            named_individuals_claimed="NO",
            violent_content="NONE",
            racial_violence_oppression="NONE",
            nudity="NONE",
            sexual_content="NONE",
            symbols_present=SymbolsPresent(
                types=["NONE"],
                names=[],
                misidentification_risk="LOW"
            ),
            stereotyping_present="NO",
            atrocities_depicted="NO",
            text_characteristics=TextCharacteristics(
                text_present="NO",
                text_type="N/A",
                legibility="N/A",
                text_sensitivity="N/A"
            ),
            reasoning="Test reasoning",
            risk_score=0,
            inconsistency_count=0
        ),
        review_assessment=ReviewAssessment(
            biased_language="NO",
            stereotyping="NO",
            value_judgments="NO",
            contradictions_between_texts="NO",
            contradictions_within_description="NO",
            offensive_language="NO",
            inconsistent_demographics="NO",
            euphemistic_language="NO",
            people_first_language="N/A",
            unsupported_inferential_claims="NO",
            safety_assessment_consistency="CONSISTENT",
            concerns_for_review=[],
            source_content_warnings=[],
            risk_score=0
        ),
        overall_risk_Score=0,
        version=VersionInfo(
            version="0.1.0",
            models={
                "full_desc": "test-model",
                "alt_text": "test-model",
                "review": "test-model"
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    )


@pytest.fixture
def mock_workflow(mock_description_result):
    """
    Provide a mocked workflow for testing route handlers.

    Uses FastAPI's dependency_overrides to replace the real workflow
    with a mock that returns test data.

    Returns the mock workflow instance so tests can assert on calls if needed.
    """
    mock_workflow_instance = AsyncMock()
    mock_workflow_instance.process_image = AsyncMock(return_value=mock_description_result)

    # Override the dependency
    app.dependency_overrides[get_describe_workflow] = lambda: mock_workflow_instance

    yield mock_workflow_instance

    # Clean up the override after the test
    app.dependency_overrides.clear()