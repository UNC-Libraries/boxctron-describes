"""Unit tests for DescribeImageWorkflow, focused on the transcribe step."""
from pathlib import Path
from unittest.mock import Mock
import pytest

from app.config import Settings
from app.services.describe_image_workflow import DescribeImageWorkflow
from app.services.image_description_service import ImageDescriptionService
from app.services.image_normalizer import ImageNormalizer
from app.services.review_assessment_service import ReviewAssessmentService


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings():
    s = Settings()
    s.litellm_full_desc_model = "azure/gpt-4o"
    s.litellm_review_model = "azure/gpt-4o"
    s.litellm_transcribe_model = "gemini/gemini-1.5-pro"
    s.review_skip_threshold = None
    return s


@pytest.fixture
def mock_normalizer():
    normalizer = Mock(spec=ImageNormalizer)
    normalizer.normalize_image.return_value = "data:image/jpeg;base64,abc123"
    return normalizer


def _make_desc_result(text_present="NONE", legibility="N/A"):
    """Build a minimal full_desc_result dict."""
    return {
        "FULL_DESCRIPTION": "Test description",
        "ALT_TEXT": "Test alt text",
        "TRANSCRIPT": "Test transcript",
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
            "symbols_present": {"types": ["NONE"], "names": [], "misidentification_risk": "LOW"},
            "stereotyping_present": "NO",
            "atrocities_depicted": "NO",
            "text_characteristics": {
                "text_present": text_present,
                "text_type": "N/A",
                "legibility": legibility,
                "sensitivity": "N/A",
                "language": "N/A",
            },
            "image_quality": "UNIMPAIRED",
        },
        "SAFETY_ASSESSMENT_REASONING": "No concerns.",
    }


def _make_transcribe_result():
    """Build a distinct full_desc_result dict for the transcribe pass."""
    result = _make_desc_result(text_present="SIGNIFICANT", legibility="CLEAR")
    result["TRANSCRIPT"] = "Transcribed text from second pass"
    result["FULL_DESCRIPTION"] = "Better description from transcribe model"
    return result


@pytest.fixture
def mock_desc_service():
    svc = Mock(spec=ImageDescriptionService)
    svc.model = "azure/gpt-4o"
    svc.generate_description.return_value = _make_desc_result()
    return svc


@pytest.fixture
def mock_transcribe_service():
    svc = Mock(spec=ImageDescriptionService)
    svc.model = "gemini/gemini-1.5-pro"
    svc.generate_description.return_value = _make_transcribe_result()
    return svc


@pytest.fixture
def mock_review_service():
    svc = Mock(spec=ReviewAssessmentService)
    svc.generate_review_assessment.return_value = {
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
        "concerns_for_review": [],
        "source_content_warnings": [],
    }
    return svc


def _build_workflow(settings, normalizer, desc_service, review_service, transcribe_service=None):
    return DescribeImageWorkflow(settings, normalizer, desc_service, review_service, transcribe_service)


# ---------------------------------------------------------------------------
# Transcribe step triggers correctly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("legibility", ["PARTIALLY_CLEAR", "DIFFICULT", "ILLEGIBLE"])
@pytest.mark.asyncio
async def test_transcribe_step_runs_for_significant_difficult(
    legibility, settings, mock_normalizer, mock_desc_service, mock_review_service, mock_transcribe_service
):
    """Transcribe step runs when text_present=SIGNIFICANT and legibility is PARTIALLY_CLEAR, DIFFICULT, or ILLEGIBLE."""
    mock_desc_service.generate_description.return_value = _make_desc_result(
        text_present="SIGNIFICANT", legibility=legibility
    )

    workflow = _build_workflow(settings, mock_normalizer, mock_desc_service, mock_review_service, mock_transcribe_service)
    result = await workflow.process_image(Path("/tmp/img.jpg"), "img.jpg", "image/jpeg")

    mock_transcribe_service.generate_description.assert_called_once()

    assert result.steps["full_desc"].status == "superseded"
    assert result.steps["full_desc"].model == "azure/gpt-4o"
    assert result.steps["transcribe"].status == "success"
    assert result.steps["transcribe"].model == "gemini/gemini-1.5-pro"


@pytest.mark.asyncio
async def test_transcribe_result_replaces_first_pass(
    settings, mock_normalizer, mock_desc_service, mock_review_service, mock_transcribe_service
):
    """Result fields come from the transcribe pass, not the first pass."""
    mock_desc_service.generate_description.return_value = _make_desc_result(
        text_present="SIGNIFICANT", legibility="DIFFICULT"
    )
    mock_transcribe_service.generate_description.return_value = _make_transcribe_result()

    workflow = _build_workflow(settings, mock_normalizer, mock_desc_service, mock_review_service, mock_transcribe_service)
    result = await workflow.process_image(Path("/tmp/img.jpg"), "img.jpg", "image/jpeg")

    assert result.transcript == "Transcribed text from second pass"
    assert result.full_description == "Better description from transcribe model"


# ---------------------------------------------------------------------------
# Transcribe step does NOT trigger
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transcribe_step_skipped_when_no_transcribe_service(
    settings, mock_normalizer, mock_desc_service, mock_review_service
):
    """Transcribe step is absent when no transcribe_service is provided."""
    mock_desc_service.generate_description.return_value = _make_desc_result(
        text_present="SIGNIFICANT", legibility="DIFFICULT"
    )

    workflow = _build_workflow(settings, mock_normalizer, mock_desc_service, mock_review_service, transcribe_service=None)
    result = await workflow.process_image(Path("/tmp/img.jpg"), "img.jpg", "image/jpeg")

    assert result.steps["full_desc"].status == "success"
    assert "transcribe" not in result.steps


@pytest.mark.asyncio
async def test_transcribe_step_skipped_when_text_not_significant(
    settings, mock_normalizer, mock_desc_service, mock_review_service, mock_transcribe_service
):
    """Transcribe step is absent when text_present is INCIDENTAL (not SIGNIFICANT)."""
    mock_desc_service.generate_description.return_value = _make_desc_result(
        text_present="INCIDENTAL", legibility="DIFFICULT"
    )

    workflow = _build_workflow(settings, mock_normalizer, mock_desc_service, mock_review_service, mock_transcribe_service)
    result = await workflow.process_image(Path("/tmp/img.jpg"), "img.jpg", "image/jpeg")

    mock_transcribe_service.generate_description.assert_not_called()
    assert result.steps["full_desc"].status == "success"
    assert "transcribe" not in result.steps


@pytest.mark.asyncio
async def test_transcribe_step_skipped_when_text_is_none(
    settings, mock_normalizer, mock_desc_service, mock_review_service, mock_transcribe_service
):
    """Transcribe step is absent when there is no text."""
    mock_desc_service.generate_description.return_value = _make_desc_result(
        text_present="NONE", legibility="N/A"
    )

    workflow = _build_workflow(settings, mock_normalizer, mock_desc_service, mock_review_service, mock_transcribe_service)
    result = await workflow.process_image(Path("/tmp/img.jpg"), "img.jpg", "image/jpeg")

    mock_transcribe_service.generate_description.assert_not_called()
    assert "transcribe" not in result.steps


@pytest.mark.parametrize("legibility", ["CLEAR", "N/A"])
@pytest.mark.asyncio
async def test_transcribe_step_skipped_when_legibility_is_clear(
    legibility, settings, mock_normalizer, mock_desc_service, mock_review_service, mock_transcribe_service
):
    """Transcribe step is absent when text is SIGNIFICANT but legibility is CLEAR or N/A."""
    mock_desc_service.generate_description.return_value = _make_desc_result(
        text_present="SIGNIFICANT", legibility=legibility
    )

    workflow = _build_workflow(settings, mock_normalizer, mock_desc_service, mock_review_service, mock_transcribe_service)
    result = await workflow.process_image(Path("/tmp/img.jpg"), "img.jpg", "image/jpeg")

    mock_transcribe_service.generate_description.assert_not_called()
    assert result.steps["full_desc"].status == "success"
    assert "transcribe" not in result.steps


# ---------------------------------------------------------------------------
# Normal path (no transcribe) produces a single full_desc step
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_normal_path_has_single_full_desc_step(
    settings, mock_normalizer, mock_desc_service, mock_review_service
):
    """Without transcribe, steps has exactly one full_desc entry with status success."""
    workflow = _build_workflow(settings, mock_normalizer, mock_desc_service, mock_review_service)
    result = await workflow.process_image(Path("/tmp/img.jpg"), "img.jpg", "image/jpeg")

    assert result.steps["full_desc"].status == "success"
    assert result.steps["full_desc"].model == "azure/gpt-4o"
    assert "transcribe" not in result.steps
