"""Tests for the ReviewAssessmentService."""
import json
import pytest
from unittest.mock import Mock, patch

from app.services.review_assessment_service import ReviewAssessmentService
from app.config import Settings


@pytest.fixture
def mock_settings():
    """Provide mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.litellm_review_model = "azure/gpt-4o"
    settings.litellm_review_temperature = 0.3
    settings.litellm_review_max_tokens = 500
    settings.litellm_review_reasoning_effort = "low"
    settings.litellm_num_retries = 3
    return settings


@pytest.fixture
def service(mock_settings):
    """Provide a ReviewAssessmentService instance."""
    return ReviewAssessmentService(mock_settings)


@pytest.fixture
def sample_safety_assessment():
    """Provide sample safety assessment data."""
    return {
        "people_visible": "YES",
        "demographics_described": "YES",
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
        }
    }


def test_init_loads_prompt_and_system_prompt(mock_settings):
    """Test that __init__ loads the prompt template correctly."""
    service = ReviewAssessmentService(mock_settings)

    assert service.settings == mock_settings
    assert len(service.review_prompt) > 0
    assert "content quality reviewer" in service.system_prompt
    assert "accessibility and bias detection" in service.system_prompt


def test_format_content_for_review(service, sample_safety_assessment):
    """Test that content is properly formatted with labels."""
    full_description = "A detailed description of an image"
    transcript = "Text visible in the image"
    safety_reasoning = "The image appears safe"
    alt_text = "A person standing in a field"

    result = service._format_content_for_review(
        full_description,
        transcript,
        sample_safety_assessment,
        safety_reasoning,
        alt_text
    )

    # Verify all labels are present
    assert "FULL_DESCRIPTION:" in result
    assert "TRANSCRIPT:" in result
    assert "SAFETY_ASSESSMENT_FORM:" in result
    assert "SAFETY_ASSESSMENT_REASONING:" in result
    assert "ALT_TEXT:" in result

    # Verify content is present
    assert full_description in result
    assert transcript in result
    assert safety_reasoning in result
    assert alt_text in result

    # Verify safety assessment is formatted as JSON
    assert '"people_visible": "YES"' in result


@patch("app.services.review_assessment_service.completion")
def test_generate_review_assessment_success(mock_completion, service, mock_settings, sample_safety_assessment):
    """Test successful review assessment generation."""
    # Mock LLM response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps({
        "bias": "N",
        "stereo": "N",
        "val_judg": "N",
        "contra_btwn": "N",
        "contra_within": "N",
        "offensive": "N",
        "incon_demog": "N",
        "euphemism": "N",
        "ppl_first": "NA",
        "unsup_infer": "N",
        "safety_consist": "CON",
        "concerns": []
    })
    mock_completion.return_value = mock_response

    result = service.generate_review_assessment(
        full_description="A person standing in a field",
        transcript="No text visible",
        safety_assessment=sample_safety_assessment,
        safety_assessment_reasoning="Image appears safe",
        alt_text="Person in field"
    )

    # Verify result has expanded keys/values
    assert result["biased_language"] == "NO"
    assert result["stereotyping"] == "NO"
    assert result["safety_assessment_consistency"] == "CONSISTENT"
    assert result["concerns_for_review"] == []

    # Verify completion was called with correct parameters
    mock_completion.assert_called_once()
    call_args = mock_completion.call_args[1]

    assert call_args["model"] == mock_settings.litellm_review_model
    assert call_args["temperature"] == mock_settings.litellm_review_temperature
    assert call_args["max_tokens"] == mock_settings.litellm_review_max_tokens
    assert call_args["reasoning_effort"] == mock_settings.litellm_review_reasoning_effort
    assert call_args["num_retries"] == mock_settings.litellm_num_retries
    assert "response_format" in call_args

    # Verify messages structure
    messages = call_args["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "content quality reviewer" in messages[0]["content"]
    assert messages[1]["role"] == "user"


@patch("app.services.review_assessment_service.completion")
def test_generate_review_assessment_with_concerns(mock_completion, service, sample_safety_assessment):
    """Test review assessment with concerns identified."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps({
        "bias": "P",
        "stereo": "Y",
        "val_judg": "N",
        "contra_btwn": "N",
        "contra_within": "N",
        "offensive": "N",
        "incon_demog": "N",
        "euphemism": "N",
        "ppl_first": "NU",
        "unsup_infer": "P",
        "safety_consist": "INCON",
        "concerns": ["Possible stereotyping in description", "Safety assessment inconsistent with content"]
    })
    mock_completion.return_value = mock_response

    result = service.generate_review_assessment(
        full_description="A description",
        transcript="Text",
        safety_assessment=sample_safety_assessment,
        safety_assessment_reasoning="Reasoning",
        alt_text="Alt text"
    )

    assert result["biased_language"] == "POSSIBLY"
    assert result["stereotyping"] == "YES"
    assert result["people_first_language"] == "NOT_USED"
    assert result["safety_assessment_consistency"] == "INCONSISTENT"
    assert len(result["concerns_for_review"]) == 2
    assert "stereotyping" in result["concerns_for_review"][0].lower()


@patch("app.services.review_assessment_service.completion")
def test_generate_review_assessment_empty_response(mock_completion, service, sample_safety_assessment):
    """Test handling of empty response from LLM."""
    mock_response = Mock()
    mock_response.choices = []
    mock_completion.return_value = mock_response

    with pytest.raises(ValueError, match="Empty response from LLM"):
        service.generate_review_assessment(
            full_description="Test",
            transcript="Test",
            safety_assessment=sample_safety_assessment,
            safety_assessment_reasoning="Test",
            alt_text="Test"
        )


@patch("app.services.review_assessment_service.completion")
def test_generate_review_assessment_no_content(mock_completion, service, sample_safety_assessment):
    """Test handling of response with no content."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = None
    mock_completion.return_value = mock_response

    with pytest.raises(ValueError, match="Empty response from LLM"):
        service.generate_review_assessment(
            full_description="Test",
            transcript="Test",
            safety_assessment=sample_safety_assessment,
            safety_assessment_reasoning="Test",
            alt_text="Test"
        )


@patch("app.services.review_assessment_service.completion")
def test_generate_review_assessment_invalid_json(mock_completion, service, sample_safety_assessment):
    """Test handling of invalid JSON response."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Not valid JSON"
    mock_completion.return_value = mock_response

    with pytest.raises(json.JSONDecodeError):
        service.generate_review_assessment(
            full_description="Test",
            transcript="Test",
            safety_assessment=sample_safety_assessment,
            safety_assessment_reasoning="Test",
            alt_text="Test"
        )


@patch("app.services.review_assessment_service.completion")
def test_generate_review_assessment_missing_required_field(mock_completion, service, sample_safety_assessment):
    """Test validation of missing required fields."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    # Missing 'concerns' field
    mock_response.choices[0].message.content = json.dumps({
        "bias": "N",
        "stereo": "N",
        "val_judg": "N",
        "contra_btwn": "N",
        "contra_within": "N",
        "offensive": "N",
        "incon_demog": "N",
        "euphemism": "N",
        "ppl_first": "NA",
        "unsup_infer": "N",
        "safety_consist": "CON"
    })
    mock_completion.return_value = mock_response

    with pytest.raises(ValueError, match="Missing required field"):
        service.generate_review_assessment(
            full_description="Test",
            transcript="Test",
            safety_assessment=sample_safety_assessment,
            safety_assessment_reasoning="Test",
            alt_text="Test"
        )


@patch("app.services.review_assessment_service.completion")
def test_generate_review_assessment_llm_error(mock_completion, service, sample_safety_assessment):
    """Test handling of LLM API errors."""
    mock_completion.side_effect = Exception("LLM API error")

    with pytest.raises(Exception, match="LLM API error"):
        service.generate_review_assessment(
            full_description="Test",
            transcript="Test",
            safety_assessment=sample_safety_assessment,
            safety_assessment_reasoning="Test",
            alt_text="Test"
        )


def test_get_response_format(service):
    """Test that response format schema is correctly defined."""
    response_format = service._get_response_format()

    assert response_format["type"] == "json_schema"
    assert "json_schema" in response_format

    schema = response_format["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "required" in schema

    # Verify all required fields are in the schema (abbreviated keys)
    required_fields = schema["required"]
    assert "bias" in required_fields
    assert "stereo" in required_fields
    assert "safety_consist" in required_fields
    assert "concerns" in required_fields
    assert len(required_fields) == 12

    # Verify enum constraints (abbreviated values)
    assert schema["properties"]["bias"]["enum"] == ["N", "P", "Y"]
    assert schema["properties"]["ppl_first"]["enum"] == ["U", "NU", "NA"]
    assert schema["properties"]["safety_consist"]["enum"] == ["CON", "INCON"]

    # Verify concerns is an array
    assert schema["properties"]["concerns"]["type"] == "array"
    assert schema["properties"]["concerns"]["items"]["type"] == "string"


@patch("app.services.review_assessment_service.completion")
def test_retries_on_invalid_response_then_succeeds(mock_completion, service, sample_safety_assessment):
    """Test that a transient parse error triggers a retry and eventual success is returned."""
    bad_response = Mock()
    bad_response.choices = [Mock()]
    bad_response.choices[0].message.content = "not valid json"

    good_response = Mock()
    good_response.choices = [Mock()]
    good_response.choices[0].message.content = json.dumps({
        "bias": "N", "stereo": "N", "val_judg": "N",
        "contra_btwn": "N", "contra_within": "N", "offensive": "N",
        "incon_demog": "N", "euphemism": "N", "ppl_first": "NA",
        "unsup_infer": "N", "safety_consist": "CON", "concerns": []
    })

    mock_completion.side_effect = [bad_response, bad_response, good_response]

    result = service.generate_review_assessment(
        full_description="Test", transcript="Test",
        safety_assessment=sample_safety_assessment,
        safety_assessment_reasoning="Test", alt_text="Test"
    )

    assert mock_completion.call_count == 3
    assert result["biased_language"] == "NO"
    assert result["safety_assessment_consistency"] == "CONSISTENT"


@patch("app.services.review_assessment_service.completion")
def test_all_retries_exhausted_raises_error(mock_completion, service, sample_safety_assessment):
    """Test that after MAX_PARSE_RETRIES attempts the error is re-raised."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = None
    mock_completion.return_value = mock_response

    with pytest.raises(ValueError, match="Empty response from LLM"):
        service.generate_review_assessment(
            full_description="Test", transcript="Test",
            safety_assessment=sample_safety_assessment,
            safety_assessment_reasoning="Test", alt_text="Test"
        )

    assert mock_completion.call_count == ReviewAssessmentService._MAX_PARSE_RETRIES


@patch("app.services.review_assessment_service.completion")
def test_non_parse_error_is_not_retried(mock_completion, service, sample_safety_assessment):
    """Test that non-ValueError/JSONDecodeError exceptions are not retried."""
    mock_completion.side_effect = RuntimeError("Network error")

    with pytest.raises(RuntimeError, match="Network error"):
        service.generate_review_assessment(
            full_description="Test", transcript="Test",
            safety_assessment=sample_safety_assessment,
            safety_assessment_reasoning="Test", alt_text="Test"
        )

    assert mock_completion.call_count == 1
