"""Tests for ImageDescriptionService."""
from unittest.mock import Mock, patch
import json
import pytest

from app.services.image_description_service import ImageDescriptionService
from app.config import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Settings()
    settings.litellm_full_desc_model = "azure/gpt-4o"
    settings.litellm_full_desc_temperature = 0.7
    settings.litellm_full_desc_max_tokens = 1000
    settings.litellm_full_desc_reasoning_effort = "low"
    settings.litellm_num_retries = 3
    return settings

@pytest.fixture
def sample_llm_response():
    """Sample valid LLM response (abbreviated form as produced by the LLM)."""
    return {
        "FULL_DESCRIPTION": "A test image description",
        "TRANSCRIPT": "Test transcript text",
        "SAF": {
            "people": "N",
            "demog": "N",
            "misid_risk": "L",
            "minors": "N",
            "named_indiv": "N",
            "violence": "0",
            "racial_viol": "0",
            "nudity": "0",
            "sexual": "0",
            "symbols": {
                "types": ["0"],
                "names": [],
                "misid_risk": "L"
            },
            "stereotyping": "N",
            "atrocities": "N",
            "text_chars": {
                "present": "N",
                "type": "NA",
                "legib": "NA"
            }
        },
        "SAR": "No safety concerns detected."
    }


@patch("app.services.image_description_service.completion")
def test_generate_description_without_context(mock_completion, mock_settings, sample_llm_response):
    """Test generating description without context."""
    # Setup mock response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps(sample_llm_response)
    mock_completion.return_value = mock_response

    # Create service and call
    service = ImageDescriptionService(mock_settings)
    result = service.generate_description("data:image/jpeg;base64,abc123")

    # Verify completion was called with correct parameters
    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args[1]

    assert call_kwargs["model"] == "azure/gpt-4o"
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 1000
    assert call_kwargs["num_retries"] == 3
    assert call_kwargs["reasoning_effort"] == "low"
    assert "response_format" in call_kwargs

    # Verify messages structure
    messages = call_kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    # Verify image is in user content
    user_content = messages[1]["content"]
    assert len(user_content) == 2  # task prompt + image
    assert user_content[0]["type"] == "text"  # task prompt
    assert "Analyze this image" in user_content[0]["text"]  # verify it's the task prompt
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"] == "data:image/jpeg;base64,abc123"

    # Verify result has expanded safety form keys/values
    assert result["FULL_DESCRIPTION"] == sample_llm_response["FULL_DESCRIPTION"]
    assert result["TRANSCRIPT"] == sample_llm_response["TRANSCRIPT"]
    assert result["SAFETY_ASSESSMENT_REASONING"] == sample_llm_response["SAR"]
    safety = result["SAFETY_ASSESSMENT_FORM"]
    assert safety["people_visible"] == "NO"
    assert safety["violent_content"] == "NONE"
    assert safety["symbols_present"]["types"] == ["NONE"]


@patch("app.services.image_description_service.completion")
def test_generate_description_with_context(mock_completion, mock_settings, sample_llm_response):
    """Test generating description with context."""
    # Setup mock response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps(sample_llm_response)
    mock_completion.return_value = mock_response

    # Create service and call
    service = ImageDescriptionService(mock_settings)
    result = service.generate_description(
        "data:image/jpeg;base64,abc123",
        context="This is a historical photograph"
    )

    # Verify completion was called
    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args[1]

    # Verify messages include context
    messages = call_kwargs["messages"]
    user_content = messages[1]["content"]

    # Should have 3 items: task prompt, context text, and image
    assert len(user_content) == 3
    assert user_content[0]["type"] == "text"  # task prompt
    assert "Analyze this image" in user_content[0]["text"]
    assert user_content[1]["type"] == "text"  # context
    assert "This is a historical photograph" in user_content[1]["text"]
    assert user_content[2]["type"] == "image_url"


@patch("app.services.image_description_service.completion")
def test_reasoning_effort_not_set_when_none(mock_completion, mock_settings, sample_llm_response):
    """Test that reasoning_effort is omitted when set to None."""
    # Setup mock response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps(sample_llm_response)
    mock_completion.return_value = mock_response

    # Set reasoning_effort to None
    mock_settings.litellm_full_desc_reasoning_effort = None

    # Create service and call
    service = ImageDescriptionService(mock_settings)
    service.generate_description("data:image/jpeg;base64,abc123")

    # Verify reasoning_effort is not in call parameters
    call_kwargs = mock_completion.call_args[1]
    assert "reasoning_effort" not in call_kwargs


@patch("app.services.image_description_service.completion")
def test_empty_response_raises_error(mock_completion, mock_settings):
    """Test that empty LLM response raises ValueError."""
    # Setup mock response with no content
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = None
    mock_completion.return_value = mock_response

    # Create service and call
    service = ImageDescriptionService(mock_settings)

    with pytest.raises(ValueError, match="Empty response from LLM"):
        service.generate_description("data:image/jpeg;base64,abc123")


@patch("app.services.image_description_service.completion")
def test_missing_required_field_raises_error(mock_completion, mock_settings):
    """Test that response missing required field raises ValueError."""
    # Setup mock response missing TRANSCRIPT
    incomplete_response = {
        "FULL_DESCRIPTION": "Test description",
        "SAF": {},
        "SAR": "Test"
        # Missing TRANSCRIPT
    }

    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps(incomplete_response)
    mock_completion.return_value = mock_response

    # Create service and call
    service = ImageDescriptionService(mock_settings)

    with pytest.raises(ValueError, match="Missing required field: TRANSCRIPT"):
        service.generate_description("data:image/jpeg;base64,abc123")


@patch("app.services.image_description_service.completion")
def test_missing_safety_field_raises_error(mock_completion, mock_settings):
    """Test that response missing safety assessment field raises ValueError."""
    # Setup mock response with incomplete safety form (using short keys)
    incomplete_response = {
        "FULL_DESCRIPTION": "Test",
        "TRANSCRIPT": "Test",
        "SAF": {
            "people": "N"
            # Missing other required safety fields
        },
        "SAR": "Test"
    }

    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps(incomplete_response)
    mock_completion.return_value = mock_response

    # Create service and call
    service = ImageDescriptionService(mock_settings)

    with pytest.raises(ValueError, match="Missing required safety assessment field"):
        service.generate_description("data:image/jpeg;base64,abc123")


def test_prompt_file_loaded(mock_settings):
    """Test that prompt file is loaded during initialization."""
    service = ImageDescriptionService(mock_settings)

    # Verify prompt was loaded from real file
    assert service.system_prompt is not None
    assert len(service.system_prompt) > 0
    # Verify it's actual prompt content, not empty
    assert isinstance(service.system_prompt, str)


@patch("app.services.image_description_service.completion")
def test_retries_on_invalid_response_then_succeeds(mock_completion, mock_settings, sample_llm_response):
    """Test that a transient parse error triggers a retry and eventual success is returned."""
    bad_response = Mock()
    bad_response.choices = [Mock()]
    bad_response.choices[0].message.content = "not valid json"

    good_response = Mock()
    good_response.choices = [Mock()]
    good_response.choices[0].message.content = json.dumps(sample_llm_response)

    mock_completion.side_effect = [bad_response, bad_response, good_response]

    service = ImageDescriptionService(mock_settings)
    result = service.generate_description("data:image/jpeg;base64,abc123")

    assert mock_completion.call_count == 3
    assert result["FULL_DESCRIPTION"] == sample_llm_response["FULL_DESCRIPTION"]


@patch("app.services.image_description_service.completion")
def test_all_retries_exhausted_raises_error(mock_completion, mock_settings):
    """Test that after MAX_PARSE_RETRIES attempts the error is re-raised."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = None
    mock_completion.return_value = mock_response

    service = ImageDescriptionService(mock_settings)

    with pytest.raises(ValueError, match="Empty response from LLM"):
        service.generate_description("data:image/jpeg;base64,abc123")

    assert mock_completion.call_count == ImageDescriptionService._MAX_PARSE_RETRIES


@patch("app.services.image_description_service.completion")
def test_non_parse_error_is_not_retried(mock_completion, mock_settings):
    """Test that non-ValueError/JSONDecodeError exceptions are not retried."""
    mock_completion.side_effect = RuntimeError("Network error")

    service = ImageDescriptionService(mock_settings)

    with pytest.raises(RuntimeError, match="Network error"):
        service.generate_description("data:image/jpeg;base64,abc123")

    assert mock_completion.call_count == 1
