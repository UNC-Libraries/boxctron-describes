"""Tests for the AltTextGenerationService."""
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from app.services.alt_text_generation_service import AltTextGenerationService
from app.config import Settings


@pytest.fixture
def mock_settings():
    """Provide mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.litellm_alt_text_model = "azure/gpt-4o"
    settings.litellm_alt_text_temperature = 0.5
    settings.litellm_alt_text_max_tokens = 150
    settings.litellm_alt_text_reasoning_effort = None
    settings.litellm_num_retries = 3
    return settings


@pytest.fixture
def service(mock_settings):
    """Provide an AltTextGenerationService instance with mocked dependencies."""
    return AltTextGenerationService(mock_settings)


def test_init_loads_prompt_and_system_prompt(mock_settings):
    """Test that __init__ loads the prompt template correctly."""
    service = AltTextGenerationService(mock_settings)

    assert service.settings == mock_settings
    assert len(service.alt_text_prompt) > 0
    assert "accessibility specialist" in service.system_prompt
    assert "screen readers" in service.system_prompt


@patch("app.services.alt_text_generation_service.completion")
def test_generate_alt_text_success(mock_completion, service, mock_settings):
    """Test successful alt text generation."""
    # Mock LLM response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "A person standing in a field at sunset"
    mock_completion.return_value = mock_response

    full_description = "A detailed description of a person standing alone in a vast field during sunset..."

    result = service.generate_alt_text(full_description)

    assert result == "A person standing in a field at sunset"

    # Verify completion was called with correct parameters
    mock_completion.assert_called_once()
    call_args = mock_completion.call_args[1]

    assert call_args["model"] == mock_settings.litellm_alt_text_model
    assert call_args["temperature"] == mock_settings.litellm_alt_text_temperature
    assert call_args["max_tokens"] == mock_settings.litellm_alt_text_max_tokens
    assert call_args["num_retries"] == mock_settings.litellm_num_retries

    # Verify messages structure
    messages = call_args["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "accessibility specialist" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert full_description in messages[1]["content"]

    # Verify the user message contains both prompt and description
    user_message = call_args["messages"][1]["content"]

    assert service.alt_text_prompt in user_message
    assert full_description in user_message
    assert "Full description:" in user_message


@patch("app.services.alt_text_generation_service.completion")
def test_generate_alt_text_with_reasoning_effort(mock_completion, mock_settings):
    """Test alt text generation with reasoning_effort parameter."""
    mock_settings.litellm_alt_text_reasoning_effort = "medium"
    service = AltTextGenerationService(mock_settings)

    # Mock LLM response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Generated alt text"
    mock_completion.return_value = mock_response

    result = service.generate_alt_text("Test description")

    assert result == "Generated alt text"

    # Verify reasoning_effort was included
    call_args = mock_completion.call_args[1]
    assert call_args["reasoning_effort"] == "medium"


@patch("app.services.alt_text_generation_service.completion")
def test_generate_alt_text_strips_whitespace(mock_completion, service):
    """Test that generated alt text has whitespace stripped."""
    # Mock LLM response with extra whitespace
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "  \n  Alt text with whitespace  \n  "
    mock_completion.return_value = mock_response

    result = service.generate_alt_text("Test description")

    assert result == "Alt text with whitespace"


@patch("app.services.alt_text_generation_service.completion")
def test_generate_alt_text_empty_response(mock_completion, service):
    """Test handling of empty response from LLM."""
    # Mock empty response
    mock_response = Mock()
    mock_response.choices = []
    mock_completion.return_value = mock_response

    with pytest.raises(ValueError, match="Empty response from LLM"):
        service.generate_alt_text("Test description")


@patch("app.services.alt_text_generation_service.completion")
def test_generate_alt_text_no_content(mock_completion, service):
    """Test handling of response with no content."""
    # Mock response with no content
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = None
    mock_completion.return_value = mock_response

    with pytest.raises(ValueError, match="Empty response from LLM"):
        service.generate_alt_text("Test description")


@patch("app.services.alt_text_generation_service.completion")
def test_generate_alt_text_llm_error(mock_completion, service):
    """Test handling of LLM API errors."""
    # Mock LLM error
    mock_completion.side_effect = Exception("LLM API error")

    with pytest.raises(Exception, match="LLM API error"):
        service.generate_alt_text("Test description")


@patch("app.services.alt_text_generation_service.completion")
def test_retries_on_empty_response_then_succeeds(mock_completion, service):
    """Test that a transient empty response triggers a retry and eventual success is returned."""
    bad_response = Mock()
    bad_response.choices = [Mock()]
    bad_response.choices[0].message.content = None

    good_response = Mock()
    good_response.choices = [Mock()]
    good_response.choices[0].message.content = "A person in a field"

    mock_completion.side_effect = [bad_response, bad_response, good_response]

    result = service.generate_alt_text("Test description")

    assert mock_completion.call_count == 3
    assert result == "A person in a field"


@patch("app.services.alt_text_generation_service.completion")
def test_all_retries_exhausted_raises_error(mock_completion, service):
    """Test that after MAX_PARSE_RETRIES attempts the error is re-raised."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = None
    mock_completion.return_value = mock_response

    with pytest.raises(ValueError, match="Empty response from LLM"):
        service.generate_alt_text("Test description")

    assert mock_completion.call_count == AltTextGenerationService._MAX_PARSE_RETRIES


@patch("app.services.alt_text_generation_service.completion")
def test_non_value_error_is_not_retried(mock_completion, service):
    """Test that non-ValueError exceptions are not retried."""
    mock_completion.side_effect = RuntimeError("Network error")

    with pytest.raises(RuntimeError, match="Network error"):
        service.generate_alt_text("Test description")

    assert mock_completion.call_count == 1
