"""Tests for the baseline evaluation LiteLLM service."""
import json
from unittest.mock import Mock, patch

from app.config import Settings
from app.services.evaluation_service import EvaluationService


def build_settings() -> Mock:
    settings = Mock(spec=Settings)
    settings.litellm_evaluation_model = "azure/gpt-4o"
    settings.litellm_evaluation_base_model = None
    settings.litellm_evaluation_temperature = 0.3
    settings.litellm_evaluation_max_tokens = 1000
    settings.litellm_evaluation_reasoning_effort = None
    settings.litellm_num_retries = 3
    return settings


def mock_response(content: str) -> Mock:
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = content
    response.usage = Mock()
    return response


@patch("app.services.evaluation_service.completion")
def test_compare_results_returns_structured_assessment(mock_completion):
    mock_completion.return_value = mock_response(
        json.dumps(
            {
                "summary": "The generated description matches the baseline.",
                "regressions": [],
                "improvements": [],
                "notable_differences": [],
                "verdict": "MATCHES_BASELINE",
            }
        )
    )
    service = EvaluationService(build_settings())

    result = service.compare_results(
        {"full_description": "Baseline", "alt_text": "Baseline", "transcript": "", "safety_assessment": {}},
        {"full_description": "Generated", "alt_text": "Generated", "transcript": "", "safety_assessment": {}},
    )

    assert result["verdict"] == "MATCHES_BASELINE"
    call_args = mock_completion.call_args.kwargs
    assert call_args["model"] == "azure/gpt-4o"
    assert "response_format" in call_args
    assert "BASELINE_RESULT:" in call_args["messages"][1]["content"]
    assert "GENERATED_RESULT:" in call_args["messages"][1]["content"]


@patch("app.services.evaluation_service.completion")
def test_generate_overall_assessment_returns_model_text(mock_completion):
    mock_completion.return_value = mock_response("The results are broadly equivalent.")
    service = EvaluationService(build_settings())

    result = service.generate_overall_assessment(
        [{"filename": "image.jpg", "verdict": "MATCHES_BASELINE"}]
    )

    assert result == "The results are broadly equivalent."
    call_args = mock_completion.call_args.kwargs
    assert "response_format" not in call_args
    assert '"filename": "image.jpg"' in call_args["messages"][1]["content"]
