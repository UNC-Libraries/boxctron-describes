"""LiteLLM service for comparing evaluation outputs to fixture baselines."""
import json
import logging
from pathlib import Path
from typing import Any

from litellm import completion

from app.config import Settings
from app.utils.llm_utils import log_token_usage


logger = logging.getLogger(__name__)


class EvaluationService:
    """Generate per-item and aggregate assessments for baseline evaluations."""

    _MAX_PARSE_RETRIES = 3

    def __init__(self, settings: Settings):
        self.settings = settings
        prompts_directory = Path(__file__).parent.parent / "prompts"
        self.comparison_prompt = (prompts_directory / "evaluation_comparison_prompt.txt").read_text(
            encoding="utf-8"
        )
        self.overall_prompt = (prompts_directory / "evaluation_overall_prompt.txt").read_text(
            encoding="utf-8"
        )

    def compare_results(self, baseline_result: dict[str, Any], generated_result: dict[str, Any]) -> dict[str, Any]:
        """Compare one generated result against its fixture baseline."""
        messages = [
            {
                "role": "system",
                "content": "You evaluate the quality of AI-generated image descriptions.",
            },
            {
                "role": "user",
                "content": (
                    f"{self.comparison_prompt}\n\nBASELINE_RESULT:\n"
                    f"{json.dumps(baseline_result, indent=2, ensure_ascii=False)}\n\n"
                    f"GENERATED_RESULT:\n{json.dumps(generated_result, indent=2, ensure_ascii=False)}"
                ),
            },
        ]

        last_error: Exception = ValueError("No attempts made")
        for _ in range(self._MAX_PARSE_RETRIES):
            response = completion(
                **self._completion_params(messages, response_format=self._comparison_response_format())
            )
            try:
                result = json.loads(self._response_content(response))
                self._validate_comparison(result)
                log_token_usage(logger, "evaluation comparison", response.usage)
                return result
            except (ValueError, json.JSONDecodeError) as error:
                last_error = error

        raise last_error

    def generate_overall_assessment(self, comparisons: list[dict[str, Any]]) -> str:
        """Generate a textual assessment covering all completed comparisons."""
        messages = [
            {
                "role": "system",
                "content": "You summarize evaluation results for AI-generated image descriptions.",
            },
            {
                "role": "user",
                "content": (
                    f"{self.overall_prompt}\n\nCOMPARISONS:\n"
                    f"{json.dumps(comparisons, indent=2, ensure_ascii=False)}"
                ),
            },
        ]
        response = completion(**self._completion_params(messages))
        content = self._response_content(response)
        log_token_usage(logger, "overall evaluation", response.usage)
        return content

    def _completion_params(self, messages: list[dict[str, str]], response_format: dict[str, Any] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.settings.litellm_evaluation_model,
            "messages": messages,
            "temperature": self.settings.litellm_evaluation_temperature,
            "max_tokens": self.settings.litellm_evaluation_max_tokens,
            "num_retries": self.settings.litellm_num_retries,
        }
        if response_format:
            params["response_format"] = response_format
        if self.settings.litellm_evaluation_reasoning_effort:
            params["reasoning_effort"] = self.settings.litellm_evaluation_reasoning_effort
        if self.settings.litellm_evaluation_base_model:
            params["base_model"] = self.settings.litellm_evaluation_base_model
        return params

    @staticmethod
    def _response_content(response: Any) -> str:
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Empty response from LLM")
        return response.choices[0].message.content

    @staticmethod
    def _validate_comparison(result: dict[str, Any]) -> None:
        required_fields = {"summary", "regressions", "improvements", "notable_differences", "verdict"}
        missing = required_fields - result.keys()
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(sorted(missing))}")
        if result["verdict"] not in {"IMPROVED", "EQUIVALENT", "REGRESSED", "MIXED"}:
            raise ValueError("Invalid verdict")

    @staticmethod
    def _comparison_response_format() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "evaluation_comparison",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "regressions": {"type": "array", "items": {"type": "string"}},
                        "improvements": {"type": "array", "items": {"type": "string"}},
                        "notable_differences": {"type": "array", "items": {"type": "string"}},
                        "verdict": {
                            "type": "string",
                            "enum": ["IMPROVED", "EQUIVALENT", "REGRESSED", "MIXED"],
                        },
                    },
                    "required": [
                        "summary",
                        "regressions",
                        "improvements",
                        "notable_differences",
                        "verdict",
                    ],
                    "additionalProperties": False,
                },
            },
        }
