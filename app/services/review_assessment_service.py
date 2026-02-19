"""Review assessment service using LiteLLM."""
import json
import logging
from pathlib import Path
from typing import Dict, Any

from litellm import completion

from app.config import Settings

logger = logging.getLogger(__name__)


class ReviewAssessmentService:
    """Service for reviewing generated content for quality and bias issues."""

    def __init__(self, settings: Settings):
        """
        Initialize the ReviewAssessmentService.

        Args:
            settings: Application settings containing LLM configuration
        """
        self.settings = settings

        # Load the prompt template
        prompt_path = Path(__file__).parent.parent / "prompts" / "review_prompt.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.review_prompt = f.read()

        self.system_prompt = (
            "You are a content quality reviewer specializing in accessibility and bias detection.\n"
            "Your task is to review AI-generated image descriptions and alt text for language issues, "
            "inconsistencies, and potential problems that require human attention."
        )

    def generate_review_assessment(
        self,
        full_description: str,
        transcript: str,
        safety_assessment: Dict[str, Any],
        safety_assessment_reasoning: str,
        alt_text: str
    ) -> Dict[str, Any]:
        """
        Generate a review assessment for all generated content.

        Args:
            full_description: The full image description
            transcript: The text transcript from the image
            safety_assessment: The safety assessment form data
            safety_assessment_reasoning: The reasoning for the safety assessment
            alt_text: The generated alt text

        Returns:
            Dictionary containing review assessment fields

        Raises:
            Exception: If LLM call fails or response is invalid
        """
        logger.info("Generating review assessment")

        # Format the content for review
        content_to_review = self._format_content_for_review(
            full_description,
            transcript,
            safety_assessment,
            safety_assessment_reasoning,
            alt_text
        )

        # Build messages
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": f"{self.review_prompt}\n\n{content_to_review}"
            }
        ]

        # Define structured output schema
        response_format = self._get_response_format()

        # Call LiteLLM
        try:
            # Build completion parameters
            completion_params = {
                "model": self.settings.litellm_review_model,
                "messages": messages,
                "temperature": self.settings.litellm_review_temperature,
                "max_tokens": self.settings.litellm_review_max_tokens,
                "response_format": response_format,
                "num_retries": self.settings.litellm_num_retries
            }

            # Specify reasoning effort for models that support it
            if self.settings.litellm_review_reasoning_effort:
                completion_params["reasoning_effort"] = self.settings.litellm_review_reasoning_effort

            response = completion(**completion_params)

            # Parse response
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from LLM")

            result = json.loads(response.choices[0].message.content)

            # Validate required fields
            self._validate_response(result)

            logger.info("Successfully generated review assessment")
            return result

        except Exception as e:
            logger.error(f"Error generating review assessment: {e}")
            raise

    def _format_content_for_review(
        self,
        full_description: str,
        transcript: str,
        safety_assessment: Dict[str, Any],
        safety_assessment_reasoning: str,
        alt_text: str
    ) -> str:
        """
        Format all content pieces with labels for review.

        Args:
            full_description: The full image description
            transcript: The text transcript
            safety_assessment: The safety assessment form
            safety_assessment_reasoning: The safety reasoning
            alt_text: The generated alt text

        Returns:
            Formatted string with labeled content
        """
        # Convert safety assessment to formatted JSON string
        safety_assessment_str = json.dumps(safety_assessment, indent=2)

        return f"""FULL_DESCRIPTION:
{full_description}

TRANSCRIPT:
{transcript}

SAFETY_ASSESSMENT_FORM:
{safety_assessment_str}

SAFETY_ASSESSMENT_REASONING:
{safety_assessment_reasoning}

ALT_TEXT:
{alt_text}"""

    def _get_response_format(self) -> Dict[str, Any]:
        """
        Get the JSON schema for structured output.

        Returns:
            Response format dict for LiteLLM
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "review_assessment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "biased_language": {
                            "type": "string",
                            "enum": ["NO", "POSSIBLY", "YES"]
                        },
                        "stereotyping": {
                            "type": "string",
                            "enum": ["NO", "POSSIBLY", "YES"]
                        },
                        "value_judgments": {
                            "type": "string",
                            "enum": ["NO", "POSSIBLY", "YES"]
                        },
                        "contradictions_between_texts": {
                            "type": "string",
                            "enum": ["NO", "YES"]
                        },
                        "contradictions_within_description": {
                            "type": "string",
                            "enum": ["NO", "POSSIBLY", "YES"]
                        },
                        "offensive_language": {
                            "type": "string",
                            "enum": ["NO", "YES"]
                        },
                        "inconsistent_demographics": {
                            "type": "string",
                            "enum": ["NO", "YES"]
                        },
                        "euphemistic_language": {
                            "type": "string",
                            "enum": ["NO", "POSSIBLY", "YES"]
                        },
                        "people_first_language": {
                            "type": "string",
                            "enum": ["USED", "NOT_USED", "N/A"]
                        },
                        "unsupported_inferential_claims": {
                            "type": "string",
                            "enum": ["NO", "POSSIBLY", "YES"]
                        },
                        "safety_assessment_consistency": {
                            "type": "string",
                            "enum": ["CONSISTENT", "INCONSISTENT"]
                        },
                        "concerns_for_review": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "biased_language",
                        "stereotyping",
                        "value_judgments",
                        "contradictions_between_texts",
                        "contradictions_within_description",
                        "offensive_language",
                        "inconsistent_demographics",
                        "euphemistic_language",
                        "people_first_language",
                        "unsupported_inferential_claims",
                        "safety_assessment_consistency",
                        "concerns_for_review"
                    ],
                    "additionalProperties": False
                }
            }
        }

    def _validate_response(self, response: Dict[str, Any]) -> None:
        """
        Validate the LLM response has required fields.

        Args:
            response: Parsed JSON response from LLM

        Raises:
            ValueError: If response is missing required fields
        """
        required_fields = [
            "biased_language",
            "stereotyping",
            "value_judgments",
            "contradictions_between_texts",
            "contradictions_within_description",
            "offensive_language",
            "inconsistent_demographics",
            "euphemistic_language",
            "people_first_language",
            "unsupported_inferential_claims",
            "safety_assessment_consistency",
            "concerns_for_review"
        ]
        for field in required_fields:
            if field not in response:
                raise ValueError(f"Missing required field: {field}")
