"""Review assessment service using LiteLLM."""
import json
import logging
from pathlib import Path
from typing import Dict, Any

from litellm import completion
from toon import encode as toon_encode

from app.config import Settings
from app.services.review_form_expander import expand_review_form, REVIEW_KEY_MAP
from app.utils.llm_utils import log_token_usage

logger = logging.getLogger(__name__)


class ReviewAssessmentService:
    """Service for reviewing generated content for quality and bias issues."""

    _MAX_PARSE_RETRIES = 3

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

        self.toon_addendum = (
            "Note: The SAFETY_ASSESSMENT_FORM is provided in TOON format, "
            "a compact key:value notation using indentation for nesting."
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

        # Run JSON-formatted review (primary)
        result = self._call_review_llm(
            full_description, transcript, safety_assessment,
            safety_assessment_reasoning, alt_text, use_toon=False
        )

        # Run TOON-formatted review (experimental comparison)
        try:
            self._call_review_llm(
                full_description, transcript, safety_assessment,
                safety_assessment_reasoning, alt_text, use_toon=True
            )
        except Exception as e:
            logger.warning(f"TOON comparison call failed (non-fatal): {e}")

        return result

    def _call_review_llm(
        self,
        full_description: str,
        transcript: str,
        safety_assessment: Dict[str, Any],
        safety_assessment_reasoning: str,
        alt_text: str,
        use_toon: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a single review LLM call.

        Args:
            full_description: The full image description
            transcript: The text transcript from the image
            safety_assessment: The safety assessment form data
            safety_assessment_reasoning: The reasoning for the safety assessment
            alt_text: The generated alt text
            use_toon: If True, encode safety_assessment in TOON format

        Returns:
            Dictionary containing expanded review assessment fields

        Raises:
            Exception: If LLM call fails or response is invalid
        """
        format_label = "TOON" if use_toon else "JSON"

        if use_toon:
            content_to_review = self._format_content_for_review_toon(
                full_description, transcript, safety_assessment,
                safety_assessment_reasoning, alt_text
            )
        else:
            content_to_review = self._format_content_for_review(
                full_description, transcript, safety_assessment,
                safety_assessment_reasoning, alt_text
            )

        # Build user prompt, prepending TOON addendum when applicable
        user_prompt = self.review_prompt
        if use_toon:
            user_prompt = f"{self.toon_addendum}\n\n{user_prompt}"

        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": f"{user_prompt}\n\n{content_to_review}"
            }
        ]

        response_format = self._get_response_format()

        try:
            completion_params = {
                "model": self.settings.litellm_review_model,
                "messages": messages,
                "temperature": self.settings.litellm_review_temperature,
                "max_tokens": self.settings.litellm_review_max_tokens,
                "response_format": response_format,
                "num_retries": self.settings.litellm_num_retries
            }

            if self.settings.litellm_review_reasoning_effort:
                completion_params["reasoning_effort"] = self.settings.litellm_review_reasoning_effort

            last_exc: Exception = ValueError("No attempts made")
            for attempt in range(1, self._MAX_PARSE_RETRIES + 1):
                try:
                    response = completion(**completion_params)

                    if not response.choices or not response.choices[0].message.content:
                        raise ValueError("Empty response from LLM")

                    result = json.loads(response.choices[0].message.content)

                    self._validate_response(result)

                    result = expand_review_form(result)

                    log_token_usage(logger, f"review assessment ({format_label})", response.usage)

                    logger.info(f"Successfully generated review assessment ({format_label})")
                    return result

                except (ValueError, json.JSONDecodeError) as e:
                    last_exc = e
                    if attempt < self._MAX_PARSE_RETRIES:
                        logger.warning(
                            f"Attempt {attempt}/{self._MAX_PARSE_RETRIES} failed with "
                            f"parse/validation error, retrying: {e}"
                        )

            raise last_exc

        except Exception as e:
            logger.error(f"Error generating review assessment ({format_label}): {e}")
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

    def _format_content_for_review_toon(
        self,
        full_description: str,
        transcript: str,
        safety_assessment: Dict[str, Any],
        safety_assessment_reasoning: str,
        alt_text: str
    ) -> str:
        """
        Format content for review with safety assessment in TOON format.

        Identical to _format_content_for_review except the SAFETY_ASSESSMENT_FORM
        section uses TOON encoding for token efficiency.
        """
        safety_assessment_str = toon_encode(safety_assessment)

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
                        "bias": {
                            "type": "string",
                            "enum": ["N", "P", "Y"]
                        },
                        "stereo": {
                            "type": "string",
                            "enum": ["N", "P", "Y"]
                        },
                        "val_judg": {
                            "type": "string",
                            "enum": ["N", "P", "Y"]
                        },
                        "contra_btwn": {
                            "type": "string",
                            "enum": ["N", "Y"]
                        },
                        "contra_within": {
                            "type": "string",
                            "enum": ["N", "P", "Y"]
                        },
                        "offensive": {
                            "type": "string",
                            "enum": ["N", "Y"]
                        },
                        "incon_demog": {
                            "type": "string",
                            "enum": ["N", "Y"]
                        },
                        "euphemism": {
                            "type": "string",
                            "enum": ["N", "P", "Y"]
                        },
                        "ppl_first": {
                            "type": "string",
                            "enum": ["U", "NU", "NA"]
                        },
                        "unsup_infer": {
                            "type": "string",
                            "enum": ["N", "P", "Y"]
                        },
                        "safety_consist": {
                            "type": "string",
                            "enum": ["CON", "INCON"]
                        },
                        "concerns": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "src_warn": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "bias",
                        "stereo",
                        "val_judg",
                        "contra_btwn",
                        "contra_within",
                        "offensive",
                        "incon_demog",
                        "euphemism",
                        "ppl_first",
                        "unsup_infer",
                        "safety_consist",
                        "concerns",
                        "src_warn"
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
        for short_key in REVIEW_KEY_MAP:
            if short_key not in response:
                raise ValueError(f"Missing required field: {short_key}")
