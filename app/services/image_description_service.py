"""Image description generation service using LiteLLM."""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from litellm import completion

from app.config import Settings
from app.services.safety_form_expander import expand_safety_form
from app.utils.llm_utils import format_llm_response_diagnostics, log_token_usage

logger = logging.getLogger(__name__)


class ImageDescriptionService:
    """Service for generating image descriptions using LLM vision models."""

    _MAX_PARSE_RETRIES = 3

    def __init__(self, settings: Settings):
        """
        Initialize the ImageDescriptionService.

        Args:
            settings: Application settings containing LLM configuration
        """
        self.settings = settings
        self.model = settings.litellm_full_desc_model
        self.temperature = settings.litellm_full_desc_temperature
        self.max_tokens = settings.litellm_full_desc_max_tokens
        self.reasoning_effort = settings.litellm_full_desc_reasoning_effort
        self.timeout: Optional[float] = None
        self.api_base: Optional[str] = None
        self.api_key: Optional[str] = None
        self.step_name = "image description"

        # Load the task prompt template
        prompt_path = Path(__file__).parent.parent / "prompts" / "full_description_prompt.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.task_prompt = f.read()

        self.system_prompt = (
            "You are a specialized visual content analyzer creating detailed descriptive metadata for archival and accessibility purposes.\n"
            "You generate clear, factual descriptions that document everything visible without interpretation.\n"
            "You use precise language appropriate to the content domain, while avoiding unnecessary jargon.\n"
            "Your descriptions are well-structured and prioritize factual documentation over stylistic concerns.\n"
            "You also generate concise, screen-reader-friendly alt text by selecting only the most contextually relevant elements from what you observe."
        )

    @classmethod
    def for_transcribe(cls, settings: Settings) -> "ImageDescriptionService":
        """Create an instance configured with the LITELLM_TRANSCRIBE_* settings."""
        instance = cls(settings)
        instance.model = settings.litellm_transcribe_model
        instance.temperature = settings.litellm_transcribe_temperature
        instance.max_tokens = settings.litellm_transcribe_max_tokens
        instance.timeout = settings.litellm_transcribe_timeout
        instance.reasoning_effort = settings.litellm_transcribe_reasoning_effort
        instance.api_base = settings.litellm_transcribe_api_base
        instance.api_key = settings.litellm_transcribe_api_key
        instance.step_name = "transcription"
        return instance

    def generate_description(
        self,
        base64_image: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate full description, transcript, and safety assessment for an image.

        Args:
            base64_image: Base64-encoded image data URL (e.g., "data:image/jpeg;base64,...")
            context: Optional contextual information about the image

        Returns:
            Dictionary containing:
                - FULL_DESCRIPTION: str
                - TRANSCRIPT: str
                - SAFETY_ASSESSMENT_FORM: dict
                - SAFETY_ASSESSMENT_REASONING: str

        Raises:
            Exception: If LLM call fails or response is invalid
        """
        logger.info(f"Generating {self.step_name} via LLM")

        # Build messages
        messages: list[Dict[str, Any]] = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        # Build user message content
        user_content: list[Dict[str, Any]] = []

        # Add task instructions
        user_content.append({
            "type": "text",
            "text": self.task_prompt
        })

        # Add context if provided
        if context:
            user_content.append({
                "type": "text",
                "text": f"\n\nReference information: {context}"
            })

        # Add image
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": base64_image
            }
        })

        messages.append({
            "role": "user",
            "content": user_content
        })

        # Define structured output schema
        response_format = self._get_response_format()

        # Call LiteLLM
        try:
            # Build completion parameters
            completion_params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": response_format,
                "num_retries": self.settings.litellm_num_retries
            }

            if self.reasoning_effort:
                completion_params["reasoning_effort"] = self.reasoning_effort
            if self.timeout is not None:
                completion_params["timeout"] = self.timeout
            if self.api_base:
                completion_params["api_base"] = self.api_base
            if self.api_key:
                completion_params["api_key"] = self.api_key

            last_exc: Exception = ValueError("No attempts made")
            response = None
            for attempt in range(1, self._MAX_PARSE_RETRIES + 1):
                try:
                    response = completion(**completion_params)

                    # Parse response
                    if not response.choices or not response.choices[0].message.content:
                        raise ValueError("Empty response from LLM")

                    result = json.loads(response.choices[0].message.content)

                    # Validate required fields
                    self._validate_response(result)

                    # Expand abbreviated safety form keys/values and rename top-level keys
                    result["SAFETY_ASSESSMENT_FORM"] = expand_safety_form(result.pop("SAF"))
                    result["SAFETY_ASSESSMENT_REASONING"] = result.pop("SAR")
                    result["ALT_TEXT"] = result.pop("ALT_TEXT")

                    log_token_usage(logger, self.step_name, response.usage)

                    logger.info(f"Successfully generated {self.step_name}")
                    return result

                except (ValueError, json.JSONDecodeError) as e:
                    last_exc = e
                    response_diagnostics = format_llm_response_diagnostics(response)
                    if attempt < self._MAX_PARSE_RETRIES:
                        logger.warning(
                            f"Attempt {attempt}/{self._MAX_PARSE_RETRIES} failed with "
                            f"parse/validation error, retrying: {e}. "
                            f"Response diagnostics: {response_diagnostics}"
                        )
                    else:
                        logger.error(
                            f"Attempt {attempt}/{self._MAX_PARSE_RETRIES} failed with "
                            f"parse/validation error: {e}. "
                            f"Response diagnostics: {response_diagnostics}"
                        )

            raise last_exc

        except Exception as e:
            logger.error(f"Error generating description: {e}")
            raise

    def _get_response_format(self) -> Dict[str, Any]:
        """
        Get the JSON schema for structured output.

        Returns:
            Response format dict for LiteLLM
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "image_analysis",
                "strict": False,
                "schema": {
                    "type": "object",
                    "properties": {
                        "FULL_DESCRIPTION": {
                            "type": "string"
                        },
                        "ALT_TEXT": {
                            "type": "string"
                        },
                        "TRANSCRIPT": {
                            "type": "string"
                        },
                        "SAF": {
                            "type": "object",
                            "properties": {
                                "people": {
                                    "type": "string",
                                    "enum": ["Y", "N"]
                                },
                                "demog": {
                                    "type": "string",
                                    "enum": ["Y", "N"]
                                },
                                "misid_risk": {
                                    "type": "string",
                                    "enum": ["L", "M", "H"]
                                },
                                "minors": {
                                    "type": "string",
                                    "enum": ["Y", "N"]
                                },
                                "named_indiv": {
                                    "type": "string",
                                    "enum": ["Y", "N"]
                                },
                                "violence": {
                                    "type": "string",
                                    "enum": ["0", "IMP", "DEP"]
                                },
                                "racial_viol": {
                                    "type": "string",
                                    "enum": ["0", "IMP", "DEP"]
                                },
                                "nudity": {
                                    "type": "string",
                                    "enum": ["0", "PAR", "FULL"]
                                },
                                "sexual": {
                                    "type": "string",
                                    "enum": ["0", "SUG", "EXP"]
                                },
                                "symbols": {
                                    "type": "object",
                                    "properties": {
                                        "types": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "enum": ["0", "CUL", "REL", "POL", "HATE", "BRD"]
                                            }
                                        },
                                        "names": {
                                            "type": "array",
                                            "items": {
                                                "type": "string"
                                            }
                                        },
                                        "misid_risk": {
                                            "type": "string",
                                            "enum": ["L", "M", "H"]
                                        }
                                    },
                                    "required": ["types", "names", "misid_risk"],
                                    "additionalProperties": False
                                },
                                "stereotyping": {
                                    "type": "string",
                                    "enum": ["N", "P", "Y"]
                                },
                                "atrocities": {
                                    "type": "string",
                                    "enum": ["N", "Y"]
                                },
                                "text_chars": {
                                    "type": "object",
                                    "properties": {
                                        "present": {
                                            "type": "string",
                                            "enum": ["N", "INC", "SIG"]
                                        },
                                        "type": {
                                            "type": "string",
                                            "enum": ["NA", "PR", "TY", "HWPR", "HWCU", "MX"]
                                        },
                                        "legib": {
                                            "type": "string",
                                            "enum": ["NA", "CL", "PC", "DIF", "ILL"]
                                        },
                                        "sensitiv": {
                                            "type": "string",
                                            "enum": ["NA", "0", "S"]
                                        },
                                        "lang": {
                                            "type": "string"
                                        }
                                    },
                                    "required": ["present"],
                                    "additionalProperties": False
                                },
                                "img_qual": {
                                    "type": "string",
                                    "enum": ["0", "DGR", "IMP"]
                                }
                            },
                            "required": [
                                "people",
                                "violence",
                                "racial_viol",
                                "nudity",
                                "sexual",
                                "symbols",
                                "stereotyping",
                                "atrocities",
                                "text_chars",
                                "img_qual"
                            ],
                            "additionalProperties": False
                        },
                        "SAR": {
                            "type": "string"
                        }
                    },
                    "required": ["FULL_DESCRIPTION", "ALT_TEXT", "TRANSCRIPT", "SAF", "SAR"],
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
        required_fields = ["FULL_DESCRIPTION", "ALT_TEXT", "TRANSCRIPT", "SAF", "SAR"]
        for field in required_fields:
            if field not in response:
                raise ValueError(f"Missing required field: {field}")

        # Validate nested SAF structure (uses abbreviated keys)
        safety_form = response["SAF"]
        always_required = [
            "people", "violence", "racial_viol", "nudity", "sexual",
            "symbols", "stereotyping", "atrocities", "text_chars", "img_qual",
        ]
        for field in always_required:
            if field not in safety_form:
                raise ValueError(f"Missing required safety assessment field: {field}")

        # demog/misid_risk/minors/named_indiv are only required when people != "N"
        if safety_form.get("people") != "N":
            for field in ["demog", "misid_risk", "minors", "named_indiv"]:
                if field not in safety_form:
                    raise ValueError(f"Missing required safety assessment field: {field}")

        # text_chars sub-fields are only required when text_chars.present != "N"
        text_chars = safety_form.get("text_chars", {})
        if isinstance(text_chars, dict) and text_chars.get("present") != "N":
            for field in ["type", "legib", "sensitiv", "lang"]:
                if field not in text_chars:
                    raise ValueError(f"Missing required safety assessment field: text_chars.{field}")
