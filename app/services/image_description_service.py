"""Image description generation service using LiteLLM."""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from litellm import completion

from app.config import Settings
from app.services.safety_form_expander import expand_safety_form, SAFETY_FORM_KEY_MAP

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

        # Load the task prompt template
        prompt_path = Path(__file__).parent.parent / "prompts" / "full_description_prompt.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.task_prompt = f.read()

        self.system_prompt = (
            "You are a specialized visual content analyzer creating detailed descriptive metadata for archival and accessibility purposes.\n"
            "You generate clear, factual descriptions that document everything visible without interpretation.\n"
            "You use precise language appropriate to the content domain, while avoiding unnecessary jargon.\n"
            "Your descriptions are well-structured and prioritize factual documentation over stylistic concerns."
        )

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
        logger.info("Generating image description via LLM")

        # Build messages
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        # Build user message content
        user_content = []

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
                "model": self.settings.litellm_full_desc_model,
                "messages": messages,
                "temperature": self.settings.litellm_full_desc_temperature,
                "max_tokens": self.settings.litellm_full_desc_max_tokens,
                "response_format": response_format,
                "num_retries": self.settings.litellm_num_retries
            }

            # Specify reasoning effort for models that support it
            if self.settings.litellm_full_desc_reasoning_effort:
                completion_params["reasoning_effort"] = self.settings.litellm_full_desc_reasoning_effort

            last_exc: Exception = ValueError("No attempts made")
            for attempt in range(1, self._MAX_PARSE_RETRIES + 1):
                try:
                    response = completion(**completion_params)

                    # Parse response
                    if not response.choices or not response.choices[0].message.content:
                        raise ValueError("Empty response from LLM")

                    result = json.loads(response.choices[0].message.content)

                    # Validate required fields
                    self._validate_response(result)

                    # Expand abbreviated safety form keys/values to full forms
                    result["SAFETY_ASSESSMENT_FORM"] = expand_safety_form(result["SAFETY_ASSESSMENT_FORM"])

                    logger.info("Successfully generated image description")
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
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "FULL_DESCRIPTION": {
                            "type": "string"
                        },
                        "TRANSCRIPT": {
                            "type": "string"
                        },
                        "SAFETY_ASSESSMENT_FORM": {
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
                                            "enum": ["Y", "N"]
                                        },
                                        "type": {
                                            "type": "string",
                                            "enum": ["NA", "PR", "TY", "HWPR", "HWCU", "MX"]
                                        },
                                        "legib": {
                                            "type": "string",
                                            "enum": ["NA", "CL", "PC", "DIF", "ILL"]
                                        }
                                    },
                                    "required": ["present", "type", "legib"],
                                    "additionalProperties": False
                                },
                                "confidence": {
                                    "type": "string",
                                    "enum": ["L", "M", "H"]
                                }
                            },
                            "required": [
                                "people",
                                "demog",
                                "misid_risk",
                                "minors",
                                "named_indiv",
                                "violence",
                                "racial_viol",
                                "nudity",
                                "sexual",
                                "symbols",
                                "stereotyping",
                                "atrocities",
                                "text_chars",
                                "confidence"
                            ],
                            "additionalProperties": False
                        },
                        "SAFETY_ASSESSMENT_REASONING": {
                            "type": "string"
                        }
                    },
                    "required": ["FULL_DESCRIPTION", "TRANSCRIPT", "SAFETY_ASSESSMENT_FORM", "SAFETY_ASSESSMENT_REASONING"],
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
        required_fields = ["FULL_DESCRIPTION", "TRANSCRIPT", "SAFETY_ASSESSMENT_FORM", "SAFETY_ASSESSMENT_REASONING"]
        for field in required_fields:
            if field not in response:
                raise ValueError(f"Missing required field: {field}")

        # Validate nested SAFETY_ASSESSMENT_FORM structure (uses abbreviated keys)
        safety_form = response["SAFETY_ASSESSMENT_FORM"]
        for short_key in SAFETY_FORM_KEY_MAP:
            if short_key not in safety_form:
                raise ValueError(f"Missing required safety assessment field: {short_key}")
