"""Image description generation service using LiteLLM."""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from litellm import completion

from app.config import Settings

logger = logging.getLogger(__name__)


class ImageDescriptionService:
    """Service for generating image descriptions using LLM vision models."""

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

            response = completion(**completion_params)

            # Parse response
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from LLM")

            result = json.loads(response.choices[0].message.content)

            # Validate required fields
            self._validate_response(result)

            logger.info("Successfully generated image description")
            return result

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
                                "people_visible": {
                                    "type": "string",
                                    "enum": ["YES", "NO"]
                                },
                                "demographics_described": {
                                    "type": "string",
                                    "enum": ["YES", "NO"]
                                },
                                "misidentification_risk_people": {
                                    "type": "string",
                                    "enum": ["LOW", "MEDIUM", "HIGH"]
                                },
                                "minors_present": {
                                    "type": "string",
                                    "enum": ["YES", "NO"]
                                },
                                "named_individuals_claimed": {
                                    "type": "string",
                                    "enum": ["YES", "NO"]
                                },
                                "violent_content": {
                                    "type": "string",
                                    "enum": ["NONE", "IMPLIED", "DEPICTED"]
                                },
                                "racial_violence_oppression": {
                                    "type": "string",
                                    "enum": ["NONE", "IMPLIED", "DEPICTED"]
                                },
                                "nudity": {
                                    "type": "string",
                                    "enum": ["NONE", "PARTIAL", "FULL"]
                                },
                                "sexual_content": {
                                    "type": "string",
                                    "enum": ["NONE", "SUGGESTIVE", "EXPLICIT"]
                                },
                                "symbols_present": {
                                    "type": "object",
                                    "properties": {
                                        "types": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "enum": ["NONE", "CULTURAL", "RELIGIOUS", "POLITICAL", "HATE", "BRAND"]
                                            }
                                        },
                                        "names": {
                                            "type": "array",
                                            "items": {
                                                "type": "string"
                                            }
                                        },
                                        "misidentification_risk": {
                                            "type": "string",
                                            "enum": ["LOW", "MEDIUM", "HIGH"]
                                        }
                                    },
                                    "required": ["types", "names", "misidentification_risk"],
                                    "additionalProperties": False
                                },
                                "stereotyping_present": {
                                    "type": "string",
                                    "enum": ["NO", "POSSIBLY", "YES"]
                                },
                                "atrocities_depicted": {
                                    "type": "string",
                                    "enum": ["NO", "YES"]
                                },
                                "text_characteristics": {
                                    "type": "object",
                                    "properties": {
                                        "text_present": {
                                            "type": "string",
                                            "enum": ["YES", "NO"]
                                        },
                                        "text_type": {
                                            "type": "string",
                                            "enum": ["N/A", "PRINTED", "TYPED", "HANDWRITTEN_PRINT", "HANDWRITTEN_CURSIVE", "MIXED"]
                                        },
                                        "legibility": {
                                            "type": "string",
                                            "enum": ["N/A", "CLEAR", "PARTIALLY_CLEAR", "DIFFICULT", "ILLEGIBLE"]
                                        }
                                    },
                                    "required": ["text_present", "text_type", "legibility"],
                                    "additionalProperties": False
                                },
                                "confidence": {
                                    "type": "string",
                                    "enum": ["LOW", "MEDIUM", "HIGH"]
                                }
                            },
                            "required": [
                                "people_visible",
                                "demographics_described",
                                "misidentification_risk_people",
                                "minors_present",
                                "named_individuals_claimed",
                                "violent_content",
                                "racial_violence_oppression",
                                "nudity",
                                "sexual_content",
                                "symbols_present",
                                "stereotyping_present",
                                "atrocities_depicted",
                                "text_characteristics",
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

        # Validate nested SAFETY_ASSESSMENT_FORM structure
        safety_form = response["SAFETY_ASSESSMENT_FORM"]
        required_safety_fields = [
            "people_visible", "demographics_described", "misidentification_risk_people",
            "minors_present", "named_individuals_claimed", "violent_content",
            "racial_violence_oppression", "nudity", "sexual_content", "symbols_present",
            "stereotyping_present", "atrocities_depicted", "text_characteristics", "confidence"
        ]
        for field in required_safety_fields:
            if field not in safety_form:
                raise ValueError(f"Missing required safety assessment field: {field}")
