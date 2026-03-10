"""Alt text generation service using LiteLLM."""
import logging
from pathlib import Path

from litellm import completion

from app.config import Settings
from app.utils.llm_utils import log_token_usage

logger = logging.getLogger(__name__)


class AltTextGenerationService:
    """Service for generating concise alt text from detailed image descriptions."""

    def __init__(self, settings: Settings):
        """
        Initialize the AltTextGenerationService.

        Args:
            settings: Application settings containing LLM configuration
        """
        self.settings = settings

        # Load the prompt template
        prompt_path = Path(__file__).parent.parent / "prompts" / "alt_text_prompt.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.alt_text_prompt = f.read()

        self.system_prompt = (
            "You are an accessibility specialist creating concise alt text from detailed descriptions.\n"
            "You identify and extract only the most contextually relevant elements from comprehensive descriptions.\n"
            "You use clear, straightforward language that works well with screen readers.\n"
            "You focus on conveying function and context rather than exhaustive visual details."
        )

    def generate_alt_text(self, full_description: str) -> str:
        """
        Generate concise alt text from a full description.

        Args:
            full_description: The full image description to summarize

        Returns:
            String containing the generated alt text

        Raises:
            Exception: If LLM call fails or response is invalid
        """
        logger.info("Generating alt text from full description")

        # Build messages
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": f"{self.alt_text_prompt}\n\nFull description:\n{full_description}"
            }
        ]

        # Call LiteLLM
        try:
            # Build completion parameters
            completion_params = {
                "model": self.settings.litellm_alt_text_model,
                "messages": messages,
                "temperature": self.settings.litellm_alt_text_temperature,
                "max_tokens": self.settings.litellm_alt_text_max_tokens,
                "num_retries": self.settings.litellm_num_retries
            }

            # Specify reasoning effort for models that support it
            if self.settings.litellm_alt_text_reasoning_effort:
                completion_params["reasoning_effort"] = self.settings.litellm_alt_text_reasoning_effort

            response = completion(**completion_params)

            # Parse response
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from LLM")

            alt_text = response.choices[0].message.content.strip()

            log_token_usage(logger, "alt text", response.usage)
            logger.info("Successfully generated alt text")
            return alt_text

        except Exception as e:
            logger.error(f"Error generating alt text: {e}")
            raise
