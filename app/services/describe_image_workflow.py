"""Image description workflow service."""
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import logging

from app.services.image_normalizer import ImageNormalizer
from app.services.image_description_service import ImageDescriptionService
from app.models import DescriptionResult, SafetyAssessment, ReviewAssessment, VersionInfo, SymbolsPresent, TextCharacteristics
from app.config import Settings

logger = logging.getLogger(__name__)

class DescribeImageWorkflow:
    """Service orchestrating the image description workflow."""

    def __init__(self, settings: Settings, image_normalizer: ImageNormalizer, image_description_service: ImageDescriptionService):
        """
        Initialize the DescribeImageWorkflow.

        Args:
            settings: Application settings
            image_normalizer: Service for normalizing images
        """
        self.settings = settings
        self.image_normalizer = image_normalizer
        self.image_description_service = image_description_service

    async def process_image(
        self,
        image_path: Path,
        filename: str,
        mimetype: str,
        context: Optional[str] = None
    ) -> DescriptionResult:
        """
        Process an image through the full description workflow.

        Args:
            image_path: Path to the image file (local file system)
            filename: Original filename
            mimetype: MIME type of the image
            context: Optional context to guide description generation

        Returns:
            DescriptionResult with full description, safety assessment, etc.

        Raises:
            IOError: If the image cannot be read or processed
            ValueError: If the image format is not supported
        """
        logger.info(f"Processing image {image_path}")
        # Normalize image
        base64_image = self.image_normalizer.normalize_image(image_path)

        # Generate full description, transcript, and safety assessment
        full_desc_result = self.image_description_service.generate_description(base64_image, context)
        logger.info(f"Generated description for {filename}")

        # Parse safety assessment from LLM response
        safety_assessment = self._parse_safety_assessment(full_desc_result)

        # TODO: Generate review assessment in a separate step
        # For now, use placeholder
        review_assessment = ReviewAssessment(
            biased_language="NO",
            stereotyping="NO",
            value_judgments="NO",
            contradictions_between_texts="NO",
            contradictions_within_description="NO",
            offensive_language="NO",
            inconsistent_demographics="NO",
            euphemistic_language="NO",
            people_first_language="N/A",
            unsupported_inferential_claims="NO",
            safety_assessment_consistency="CONSISTENT",
            concerns_for_review=[]
        )

        # TODO: Generate alt text in a separate step
        # For now, use a simplified version from full description
        alt_text = full_desc_result.get("FULL_DESCRIPTION", "")[:200] + "..."

        return DescriptionResult(
            full_description=full_desc_result.get("FULL_DESCRIPTION", ""),
            alt_text=alt_text,
            transcript=full_desc_result.get("TRANSCRIPT", ""),
            safety_assessment=safety_assessment,
            review_assessment=review_assessment,
            version=VersionInfo(
                version=self.settings.app_version,
                models=[self.settings.litellm_full_desc_model],
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        )

    def _parse_safety_assessment(self, full_desc_result: dict) -> SafetyAssessment:
        """
        Parse safety assessment from LLM response.

        Args:
            full_desc_result: Dictionary containing LLM response with SAFETY_ASSESSMENT_FORM

        Returns:
            SafetyAssessment object
        """
        safety_form = full_desc_result.get("SAFETY_ASSESSMENT_FORM", {})
        symbols_data = safety_form.get("symbols_present", {})

        return SafetyAssessment(
            people_visible=safety_form.get("people_visible", "UNKNOWN"),
            demographics_described=safety_form.get("demographics_described", "UNKNOWN"),
            misidentification_risk_people=safety_form.get("misidentification_risk_people", "UNKNOWN"),
            minors_present=safety_form.get("minors_present", "UNKNOWN"),
            named_individuals_claimed=safety_form.get("named_individuals_claimed", "UNKNOWN"),
            violent_content=safety_form.get("violent_content", "UNKNOWN"),
            racial_violence_oppression=safety_form.get("racial_violence_oppression", "UNKNOWN"),
            nudity=safety_form.get("nudity", "UNKNOWN"),
            sexual_content=safety_form.get("sexual_content", "UNKNOWN"),
            symbols_present=SymbolsPresent(
                types=symbols_data.get("types", ["NONE"]),
                names=symbols_data.get("names", []),
                misidentification_risk=symbols_data.get("misidentification_risk", "UNKNOWN")
            ),
            stereotyping_present=safety_form.get("stereotyping_present", "UNKNOWN"),
            atrocities_depicted=safety_form.get("atrocities_depicted", "UNKNOWN"),
            text_characteristics=TextCharacteristics(
                text_present=safety_form.get("text_characteristics", {}).get("text_present", "UNKNOWN"),
                text_type=safety_form.get("text_characteristics", {}).get("text_type", "N/A"),
                legibility=safety_form.get("text_characteristics", {}).get("legibility", "N/A")
            ),
            confidence=safety_form.get("confidence", "UNKNOWN"),
            reasoning=full_desc_result.get("SAFETY_ASSESSMENT_REASONING", "")
        )
