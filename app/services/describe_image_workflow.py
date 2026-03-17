"""Image description workflow service."""
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import logging

from app.services.image_normalizer import ImageNormalizer
from app.services.image_description_service import ImageDescriptionService
from app.services.alt_text_generation_service import AltTextGenerationService
from app.services.review_assessment_service import ReviewAssessmentService
from app.services.safety_risk_scoring_service import calculate_risk_score
from app.services.safety_inconsistency_service import count_safety_inconsistencies
from app.services.review_risk_scoring_service import calculate_review_risk_score
from app.models import DescriptionResult, SafetyAssessment, ReviewAssessment, VersionInfo, SymbolsPresent, TextCharacteristics
from app.config import Settings

logger = logging.getLogger(__name__)

class DescribeImageWorkflow:
    """Service orchestrating the image description workflow."""

    def __init__(
        self,
        settings: Settings,
        image_normalizer: ImageNormalizer,
        image_description_service: ImageDescriptionService,
        alt_text_service: AltTextGenerationService,
        review_service: ReviewAssessmentService
    ):
        """
        Initialize the DescribeImageWorkflow.

        Args:
            settings: Application settings
            image_normalizer: Service for normalizing images
            image_description_service: Service for generating image descriptions
            alt_text_service: Service for generating alt text
            review_service: Service for reviewing generated content
        """
        self.settings = settings
        self.image_normalizer = image_normalizer
        self.image_description_service = image_description_service
        self.alt_text_service = alt_text_service
        self.review_service = review_service

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

        full_description = full_desc_result.get("FULL_DESCRIPTION", "")
        transcript = full_desc_result.get("TRANSCRIPT", "")
        safety_form = full_desc_result.get("SAFETY_ASSESSMENT_FORM", {})
        safety_reasoning = full_desc_result.get("SAFETY_ASSESSMENT_REASONING", "")

        # Summarize the full description into alt text
        alt_text = self.alt_text_service.generate_alt_text(full_description)

        # Generate review assessment
        review_assessment_result = self.review_service.generate_review_assessment(
            full_description,
            transcript,
            safety_form,
            safety_reasoning,
            alt_text
        )
        review_assessment = self._parse_review_assessment(review_assessment_result)

        scores = [s for s in [safety_assessment.risk_score, review_assessment.risk_score] if s is not None]
        overall_risk_Score = round(sum(scores) / len(scores)) if scores else None

        return DescriptionResult(
            full_description=full_description,
            alt_text=alt_text,
            transcript=transcript,
            safety_assessment=safety_assessment,
            review_assessment=review_assessment,
            overall_risk_Score=overall_risk_Score,
            version=VersionInfo(
                version=self.settings.app_version,
                models={
                    "full_desc": self.settings.litellm_full_desc_model,
                    "alt_text": self.settings.litellm_alt_text_model,
                    "review": self.settings.litellm_review_model
                },
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

        assessment = SafetyAssessment(
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
            reasoning=full_desc_result.get("SAFETY_ASSESSMENT_REASONING", "")
        )

        assessment.risk_score = calculate_risk_score(assessment)
        assessment.inconsistency_count = count_safety_inconsistencies(assessment)
        return assessment

    def _parse_review_assessment(self, review_result: dict) -> ReviewAssessment:
        """
        Parse review assessment from LLM response.

        Args:
            review_result: Dictionary containing LLM response with review assessment fields

        Returns:
            ReviewAssessment object
        """
        assessment = ReviewAssessment(
            biased_language=review_result.get("biased_language", "NO"),
            stereotyping=review_result.get("stereotyping", "NO"),
            value_judgments=review_result.get("value_judgments", "NO"),
            contradictions_between_texts=review_result.get("contradictions_between_texts", "NO"),
            contradictions_within_description=review_result.get("contradictions_within_description", "NO"),
            offensive_language=review_result.get("offensive_language", "NO"),
            inconsistent_demographics=review_result.get("inconsistent_demographics", "NO"),
            euphemistic_language=review_result.get("euphemistic_language", "NO"),
            people_first_language=review_result.get("people_first_language", "N/A"),
            unsupported_inferential_claims=review_result.get("unsupported_inferential_claims", "NO"),
            safety_assessment_consistency=review_result.get("safety_assessment_consistency", "CONSISTENT"),
            concerns_for_review=review_result.get("concerns_for_review", []),
            source_content_warnings=review_result.get("source_content_warnings", [])
        )
        assessment.risk_score = calculate_review_risk_score(assessment)
        return assessment
