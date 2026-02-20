"""
Response models for the API endpoints.
"""
from typing import Optional, List, Literal, Dict
from pydantic import BaseModel, ConfigDict, Field


class SymbolsPresent(BaseModel):
    """Symbols detected in the image."""

    types: List[Literal["NONE", "CULTURAL", "RELIGIOUS", "POLITICAL", "HATE", "BRAND"]] = Field(
        ...,
        description="Types of symbols present in the image"
    )

    names: List[str] = Field(
        ...,
        description="Names or descriptions of identified symbols"
    )

    misidentification_risk: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ...,
        description="Risk level of symbol misidentification"
    )


class TextCharacteristics(BaseModel):
    """Characteristics of text present in the image."""

    text_present: Literal["YES", "NO"] = Field(
        ...,
        description="Whether text is present in the image"
    )

    text_type: Literal["N/A", "PRINTED", "TYPED", "HANDWRITTEN_PRINT", "HANDWRITTEN_CURSIVE", "MIXED"] = Field(
        ...,
        description="Type of text present"
    )

    legibility: Literal["N/A", "CLEAR", "PARTIALLY_CLEAR", "DIFFICULT", "ILLEGIBLE"] = Field(
        ...,
        description="How legible the text is"
    )


class SafetyAssessment(BaseModel):
    """Safety assessment of the image content."""

    people_visible: Literal["YES", "NO"] = Field(
        ...,
        description="Whether people are visible in the image"
    )

    demographics_described: Literal["YES", "NO"] = Field(
        ...,
        description="Whether demographics are described"
    )

    misidentification_risk_people: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ...,
        description="Risk level of misidentifying people"
    )

    minors_present: Literal["YES", "NO"] = Field(
        ...,
        description="Whether minors are present in the image"
    )

    named_individuals_claimed: Literal["YES", "NO"] = Field(
        ...,
        description="Whether named individuals are claimed to be present"
    )

    violent_content: Literal["NONE", "IMPLIED", "DEPICTED"] = Field(
        ...,
        description="Level of violent content"
    )

    racial_violence_oppression: Literal["NONE", "IMPLIED", "DEPICTED"] = Field(
        ...,
        description="Level of racial violence or oppression depicted"
    )

    nudity: Literal["NONE", "PARTIAL", "FULL"] = Field(
        ...,
        description="Level of nudity present"
    )

    sexual_content: Literal["NONE", "SUGGESTIVE", "EXPLICIT"] = Field(
        ...,
        description="Level of sexual content"
    )

    symbols_present: SymbolsPresent = Field(
        ...,
        description="Information about symbols present in the image"
    )

    stereotyping_present: Literal["NO", "POSSIBLY", "YES"] = Field(
        ...,
        description="Whether stereotyping is present"
    )

    atrocities_depicted: Literal["NO", "YES"] = Field(
        ...,
        description="Whether atrocities are depicted"
    )

    text_characteristics: TextCharacteristics = Field(
        ...,
        description="Characteristics of text in the image"
    )

    confidence: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ...,
        description="Overall confidence in the safety assessment"
    )

    reasoning: Optional[str] = Field(
        None,
        description="Reasoning behind the safety assessment"
    )


class ReviewAssessment(BaseModel):
    """Assessment for content review requirements."""

    biased_language: Literal["NO", "POSSIBLY", "YES"] = Field(
        ...,
        description="Whether biased language is detected"
    )

    stereotyping: Literal["NO", "POSSIBLY", "YES"] = Field(
        ...,
        description="Whether stereotyping is detected"
    )

    value_judgments: Literal["NO", "POSSIBLY", "YES"] = Field(
        ...,
        description="Whether value judgments are present"
    )

    contradictions_between_texts: Literal["NO", "YES"] = Field(
        ...,
        description="Whether there are contradictions between different text outputs"
    )

    contradictions_within_description: Literal["NO", "POSSIBLY", "YES"] = Field(
        ...,
        description="Whether there are internal contradictions in the description"
    )

    offensive_language: Literal["NO", "YES"] = Field(
        ...,
        description="Whether offensive language is present"
    )

    inconsistent_demographics: Literal["NO", "YES"] = Field(
        ...,
        description="Whether demographic descriptions are inconsistent"
    )

    euphemistic_language: Literal["NO", "POSSIBLY", "YES"] = Field(
        ...,
        description="Whether euphemistic language is used"
    )

    people_first_language: Literal["USED", "NOT_USED", "N/A"] = Field(
        ...,
        description="Whether people-first language is used"
    )

    unsupported_inferential_claims: Literal["NO", "POSSIBLY", "YES"] = Field(
        ...,
        description="Whether there are unsupported inferential claims"
    )

    safety_assessment_consistency: Literal["CONSISTENT", "INCONSISTENT"] = Field(
        ...,
        description="Whether the safety assessment is consistent with the description"
    )

    concerns_for_review: List[str] = Field(
        ...,
        description="List of specific concerns that may require human review"
    )


class VersionInfo(BaseModel):
    """Version information about the processing."""

    version: str = Field(
        ...,
        description="Version of the application"
    )

    models: Dict[str, str] = Field(
        ...,
        description="Map of task names to model names used for processing"
    )

    timestamp: str = Field(
        ...,
        description="ISO8601 timestamp when the response was generated"
    )


class DescriptionResult(BaseModel):
    """Contains the generated description and assessment data."""

    full_description: str = Field(
        ...,
        description="Long form description of the image"
    )

    alt_text: str = Field(
        ...,
        description="Short form alt text description of the image"
    )

    transcript: str = Field(
        ...,
        description="Transcript of any text contained in the image (empty if no text present)"
    )

    safety_assessment: SafetyAssessment = Field(
        ...,
        description="Safety information about the image content"
    )

    review_assessment: ReviewAssessment = Field(
        ...,
        description="Analysis for determining if the image needs human review"
    )

    version: VersionInfo = Field(
        ...,
        description="Version information about models used and generation timestamp"
    )


class DescribeResponse(BaseModel):
    """
    Response model for the describe endpoints.
    """

    success: bool = Field(
        ...,
        description="Indicates whether the request was successful"
    )

    filename: str = Field(
        ...,
        description="The filename that was processed"
    )

    result: Optional[DescriptionResult] = Field(
        None,
        description="The description result, present if success is True"
    )

    error: Optional[str] = Field(
        None,
        description="Error message if success is False"
    )

    processing_time_ms: Optional[float] = Field(
        None,
        description="Time taken to process the request in milliseconds"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "filename": "photo.jpg",
                "result": {
                    "full_description": "A scenic mountain landscape with snow-capped peaks rising above a forested valley",
                    "alt_text": "Mountain landscape with snow-covered peaks",
                    "transcript": "",
                    "safety_assessment": {
                        "people_visible": "NO",
                        "demographics_described": "NO",
                        "misidentification_risk_people": "LOW",
                        "minors_present": "NO",
                        "named_individuals_claimed": "NO",
                        "violent_content": "NONE",
                        "racial_violence_oppression": "NONE",
                        "nudity": "NONE",
                        "sexual_content": "NONE",
                        "symbols_present": {
                            "types": ["NONE"],
                            "names": [],
                            "misidentification_risk": "LOW"
                        },
                        "stereotyping_present": "NO",
                        "atrocities_depicted": "NO",
                        "text_characteristics": {
                            "text_present": "NO",
                            "text_type": "N/A",
                            "legibility": "N/A"
                        },
                        "confidence": "HIGH",
                        "reasoning": "Clear natural landscape with no sensitive content"
                    },
                    "review_assessment": {
                        "biased_language": "NO",
                        "stereotyping": "NO",
                        "value_judgments": "NO",
                        "contradictions_between_texts": "NO",
                        "contradictions_within_description": "NO",
                        "offensive_language": "NO",
                        "inconsistent_demographics": "NO",
                        "euphemistic_language": "NO",
                        "people_first_language": "N/A",
                        "unsupported_inferential_claims": "NO",
                        "safety_assessment_consistency": "CONSISTENT",
                        "concerns_for_review": []
                    },
                    "version": {
                        "version": "0.1.0",
                        "models": {
                            "full_desc": "gpt-4o-2024-08-06"
                        },
                        "timestamp": "2024-08-15T10:30:00Z"
                    }
                },
                "processing_time_ms": 1250.5
            }
        }
    )
