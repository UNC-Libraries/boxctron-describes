"""Service for calculating a numeric risk score from a SafetyAssessment."""
from app.models.describe_response import SafetyAssessment

# Weights for top-level SafetyAssessment fields.
# Each key maps a field value to its risk weight contribution.
SAFETY_FIELD_WEIGHTS: dict[str, dict[str, int]] = {
    "people_visible": {
        "YES": 20,
        "NO": 0,
    },
    "demographics_described": {
        "YES": 10,
        "NO": 0,
    },
    "misidentification_risk_people": {
        "LOW": 0,
        "MEDIUM": 10,
        "HIGH": 20,
    },
    "minors_present": {
        "YES": 10,
        "NO": 0,
    },
    "named_individuals_claimed": {
        "YES": 10,
        "NO": 0,
    },
    "violent_content": {
        "NONE": 0,
        "IMPLIED": 1,
        "DEPICTED": 5,
    },
    "racial_violence_oppression": {
        "NONE": 0,
        "IMPLIED": 5,
        "DEPICTED": 10,
    },
    "nudity": {
        "NONE": 0,
        "PARTIAL": 1,
        "FULL": 5,
    },
    "sexual_content": {
        "NONE": 0,
        "SUGGESTIVE": 1,
        "EXPLICIT": 10,
    },
    "stereotyping_present": {
        "NO": 0,
        "POSSIBLY": 5,
        "YES": 10,
    },
    "atrocities_depicted": {
        "NO": 0,
        "YES": 10,
    },
    "confidence": {
        "LOW": 0,
        "MEDIUM": 0,
        "HIGH": 0,
    },
}

# Weights for each symbol type value.
# When multiple types are present, the highest-weighted type is used.
SYMBOL_TYPE_WEIGHTS: dict[str, int] = {
    "NONE": 0,
    "CULTURAL": 5,
    "RELIGIOUS": 5,
    "POLITICAL": 5,
    "HATE": 10,
    "BRAND": 1,
}

# Weights for the symbols_present.misidentification_risk field.
SYMBOL_MISIDENTIFICATION_RISK_WEIGHTS: dict[str, int] = {
    "LOW": 0,
    "MEDIUM": 2,
    "HIGH": 5,
}

# Weights for fields nested inside TextCharacteristics.
TEXT_FIELD_WEIGHTS: dict[str, dict[str, int]] = {
    "text_present": {
        "YES": 1,
        "NO": 0,
    },
    "text_type": {
        "N/A": 0,
        "PRINTED": 1,
        "TYPED": 1,
        "HANDWRITTEN_PRINT": 5,
        "HANDWRITTEN_CURSIVE": 10,
        "MIXED": 5,
    },
    "legibility": {
        "N/A": 0,
        "CLEAR": 0,
        "PARTIALLY_CLEAR": 2,
        "DIFFICULT": 10,
        "ILLEGIBLE": 10,
    },
}

# Pre-computed maximum possible score across all fields.
# Recalculate this whenever any weight table is updated.
_MAX_POSSIBLE_SCORE: int = (
    sum(max(weights.values()) for weights in SAFETY_FIELD_WEIGHTS.values())
    + max(SYMBOL_TYPE_WEIGHTS.values())
    + max(SYMBOL_MISIDENTIFICATION_RISK_WEIGHTS.values())
    + sum(max(weights.values()) for weights in TEXT_FIELD_WEIGHTS.values())
)


def calculate_risk_score(assessment: SafetyAssessment) -> int:
    """Calculate a normalized risk score from a SafetyAssessment.

    The raw score is the sum of per-field risk weights. It is then divided by
    the maximum possible raw score and multiplied by 100 to produce a
    whole-number percentage in the range [0, 100].

    Args:
        assessment: The SafetyAssessment to score.

    Returns:
        An integer risk score between 0 and 100 (inclusive).
    """
    raw_score = 0

    # Top-level fields
    for field, weights in SAFETY_FIELD_WEIGHTS.items():
        value = getattr(assessment, field)
        raw_score += weights.get(value, 0)

    # symbols_present.types — use the highest-weighted type present
    symbol_type_score = max(
        (SYMBOL_TYPE_WEIGHTS.get(t, 0) for t in assessment.symbols_present.types),
        default=0,
    )
    raw_score += symbol_type_score

    # symbols_present.misidentification_risk
    raw_score += SYMBOL_MISIDENTIFICATION_RISK_WEIGHTS.get(
        assessment.symbols_present.misidentification_risk, 0
    )

    # text_characteristics fields
    for field, weights in TEXT_FIELD_WEIGHTS.items():
        value = getattr(assessment.text_characteristics, field)
        raw_score += weights.get(value, 0)

    return round((raw_score / _MAX_POSSIBLE_SCORE) * 100)
