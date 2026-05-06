"""Service for calculating a numeric risk score from a SafetyAssessment."""
from app.models.describe_response import SafetyAssessment

# Weights for top-level SafetyAssessment fields.
# Each key maps a field value to its risk weight contribution.
SAFETY_FIELD_WEIGHTS: dict[str, dict[str, int]] = {
    "people_visible": {
        "YES": 20,
        "NO": 0,
        "UNKNOWN": 10,
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
        "UNKNOWN": 5,
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
    "image_quality": {
        "UNIMPAIRED": 0,
        "DEGRADED": 5,
        "IMPAIRED": 15,
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
        "NONE": 0,
        "INCIDENTAL": 1,
        "SIGNIFICANT": 2,
    },
    "text_type": {
        "N/A": 0,
        "PRINTED": 1,
        "TYPED": 0,
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
    "sensitivity": {
        "N/A": 0,
        "NONE": 0,
        "SENSITIVE": 10,
    },
    "language": {
        "UNKNOWN": 3,
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

_PRACTICAL_MAX_FRACTION: float = 0.60

_EFFECTIVE_MAX_SCORE: float = _MAX_POSSIBLE_SCORE * _PRACTICAL_MAX_FRACTION

# Scaling factor applied to text_type and legibility weights when text_present is INCIDENTAL.
# Sensitivity is not scaled — a racial slur on a background sign still carries full weight.
_INCIDENTAL_TEXT_FACTOR: float = 0.2


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
    text_chars = assessment.text_characteristics
    raw_score += TEXT_FIELD_WEIGHTS["text_present"].get(text_chars.text_present, 0)

    text_scale = _INCIDENTAL_TEXT_FACTOR if text_chars.text_present == "INCIDENTAL" else 1.0
    raw_score += TEXT_FIELD_WEIGHTS["text_type"].get(text_chars.text_type, 0) * text_scale
    raw_score += TEXT_FIELD_WEIGHTS["legibility"].get(text_chars.legibility, 0) * text_scale
    raw_score += TEXT_FIELD_WEIGHTS["sensitivity"].get(text_chars.sensitivity, 0)
    if text_chars.language is not None:
        raw_score += TEXT_FIELD_WEIGHTS["language"].get(text_chars.language, 0)

    return min(100, round((raw_score / _EFFECTIVE_MAX_SCORE) * 100))
