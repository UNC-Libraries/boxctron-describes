"""Service for calculating a numeric risk score from a ReviewAssessment."""
from app.models.describe_response import ReviewAssessment

# Weights for ReviewAssessment fields.
# Each key maps a field value to its risk weight contribution.
REVIEW_FIELD_WEIGHTS: dict[str, dict[str, int]] = {
    "biased_language": {
        "NO": 0,
        "POSSIBLY": 2,
        "YES": 5,
    },
    "stereotyping": {
        "NO": 0,
        "POSSIBLY": 2,
        "YES": 5,
    },
    "value_judgments": {
        "NO": 0,
        "POSSIBLY": 2,
        "YES": 5,
    },
    "contradictions_between_texts": {
        "NO": 0,
        "YES": 10,
    },
    "contradictions_within_description": {
        "NO": 0,
        "POSSIBLY": 2,
        "YES": 10,
    },
    "offensive_language": {
        "NO": 0,
        "YES": 2,
    },
    "inconsistent_demographics": {
        "NO": 0,
        "YES": 5,
    },
    "euphemistic_language": {
        "NO": 0,
        "POSSIBLY": 2,
        "YES": 5,
    },
    "people_first_language": {
        "USED": 0,
        "NOT_USED": 5,
        "N/A": 0,
    },
    "unsupported_inferential_claims": {
        "NO": 0,
        "POSSIBLY": 2,
        "YES": 10,
    },
    "safety_assessment_consistency": {
        "CONSISTENT": 0,
        "INCONSISTENT": 10,
    },
}

# Pre-computed maximum possible score across all fields.
# Recalculate this whenever any weight table is updated.
_MAX_POSSIBLE_SCORE: int = sum(
    max(weights.values()) for weights in REVIEW_FIELD_WEIGHTS.values()
)


def calculate_review_risk_score(assessment: ReviewAssessment) -> int:
    """Calculate a normalized risk score from a ReviewAssessment.

    The raw score is the sum of per-field risk weights. It is then divided by
    the maximum possible raw score and multiplied by 100 to produce a
    whole-number percentage in the range [0, 100].

    Note: concerns_for_review is a free-text list and is not included in
    scoring — it is descriptive context for human reviewers.

    Args:
        assessment: The ReviewAssessment to score.

    Returns:
        An integer risk score between 0 and 100 (inclusive).
    """

    raw_score = 0
    for field, weights in REVIEW_FIELD_WEIGHTS.items():
        value = getattr(assessment, field)
        raw_score += weights.get(value, 0)

    return round((raw_score / _MAX_POSSIBLE_SCORE) * 100)
