"""Service for detecting logical inconsistencies in a SafetyAssessment."""
from app.models.describe_response import SafetyAssessment


def count_safety_inconsistencies(assessment: SafetyAssessment) -> int:
    """Count the number of logical inconsistencies in a SafetyAssessment.

    Inconsistencies indicate the model made internally contradictory assessments,
    which typically warrants re-running the safety prompt rather than proceeding
    to downstream review steps.

    Args:
        assessment: The SafetyAssessment to check.

    Returns:
        The number of distinct inconsistencies detected (0 means fully consistent).
    """
    count = 0

    # Fields that are meaningless or contradictory when no people are visible.
    if assessment.people_visible == "NO":
        if assessment.demographics_described == "YES":
            count += 1
        if assessment.misidentification_risk_people != "LOW":
            count += 1
        if assessment.minors_present == "YES":
            count += 1
        if assessment.named_individuals_claimed == "YES":
            count += 1
        if assessment.nudity != "NONE":
            count += 1

    # When people presence is unknown, minors being YES is contradictory
    if assessment.people_visible == "UNKNOWN":
        if assessment.minors_present == "YES":
            count += 1
        if assessment.named_individuals_claimed == "YES":
            count += 1

    # NONE cannot coexist with other symbol types.
    if "NONE" in assessment.symbols_present.types and len(assessment.symbols_present.types) > 1:
        count += 1
    if "NONE" in assessment.symbols_present.types and assessment.symbols_present.misidentification_risk != "LOW":
        count += 1

    # Non-NONE symbol types should have at least one name provided.
    if any(t != "NONE" for t in assessment.symbols_present.types) and not assessment.symbols_present.names:
        count += 1

    # Text sub-fields should not be N/A when text is present.
    if assessment.text_characteristics.text_present == "YES":
        if assessment.text_characteristics.text_type == "N/A":
            count += 1
        if assessment.text_characteristics.legibility == "N/A":
            count += 1
        if assessment.text_characteristics.sensitivity == "N/A":
            count += 1

    return count
