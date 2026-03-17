"""Tests for the review risk scoring service."""
import pytest

from app.models.describe_response import ReviewAssessment
from app.services.review_risk_scoring_service import (
    _MAX_POSSIBLE_SCORE,
    REVIEW_FIELD_WEIGHTS,
    calculate_review_risk_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_assessment(**overrides) -> ReviewAssessment:
    """Build a ReviewAssessment with all zero-weighted values, allowing targeted overrides."""
    defaults = dict(
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
        concerns_for_review=[],
        source_content_warnings=[],
    )
    defaults.update(overrides)
    return ReviewAssessment(**defaults)


# Thresholds used throughout — deliberately loose so tests survive weight tuning.
LOW_SCORE_THRESHOLD = 34
HIGH_SCORE_THRESHOLD = 66


# ---------------------------------------------------------------------------
# Boundary scores: 0 and 100
# ---------------------------------------------------------------------------

def test_score_is_zero_for_all_zero_weighted_values():
    """All zero-weighted field values should produce a score of 0."""
    assessment = make_assessment()
    assert calculate_review_risk_score(assessment) == 0


def test_score_is_100_for_all_maximum_weighted_values():
    """All maximum-weighted field values should produce a score of 100."""
    assessment = make_assessment(
        biased_language="YES",
        stereotyping="YES",
        value_judgments="YES",
        contradictions_between_texts="YES",
        contradictions_within_description="YES",
        offensive_language="YES",
        inconsistent_demographics="YES",
        euphemistic_language="YES",
        people_first_language="NOT_USED",
        unsupported_inferential_claims="YES",
        safety_assessment_consistency="INCONSISTENT",
        concerns_for_review=["Multiple issues detected"],
    )
    assert calculate_review_risk_score(assessment) == 100


# ---------------------------------------------------------------------------
# Qualitative range tests
# ---------------------------------------------------------------------------

def test_low_risk_score_for_clean_inconsistent_review():
    """A fully clean review with a inconsistent safety assessment should score in the low range."""
    assessment = make_assessment(safety_assessment_consistency="INCONSISTENT")
    score = calculate_review_risk_score(assessment)
    assert score < LOW_SCORE_THRESHOLD, (
        f"Expected low score (< {LOW_SCORE_THRESHOLD}), got {score}"
    )


def test_medium_risk_score_for_review_with_possible_issues():
    """
    A review with several POSSIBLY flags plus people-first language issues
    should land in the medium range.
    """
    assessment = make_assessment(
        biased_language="POSSIBLY",
        stereotyping="POSSIBLY",
        value_judgments="POSSIBLY",
        euphemistic_language="POSSIBLY",
        unsupported_inferential_claims="POSSIBLY",
        people_first_language="NOT_USED",
        safety_assessment_consistency="INCONSISTENT",
    )
    score = calculate_review_risk_score(assessment)
    assert LOW_SCORE_THRESHOLD <= score <= HIGH_SCORE_THRESHOLD, (
        f"Expected medium score ({LOW_SCORE_THRESHOLD}–{HIGH_SCORE_THRESHOLD}), got {score}"
    )


def test_high_risk_score_for_review_with_definitive_issues():
    """
    A review with definitive YES findings across major fields should score
    in the high range.
    """
    assessment = make_assessment(
        biased_language="YES",
        stereotyping="YES",
        value_judgments="YES",
        contradictions_between_texts="YES",
        inconsistent_demographics="YES",
        unsupported_inferential_claims="YES",
        safety_assessment_consistency="INCONSISTENT",
    )
    score = calculate_review_risk_score(assessment)
    assert score > HIGH_SCORE_THRESHOLD, (
        f"Expected high score (> {HIGH_SCORE_THRESHOLD}), got {score}"
    )


# ---------------------------------------------------------------------------
# Individual field ordering
# ---------------------------------------------------------------------------

def test_yes_scores_higher_than_possibly_for_biased_language():
    possibly = make_assessment(biased_language="POSSIBLY")
    yes = make_assessment(biased_language="YES")
    assert calculate_review_risk_score(yes) > calculate_review_risk_score(possibly)


def test_yes_scores_higher_than_possibly_for_unsupported_inferential_claims():
    possibly = make_assessment(unsupported_inferential_claims="POSSIBLY")
    yes = make_assessment(unsupported_inferential_claims="YES")
    assert calculate_review_risk_score(yes) > calculate_review_risk_score(possibly)


def test_contradictions_between_texts_scores_higher_than_baseline():
    baseline = make_assessment()
    with_contradiction = make_assessment(contradictions_between_texts="YES")
    assert calculate_review_risk_score(with_contradiction) > calculate_review_risk_score(baseline)


def test_people_first_not_used_scores_higher_than_na():
    na = make_assessment(people_first_language="N/A")
    not_used = make_assessment(people_first_language="NOT_USED")
    assert calculate_review_risk_score(not_used) > calculate_review_risk_score(na)


def test_people_first_used_and_na_score_equally():
    """Both USED and N/A carry weight 0 — they should produce the same score."""
    used = make_assessment(people_first_language="USED")
    na = make_assessment(people_first_language="N/A")
    assert calculate_review_risk_score(used) == calculate_review_risk_score(na)


def test_contradictions_within_description_possibly_scores_higher_than_no():
    """POSSIBLY is the maximum-weighted value for contradictions_within_description."""
    no = make_assessment(contradictions_within_description="NO")
    possibly = make_assessment(contradictions_within_description="POSSIBLY")
    assert calculate_review_risk_score(possibly) > calculate_review_risk_score(no)


def test_multiple_issues_compound():
    """Stacking multiple YES/POSSIBLY findings should accumulate the score."""
    one_issue = make_assessment(biased_language="YES")
    two_issues = make_assessment(biased_language="YES", stereotyping="YES")
    assert calculate_review_risk_score(two_issues) > calculate_review_risk_score(one_issue)


# ---------------------------------------------------------------------------
# Score properties
# ---------------------------------------------------------------------------

def test_score_is_always_within_valid_range():
    """The risk score should always be between 0 and 100 inclusive."""
    assessments = [
        make_assessment(),
        make_assessment(safety_assessment_consistency="INCONSISTENT"),
        make_assessment(
            biased_language="YES",
            stereotyping="YES",
            value_judgments="YES",
            contradictions_between_texts="YES",
            contradictions_within_description="POSSIBLY",
            offensive_language="YES",
            inconsistent_demographics="YES",
            euphemistic_language="YES",
            people_first_language="NOT_USED",
            unsupported_inferential_claims="YES",
            safety_assessment_consistency="INCONSISTENT",
        ),
    ]
    for assessment in assessments:
        score = calculate_review_risk_score(assessment)
        assert 0 <= score <= 100, f"Score {score} is outside the valid range [0, 100]"


def test_max_possible_score_is_positive():
    """Weight tables must be configured so that a non-zero maximum exists."""
    assert _MAX_POSSIBLE_SCORE > 0


def test_concerns_for_review_does_not_affect_score():
    """The free-text concerns list should have no impact on the numeric score."""
    without_concerns = make_assessment(concerns_for_review=[])
    with_concerns = make_assessment(
        concerns_for_review=["Biased framing", "Offensive terminology detected"]
    )
    assert calculate_review_risk_score(without_concerns) == calculate_review_risk_score(with_concerns)


def test_source_content_warnings_does_not_affect_score():
    """Source content warnings should have no impact on the numeric score."""
    without_warnings = make_assessment(source_content_warnings=[])
    with_warnings = make_assessment(
        source_content_warnings=["Contains historically dated racial language", "Offensive imagery"]
    )
    assert calculate_review_risk_score(without_warnings) == calculate_review_risk_score(with_warnings)
