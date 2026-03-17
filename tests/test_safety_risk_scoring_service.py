"""Tests for the safety risk scoring service."""
import pytest

from app.models.describe_response import SafetyAssessment, SymbolsPresent, TextCharacteristics
from app.services.safety_risk_scoring_service import (
    _MAX_POSSIBLE_SCORE,
    calculate_risk_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_assessment(**overrides) -> SafetyAssessment:
    """Build a SafetyAssessment with all-minimum-risk values, allowing targeted overrides."""
    defaults = dict(
        people_visible="NO",
        demographics_described="NO",
        misidentification_risk_people="LOW",
        minors_present="NO",
        named_individuals_claimed="NO",
        violent_content="NONE",
        racial_violence_oppression="NONE",
        nudity="NONE",
        sexual_content="NONE",
        symbols_present=SymbolsPresent(
            types=["NONE"],
            names=[],
            misidentification_risk="LOW",
        ),
        stereotyping_present="NO",
        atrocities_depicted="NO",
        text_characteristics=TextCharacteristics(
            text_present="NO",
            text_type="N/A",
            legibility="N/A",
        ),
        reasoning=None,
    )
    defaults.update(overrides)
    return SafetyAssessment(**defaults)


# Thresholds used throughout — deliberately loose so tests survive weight tuning.
LOW_SCORE_THRESHOLD = 34
HIGH_SCORE_THRESHOLD = 66


# ---------------------------------------------------------------------------
# Boundary scores: 0 and 100
# ---------------------------------------------------------------------------

def test_score_is_zero_for_all_minimum_risk_values():
    """All minimum-risk field values should produce a score of 0."""
    assessment = make_assessment()
    assert calculate_risk_score(assessment) == 0


def test_score_is_100_for_all_maximum_risk_values():
    """All maximum-risk field values should produce a score of 100."""
    assessment = make_assessment(
        people_visible="YES",
        demographics_described="YES",
        misidentification_risk_people="HIGH",
        minors_present="YES",
        named_individuals_claimed="YES",
        violent_content="DEPICTED",
        racial_violence_oppression="DEPICTED",
        nudity="FULL",
        sexual_content="EXPLICIT",
        symbols_present=SymbolsPresent(
            types=["HATE"],
            names=[],
            misidentification_risk="HIGH",
        ),
        stereotyping_present="YES",
        atrocities_depicted="YES",
        text_characteristics=TextCharacteristics(
            text_present="YES",
            text_type="HANDWRITTEN_CURSIVE",
            legibility="DIFFICULT",
        ),
    )
    assert calculate_risk_score(assessment) == 100


# ---------------------------------------------------------------------------
# Qualitative range tests
# ---------------------------------------------------------------------------

def test_low_risk_score_for_benign_portrait():
    """A people-visible image with no other risk factors should score in the low range."""
    assessment = make_assessment(people_visible="YES")
    score = calculate_risk_score(assessment)
    assert score < LOW_SCORE_THRESHOLD, (
        f"Expected low score (< {LOW_SCORE_THRESHOLD}), got {score}"
    )


def test_medium_risk_score_for_moderately_sensitive_image():
    """
    An image with people, claimed identities, implied content, possible stereotyping,
    and cultural symbols should land in the medium range.
    """
    assessment = make_assessment(
        people_visible="YES",
        demographics_described="YES",
        misidentification_risk_people="MEDIUM",
        named_individuals_claimed="YES",
        violent_content="IMPLIED",
        sexual_content="SUGGESTIVE",
        stereotyping_present="POSSIBLY",
        symbols_present=SymbolsPresent(
            types=["CULTURAL"],
            names=[],
            misidentification_risk="MEDIUM",
        ),
        text_characteristics=TextCharacteristics(
            text_present="YES",
            text_type="PRINTED",
            legibility="CLEAR",
        ),
    )
    score = calculate_risk_score(assessment)
    assert LOW_SCORE_THRESHOLD <= score <= HIGH_SCORE_THRESHOLD, (
        f"Expected medium score ({LOW_SCORE_THRESHOLD}–{HIGH_SCORE_THRESHOLD}), got {score}"
    )


def test_high_risk_score_for_violent_image_with_minors():
    """
    An image depicting violence, racial oppression, minors, and hate symbols
    should score in the high range.
    """
    assessment = make_assessment(
        people_visible="YES",
        demographics_described="YES",
        misidentification_risk_people="HIGH",
        minors_present="YES",
        violent_content="DEPICTED",
        racial_violence_oppression="DEPICTED",
        stereotyping_present="YES",
        atrocities_depicted="YES",
        symbols_present=SymbolsPresent(
            types=["HATE"],
            names=[],
            misidentification_risk="HIGH",
        ),
    )
    score = calculate_risk_score(assessment)
    assert score > HIGH_SCORE_THRESHOLD, (
        f"Expected high score (> {HIGH_SCORE_THRESHOLD}), got {score}"
    )


# ---------------------------------------------------------------------------
# Symbol type handling
# ---------------------------------------------------------------------------

def test_symbol_type_uses_highest_weight_when_multiple_types_present():
    """When multiple symbol types are listed, the highest-weighted one should dominate."""
    low_symbols = make_assessment(
        symbols_present=SymbolsPresent(
            types=["BRAND"],
            names=[],
            misidentification_risk="LOW",
        )
    )
    mixed_symbols = make_assessment(
        symbols_present=SymbolsPresent(
            types=["BRAND", "HATE"],
            names=[],
            misidentification_risk="LOW",
        )
    )
    hate_only = make_assessment(
        symbols_present=SymbolsPresent(
            types=["HATE"],
            names=[],
            misidentification_risk="LOW",
        )
    )
    assert calculate_risk_score(mixed_symbols) == calculate_risk_score(hate_only)
    assert calculate_risk_score(mixed_symbols) > calculate_risk_score(low_symbols)


def test_none_symbol_type_does_not_increase_score():
    """NONE symbol type should add no risk beyond the baseline of zero."""
    assessment = make_assessment()
    assert calculate_risk_score(assessment) == 0


# ---------------------------------------------------------------------------
# Text characteristics
# ---------------------------------------------------------------------------

def test_handwritten_cursive_scores_higher_than_printed():
    """Handwritten cursive text should contribute more risk than printed text."""
    printed = make_assessment(
        text_characteristics=TextCharacteristics(
            text_present="YES",
            text_type="PRINTED",
            legibility="CLEAR",
        )
    )
    cursive = make_assessment(
        text_characteristics=TextCharacteristics(
            text_present="YES",
            text_type="HANDWRITTEN_CURSIVE",
            legibility="CLEAR",
        )
    )
    assert calculate_risk_score(cursive) > calculate_risk_score(printed)


def test_illegible_text_scores_higher_than_clear():
    """Difficult-to-read text should contribute more risk than clearly legible text."""
    clear = make_assessment(
        text_characteristics=TextCharacteristics(
            text_present="YES",
            text_type="PRINTED",
            legibility="CLEAR",
        )
    )
    illegible = make_assessment(
        text_characteristics=TextCharacteristics(
            text_present="YES",
            text_type="PRINTED",
            legibility="ILLEGIBLE",
        )
    )
    assert calculate_risk_score(illegible) > calculate_risk_score(clear)


# ---------------------------------------------------------------------------
# Score properties
# ---------------------------------------------------------------------------

def test_score_is_always_within_valid_range():
    """The risk score should always be between 0 and 100 inclusive."""
    assessments = [
        make_assessment(),
        make_assessment(people_visible="YES", violent_content="DEPICTED"),
        make_assessment(
            people_visible="YES",
            demographics_described="YES",
            misidentification_risk_people="HIGH",
            minors_present="YES",
            named_individuals_claimed="YES",
            violent_content="DEPICTED",
            racial_violence_oppression="DEPICTED",
            nudity="FULL",
            sexual_content="EXPLICIT",
            symbols_present=SymbolsPresent(
                types=["HATE"], names=[], misidentification_risk="HIGH"
            ),
            stereotyping_present="YES",
            atrocities_depicted="YES",
            text_characteristics=TextCharacteristics(
                text_present="YES", text_type="HANDWRITTEN_CURSIVE", legibility="DIFFICULT"
            ),
        ),
    ]
    for assessment in assessments:
        score = calculate_risk_score(assessment)
        assert 0 <= score <= 100, f"Score {score} is outside the valid range [0, 100]"


def test_max_possible_score_is_positive():
    """Weight tables must be configured so that a non-zero maximum exists."""
    assert _MAX_POSSIBLE_SCORE > 0


# ---------------------------------------------------------------------------
# UNKNOWN field values
# ---------------------------------------------------------------------------

def test_unknown_people_scores_between_no_and_yes():
    """people_visible=UNKNOWN should score higher than NO but lower than YES."""
    score_no = calculate_risk_score(make_assessment(people_visible="NO"))
    score_unknown = calculate_risk_score(make_assessment(people_visible="UNKNOWN"))
    score_yes = calculate_risk_score(make_assessment(people_visible="YES"))
    assert score_no < score_unknown < score_yes


def test_unknown_minors_scores_between_no_and_yes():
    """minors_present=UNKNOWN should score higher than NO but lower than YES."""
    score_no = calculate_risk_score(make_assessment(people_visible="YES", minors_present="NO"))
    score_unknown = calculate_risk_score(make_assessment(people_visible="YES", minors_present="UNKNOWN"))
    score_yes = calculate_risk_score(make_assessment(people_visible="YES", minors_present="YES"))
    assert score_no < score_unknown < score_yes
