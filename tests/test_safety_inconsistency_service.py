"""Tests for the safety inconsistency service."""
import pytest

from app.models.describe_response import SafetyAssessment, SymbolsPresent, TextCharacteristics
from app.services.safety_inconsistency_service import count_safety_inconsistencies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_assessment(**overrides) -> SafetyAssessment:
    """Build a SafetyAssessment with all-consistent minimum-risk values, allowing overrides."""
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
        confidence="HIGH",
        reasoning=None,
    )
    defaults.update(overrides)
    return SafetyAssessment(**defaults)


# ---------------------------------------------------------------------------
# Baseline: no inconsistencies
# ---------------------------------------------------------------------------

def test_no_inconsistencies_for_fully_consistent_assessment():
    """A logically consistent all-clear assessment should return 0."""
    assert count_safety_inconsistencies(make_assessment()) == 0


def test_consistent_people_fields_do_not_trigger_inconsistency():
    """When people are visible, people-related fields should not flag inconsistencies."""
    assessment = make_assessment(
        people_visible="YES",
        demographics_described="YES",
        misidentification_risk_people="HIGH",
        minors_present="YES",
        named_individuals_claimed="YES",
        nudity="PARTIAL",
    )
    assert count_safety_inconsistencies(assessment) == 0


def test_consistent_text_fields_do_not_trigger_inconsistency():
    """Valid text field combinations should produce no inconsistencies."""
    has_text = make_assessment(
        text_characteristics=TextCharacteristics(
            text_present="YES", text_type="PRINTED", legibility="CLEAR"
        )
    )
    assert count_safety_inconsistencies(has_text) == 0


def test_single_symbol_type_without_none_is_consistent():
    """A single non-NONE symbol type with a name provided should not be flagged."""
    assessment = make_assessment(
        symbols_present=SymbolsPresent(types=["HATE"], names=["angryface"], misidentification_risk="LOW")
    )
    assert count_safety_inconsistencies(assessment) == 0


# ---------------------------------------------------------------------------
# People-visible inconsistencies (each triggers exactly one count)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("override,description", [
    ({"demographics_described": "YES"}, "demographics_described=YES with people_visible=NO"),
    ({"misidentification_risk_people": "MEDIUM"}, "misidentification_risk=MEDIUM with people_visible=NO"),
    ({"misidentification_risk_people": "HIGH"}, "misidentification_risk=HIGH with people_visible=NO"),
    ({"minors_present": "YES"}, "minors_present=YES with people_visible=NO"),
    ({"named_individuals_claimed": "YES"}, "named_individuals_claimed=YES with people_visible=NO"),
    ({"nudity": "PARTIAL"}, "nudity=PARTIAL with people_visible=NO"),
    ({"nudity": "FULL"}, "nudity=FULL with people_visible=NO"),
])
def test_each_people_field_inconsistency_counts_once(override, description):
    """Each contradictory people-related field should add exactly 1 to the count."""
    assert count_safety_inconsistencies(make_assessment(**override)) == 1, (
        f"Expected count of 1 for: {description}"
    )


# ---------------------------------------------------------------------------
# Symbol type inconsistency
# ---------------------------------------------------------------------------

def test_none_symbol_combined_with_other_type_counts_twice():
    """NONE alongside another symbol type with no names should count 2: mixed types + missing names."""
    assessment = make_assessment(
        symbols_present=SymbolsPresent(types=["NONE", "CULTURAL"], names=[], misidentification_risk="LOW")
    )
    assert count_safety_inconsistencies(assessment) == 2


def test_none_symbol_combined_with_other_type_and_names_counts_once():
    """NONE alongside another symbol type but with names provided should count only 1 (mixed types)."""
    assessment = make_assessment(
        symbols_present=SymbolsPresent(types=["NONE", "RELIGIOUS"], names=["cross"], misidentification_risk="LOW")
    )
    assert count_safety_inconsistencies(assessment) == 1


def test_none_symbol_combined_with_multiple_other_types_names_missing_counts_twice():
    """NONE alongside several symbol types with no names should count 2: mixed types + missing names."""
    assessment = make_assessment(
        symbols_present=SymbolsPresent(
            types=["NONE", "CULTURAL", "HATE"], names=[], misidentification_risk="LOW"
        )
    )
    assert count_safety_inconsistencies(assessment) == 2

def test_none_symbol_combined_with_elevated_misid_risk():
    """NONE alongside several symbol types is still a single inconsistency."""
    assessment = make_assessment(
        symbols_present=SymbolsPresent(
            types=["NONE"], names=[], misidentification_risk="MEDIUM"
        )
    )
    assert count_safety_inconsistencies(assessment) == 1


def test_non_none_symbol_types_with_no_names_counts_once():
    """Non-NONE symbol types present but names list is empty should add 1 to the count."""
    assessment = make_assessment(
        symbols_present=SymbolsPresent(types=["HATE"], names=[], misidentification_risk="LOW")
    )
    assert count_safety_inconsistencies(assessment) == 1


def test_non_none_symbol_types_with_names_is_consistent():
    """Non-NONE symbol types with at least one name provided should not be flagged."""
    assessment = make_assessment(
        symbols_present=SymbolsPresent(types=["CULTURAL", "RELIGIOUS"], names=["cross"], misidentification_risk="LOW")
    )
    assert count_safety_inconsistencies(assessment) == 0

# ---------------------------------------------------------------------------
# Text field inconsistencies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text_chars,description", [
    (
        TextCharacteristics(text_present="YES", text_type="N/A", legibility="CLEAR"),
        "text_present=YES but text_type=N/A",
    ),
    (
        TextCharacteristics(text_present="YES", text_type="PRINTED", legibility="N/A"),
        "text_present=YES but legibility=N/A",
    ),
])
def test_each_text_inconsistency_counts_once(text_chars, description):
    """Each contradictory text sub-field should add exactly 1 to the count."""
    assessment = make_assessment(text_characteristics=text_chars)
    assert count_safety_inconsistencies(assessment) == 1, (
        f"Expected count of 1 for: {description}"
    )


def test_both_text_sub_fields_na_with_text_present_counts_two():
    """Both text sub-fields being N/A when text is present should add 2 to the count."""
    assessment = make_assessment(
        text_characteristics=TextCharacteristics(
            text_present="YES", text_type="N/A", legibility="N/A"
        )
    )
    assert count_safety_inconsistencies(assessment) == 2


# ---------------------------------------------------------------------------
# Multiple inconsistencies compound
# ---------------------------------------------------------------------------

def test_multiple_people_inconsistencies_compound():
    """Multiple contradictory people-related fields should each add to the count."""
    assessment = make_assessment(
        demographics_described="YES",
        minors_present="YES",
    )
    assert count_safety_inconsistencies(assessment) == 2


def test_inconsistencies_across_groups_compound():
    """Inconsistencies from different groups (people, symbols, text) all accumulate."""
    assessment = make_assessment(
        demographics_described="YES",  # people group: +1
        symbols_present=SymbolsPresent(
            types=["NONE", "HATE"], names=["angryface"], misidentification_risk="LOW"
        ),  # symbol group: +2 (NONE+other)
        text_characteristics=TextCharacteristics(
            text_present="YES", text_type="N/A", legibility="N/A"
        ),  # text group: +2
    )
    assert count_safety_inconsistencies(assessment) == 4
