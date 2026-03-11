"""Tests for the review form expander."""
import pytest

from app.services.review_form_expander import expand_review_form


@pytest.fixture
def short_form_no_concerns():
    """A typical short-form review assessment with no issues."""
    return {
        "bias": "N",
        "stereo": "N",
        "val_judg": "N",
        "contra_btwn": "N",
        "contra_within": "N",
        "offensive": "N",
        "incon_demog": "N",
        "euphemism": "N",
        "ppl_first": "NA",
        "unsup_infer": "N",
        "safety_consist": "CON",
        "concerns": []
    }


@pytest.fixture
def short_form_with_concerns():
    """A short-form review assessment with elevated values."""
    return {
        "bias": "P",
        "stereo": "Y",
        "val_judg": "P",
        "contra_btwn": "Y",
        "contra_within": "P",
        "offensive": "Y",
        "incon_demog": "Y",
        "euphemism": "Y",
        "ppl_first": "NU",
        "unsup_infer": "Y",
        "safety_consist": "INCON",
        "concerns": ["Stereotyping detected", "Offensive language found"]
    }


def test_expand_no_concerns(short_form_no_concerns):
    """Expansion of a typical all-clear review."""
    result = expand_review_form(short_form_no_concerns)

    assert result["biased_language"] == "NO"
    assert result["stereotyping"] == "NO"
    assert result["value_judgments"] == "NO"
    assert result["contradictions_between_texts"] == "NO"
    assert result["contradictions_within_description"] == "NO"
    assert result["offensive_language"] == "NO"
    assert result["inconsistent_demographics"] == "NO"
    assert result["euphemistic_language"] == "NO"
    assert result["people_first_language"] == "N/A"
    assert result["unsupported_inferential_claims"] == "NO"
    assert result["safety_assessment_consistency"] == "CONSISTENT"
    assert result["concerns_for_review"] == []


def test_expand_with_concerns(short_form_with_concerns):
    """Expansion of a review with elevated risk values."""
    result = expand_review_form(short_form_with_concerns)

    assert result["biased_language"] == "POSSIBLY"
    assert result["stereotyping"] == "YES"
    assert result["value_judgments"] == "POSSIBLY"
    assert result["contradictions_between_texts"] == "YES"
    assert result["contradictions_within_description"] == "POSSIBLY"
    assert result["offensive_language"] == "YES"
    assert result["inconsistent_demographics"] == "YES"
    assert result["euphemistic_language"] == "YES"
    assert result["people_first_language"] == "NOT_USED"
    assert result["unsupported_inferential_claims"] == "YES"
    assert result["safety_assessment_consistency"] == "INCONSISTENT"
    assert result["concerns_for_review"] == ["Stereotyping detected", "Offensive language found"]


def test_expand_people_first_used():
    """USED value expands correctly."""
    short = {
        "bias": "N", "stereo": "N", "val_judg": "N",
        "contra_btwn": "N", "contra_within": "N", "offensive": "N",
        "incon_demog": "N", "euphemism": "N", "ppl_first": "U",
        "unsup_infer": "N", "safety_consist": "CON", "concerns": []
    }
    result = expand_review_form(short)
    assert result["people_first_language"] == "USED"


def test_expand_preserves_concerns_list():
    """Concerns list (free text) is preserved as-is."""
    short = {
        "bias": "N", "stereo": "N", "val_judg": "N",
        "contra_btwn": "N", "contra_within": "N", "offensive": "N",
        "incon_demog": "N", "euphemism": "N", "ppl_first": "NA",
        "unsup_infer": "N", "safety_consist": "CON",
        "concerns": ["Issue one", "Issue two", "Issue three"]
    }
    result = expand_review_form(short)
    assert result["concerns_for_review"] == ["Issue one", "Issue two", "Issue three"]


def test_expand_unknown_key_raises():
    """Unknown short key raises ValueError with the key name in the message."""
    with pytest.raises(ValueError, match="unknown_field"):
        expand_review_form({"unknown_field": "N"})


def test_expand_unknown_value_raises():
    """Unknown value for a field raises ValueError naming the field."""
    with pytest.raises(ValueError, match="bias"):
        expand_review_form({
            "bias": "MAYBE",
            "stereo": "N", "val_judg": "N",
            "contra_btwn": "N", "contra_within": "N", "offensive": "N",
            "incon_demog": "N", "euphemism": "N", "ppl_first": "NA",
            "unsup_infer": "N", "safety_consist": "CON", "concerns": []
        })
