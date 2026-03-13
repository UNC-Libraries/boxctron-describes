"""Tests for the safety form expander."""
import pytest

from app.services.safety_form_expander import expand_safety_form


@pytest.fixture
def short_form_no_concerns():
    """A typical short-form safety assessment with no concerns."""
    return {
        "people": "N",
        "demog": "N",
        "misid_risk": "L",
        "minors": "N",
        "named_indiv": "N",
        "violence": "0",
        "racial_viol": "0",
        "nudity": "0",
        "sexual": "0",
        "symbols": {
            "types": ["0"],
            "names": [],
            "misid_risk": "L"
        },
        "stereotyping": "N",
        "atrocities": "N",
        "text_chars": {
            "present": "N",
            "type": "NA",
            "legib": "NA"
        }
    }


@pytest.fixture
def short_form_with_concerns():
    """A short-form safety assessment with elevated values."""
    return {
        "people": "Y",
        "demog": "Y",
        "misid_risk": "M",
        "minors": "N",
        "named_indiv": "Y",
        "violence": "IMP",
        "racial_viol": "DEP",
        "nudity": "PAR",
        "sexual": "SUG",
        "symbols": {
            "types": ["POL", "BRD"],
            "names": ["American flag", "Coca-Cola logo"],
            "misid_risk": "M"
        },
        "stereotyping": "P",
        "atrocities": "Y",
        "text_chars": {
            "present": "Y",
            "type": "HWCU",
            "legib": "DIF"
        }
    }


def test_expand_no_concerns(short_form_no_concerns):
    """Expansion of a typical all-clear assessment."""
    result = expand_safety_form(short_form_no_concerns)

    assert result["people_visible"] == "NO"
    assert result["demographics_described"] == "NO"
    assert result["misidentification_risk_people"] == "LOW"
    assert result["minors_present"] == "NO"
    assert result["named_individuals_claimed"] == "NO"
    assert result["violent_content"] == "NONE"
    assert result["racial_violence_oppression"] == "NONE"
    assert result["nudity"] == "NONE"
    assert result["sexual_content"] == "NONE"
    assert result["symbols_present"] == {
        "types": ["NONE"],
        "names": [],
        "misidentification_risk": "LOW"
    }
    assert result["stereotyping_present"] == "NO"
    assert result["atrocities_depicted"] == "NO"
    assert result["text_characteristics"] == {
        "text_present": "NO",
        "text_type": "N/A",
        "legibility": "N/A"
    }


def test_expand_with_concerns(short_form_with_concerns):
    """Expansion of an assessment with elevated risk values."""
    result = expand_safety_form(short_form_with_concerns)

    assert result["people_visible"] == "YES"
    assert result["demographics_described"] == "YES"
    assert result["misidentification_risk_people"] == "MEDIUM"
    assert result["minors_present"] == "NO"
    assert result["named_individuals_claimed"] == "YES"
    assert result["violent_content"] == "IMPLIED"
    assert result["racial_violence_oppression"] == "DEPICTED"
    assert result["nudity"] == "PARTIAL"
    assert result["sexual_content"] == "SUGGESTIVE"
    assert result["symbols_present"] == {
        "types": ["POLITICAL", "BRAND"],
        "names": ["American flag", "Coca-Cola logo"],
        "misidentification_risk": "MEDIUM"
    }
    assert result["stereotyping_present"] == "POSSIBLY"
    assert result["atrocities_depicted"] == "YES"
    assert result["text_characteristics"] == {
        "text_present": "YES",
        "text_type": "HANDWRITTEN_CURSIVE",
        "legibility": "DIFFICULT"
    }


def test_expand_all_text_types():
    """Each text type abbreviation maps correctly."""
    base = {
        "people": "N", "demog": "N", "misid_risk": "L", "minors": "N",
        "named_indiv": "N", "violence": "0", "racial_viol": "0",
        "nudity": "0", "sexual": "0",
        "symbols": {"types": ["0"], "names": [], "misid_risk": "L"},
        "stereotyping": "N", "atrocities": "N",
    }

    type_map = {
        "PR": "PRINTED", "TY": "TYPED", "HWPR": "HANDWRITTEN_PRINT",
        "HWCU": "HANDWRITTEN_CURSIVE", "MX": "MIXED",
    }
    for short, full in type_map.items():
        form = {**base, "text_chars": {"present": "Y", "type": short, "legib": "CL"}}
        result = expand_safety_form(form)
        assert result["text_characteristics"]["text_type"] == full


def test_expand_all_legibility_values():
    """Each legibility abbreviation maps correctly."""
    base = {
        "people": "N", "demog": "N", "misid_risk": "L", "minors": "N",
        "named_indiv": "N", "violence": "0", "racial_viol": "0",
        "nudity": "0", "sexual": "0",
        "symbols": {"types": ["0"], "names": [], "misid_risk": "L"},
        "stereotyping": "N", "atrocities": "N",
    }

    legib_map = {
        "CL": "CLEAR", "PC": "PARTIALLY_CLEAR", "DIF": "DIFFICULT", "ILL": "ILLEGIBLE",
    }
    for short, full in legib_map.items():
        form = {**base, "text_chars": {"present": "Y", "type": "PR", "legib": short}}
        result = expand_safety_form(form)
        assert result["text_characteristics"]["legibility"] == full


def test_expand_all_symbol_types():
    """Each symbol type abbreviation maps correctly."""
    base = {
        "people": "N", "demog": "N", "misid_risk": "L", "minors": "N",
        "named_indiv": "N", "violence": "0", "racial_viol": "0",
        "nudity": "0", "sexual": "0", "stereotyping": "N",
        "atrocities": "N",
        "text_chars": {"present": "N", "type": "NA", "legib": "NA"},
    }

    sym_map = {
        "CUL": "CULTURAL", "REL": "RELIGIOUS", "POL": "POLITICAL",
        "HATE": "HATE", "BRD": "BRAND",
    }
    for short, full in sym_map.items():
        form = {**base, "symbols": {"types": [short], "names": ["test symbol"], "misid_risk": "L"}}
        result = expand_safety_form(form)
        assert full in result["symbols_present"]["types"]


def test_expand_nudity_full():
    """FULL nudity value stays as FULL."""
    base = {
        "people": "Y", "demog": "N", "misid_risk": "L", "minors": "N",
        "named_indiv": "N", "violence": "0", "racial_viol": "0",
        "nudity": "FULL", "sexual": "EXP",
        "symbols": {"types": ["0"], "names": [], "misid_risk": "L"},
        "stereotyping": "N", "atrocities": "N",
        "text_chars": {"present": "N", "type": "NA", "legib": "NA"},
    }
    result = expand_safety_form(base)
    assert result["nudity"] == "FULL"
    assert result["sexual_content"] == "EXPLICIT"


def test_expand_unknown_key_raises():
    """Unknown short key raises ValueError with the key name in the message."""
    with pytest.raises(ValueError, match="unknown_field"):
        expand_safety_form({"unknown_field": "Y"})


def test_expand_unknown_top_level_value_raises():
    """Unknown value for a top-level field raises ValueError naming the field."""
    with pytest.raises(ValueError, match="people"):
        expand_safety_form({
            "people": "MAYBE",
            "demog": "N", "misid_risk": "L", "minors": "N",
            "named_indiv": "N", "violence": "0", "racial_viol": "0",
            "nudity": "0", "sexual": "0",
            "symbols": {"types": ["0"], "names": [], "misid_risk": "L"},
            "stereotyping": "N", "atrocities": "N",
            "text_chars": {"present": "N", "type": "NA", "legib": "NA"},
        })


def test_expand_unknown_symbol_type_raises():
    """Unknown value in symbols.types list raises ValueError naming the field."""
    with pytest.raises(ValueError, match="symbols.types"):
        expand_safety_form({
            "people": "N", "demog": "N", "misid_risk": "L", "minors": "N",
            "named_indiv": "N", "violence": "0", "racial_viol": "0",
            "nudity": "0", "sexual": "0",
            "symbols": {"types": ["UNKNOWN"], "names": [], "misid_risk": "L"},
            "stereotyping": "N", "atrocities": "N",
            "text_chars": {"present": "N", "type": "NA", "legib": "NA"},
        })


def test_expand_unknown_text_chars_value_raises():
    """Unknown value in text_chars raises ValueError naming the sub-field."""
    with pytest.raises(ValueError, match="text_chars.legib"):
        expand_safety_form({
            "people": "N", "demog": "N", "misid_risk": "L", "minors": "N",
            "named_indiv": "N", "violence": "0", "racial_viol": "0",
            "nudity": "0", "sexual": "0",
            "symbols": {"types": ["0"], "names": [], "misid_risk": "L"},
            "stereotyping": "N", "atrocities": "N",
            "text_chars": {"present": "Y", "type": "PR", "legib": "UNCLEAR"},
        })


def test_expand_preserves_names_list():
    """Symbol names list (free text) is preserved as-is."""
    base = {
        "people": "N", "demog": "N", "misid_risk": "L", "minors": "N",
        "named_indiv": "N", "violence": "0", "racial_viol": "0",
        "nudity": "0", "sexual": "0",
        "symbols": {
            "types": ["REL", "CUL"],
            "names": ["Star of David", "Menorah"],
            "misid_risk": "M"
        },
        "stereotyping": "N", "atrocities": "N",
        "text_chars": {"present": "N", "type": "NA", "legib": "NA"},
    }
    result = expand_safety_form(base)
    assert result["symbols_present"]["names"] == ["Star of David", "Menorah"]
    assert result["symbols_present"]["types"] == ["RELIGIOUS", "CULTURAL"]
