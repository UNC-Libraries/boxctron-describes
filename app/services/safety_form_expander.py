"""Expands abbreviated safety assessment form keys and values to their full forms.

The image description LLM produces a compact SAFETY_ASSESSMENT_FORM with shortened
keys and enum values to reduce output token usage. This module maps them back to
full, human-readable forms used by the rest of the application.
"""
from typing import Any, Dict

# ── Key mappings (short → full) ──────────────────────────────────────────────

SAFETY_FORM_KEY_MAP: Dict[str, str] = {
    "people": "people_visible",
    "demog": "demographics_described",
    "misid_risk": "misidentification_risk_people",
    "minors": "minors_present",
    "named_indiv": "named_individuals_claimed",
    "violence": "violent_content",
    "racial_viol": "racial_violence_oppression",
    "nudity": "nudity",
    "sexual": "sexual_content",
    "symbols": "symbols_present",
    "stereotyping": "stereotyping_present",
    "atrocities": "atrocities_depicted",
    "text_chars": "text_characteristics",
}

SYMBOLS_KEY_MAP: Dict[str, str] = {
    "types": "types",
    "names": "names",
    "misid_risk": "misidentification_risk",
}

TEXT_CHARS_KEY_MAP: Dict[str, str] = {
    "present": "text_present",
    "type": "text_type",
    "legib": "legibility",
}

# ── Value mappings (short → full) ────────────────────────────────────────────

BINARY_VALUE_MAP: Dict[str, str] = {
    "Y": "YES",
    "N": "NO",
}

RISK_VALUE_MAP: Dict[str, str] = {
    "L": "LOW",
    "M": "MEDIUM",
    "H": "HIGH",
}

TERNARY_NPY_VALUE_MAP: Dict[str, str] = {
    "N": "NO",
    "P": "POSSIBLY",
    "Y": "YES",
}

VIOLENCE_VALUE_MAP: Dict[str, str] = {
    "0": "NONE",
    "IMP": "IMPLIED",
    "DEP": "DEPICTED",
}

NUDITY_VALUE_MAP: Dict[str, str] = {
    "0": "NONE",
    "PAR": "PARTIAL",
    "FULL": "FULL",
}

SEXUAL_VALUE_MAP: Dict[str, str] = {
    "0": "NONE",
    "SUG": "SUGGESTIVE",
    "EXP": "EXPLICIT",
}

SYMBOL_TYPE_VALUE_MAP: Dict[str, str] = {
    "0": "NONE",
    "CUL": "CULTURAL",
    "REL": "RELIGIOUS",
    "POL": "POLITICAL",
    "HATE": "HATE",
    "BRD": "BRAND",
}

TEXT_TYPE_VALUE_MAP: Dict[str, str] = {
    "NA": "N/A",
    "PR": "PRINTED",
    "TY": "TYPED",
    "HWPR": "HANDWRITTEN_PRINT",
    "HWCU": "HANDWRITTEN_CURSIVE",
    "MX": "MIXED",
}

LEGIBILITY_VALUE_MAP: Dict[str, str] = {
    "NA": "N/A",
    "CL": "CLEAR",
    "PC": "PARTIALLY_CLEAR",
    "DIF": "DIFFICULT",
    "ILL": "ILLEGIBLE",
}

# Map each short key to its value expansion map
_FIELD_VALUE_MAPS: Dict[str, Dict[str, str]] = {
    "people": BINARY_VALUE_MAP,
    "demog": BINARY_VALUE_MAP,
    "misid_risk": RISK_VALUE_MAP,
    "minors": BINARY_VALUE_MAP,
    "named_indiv": BINARY_VALUE_MAP,
    "violence": VIOLENCE_VALUE_MAP,
    "racial_viol": VIOLENCE_VALUE_MAP,
    "nudity": NUDITY_VALUE_MAP,
    "sexual": SEXUAL_VALUE_MAP,
    "stereotyping": TERNARY_NPY_VALUE_MAP,
    "atrocities": BINARY_VALUE_MAP,
}


def _map_value(value: str, value_map: Dict[str, str], field: str) -> str:
    """Look up value in map, raising ValueError with context if not found."""
    if value not in value_map:
        raise ValueError(f"Unexpected value {value!r} for field {field!r}")
    return value_map[value]


def expand_safety_form(short_form: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand abbreviated safety form keys and values to full forms.

    Args:
        short_form: Safety assessment form dict with abbreviated keys/values
                    as produced by the LLM.

    Returns:
        Dictionary with full-length keys and values matching the SafetyAssessment model.

    Raises:
        ValueError: If a short key or value is not in the mapping.
    """
    expanded: Dict[str, Any] = {}

    for short_key, value in short_form.items():
        if short_key not in SAFETY_FORM_KEY_MAP:
            raise ValueError(f"Unknown short key {short_key!r}")
        full_key = SAFETY_FORM_KEY_MAP[short_key]

        if short_key == "symbols":
            expanded[full_key] = _expand_symbols(value)
        elif short_key == "text_chars":
            expanded[full_key] = _expand_text_chars(value)
        else:
            value_map = _FIELD_VALUE_MAPS.get(short_key)
            if value_map and isinstance(value, str):
                expanded[full_key] = _map_value(value, value_map, short_key)
            else:
                expanded[full_key] = value

    return expanded


def _expand_symbols(symbols: Dict[str, Any]) -> Dict[str, Any]:
    """Expand the symbols sub-object."""
    expanded: Dict[str, Any] = {}
    for short_key, value in symbols.items():
        if short_key not in SYMBOLS_KEY_MAP:
            raise ValueError(f"Unknown symbols key {short_key!r}")
        full_key = SYMBOLS_KEY_MAP[short_key]
        if short_key == "types":
            expanded[full_key] = [_map_value(v, SYMBOL_TYPE_VALUE_MAP, "symbols.types") for v in value]
        elif short_key == "misid_risk":
            expanded[full_key] = _map_value(value, RISK_VALUE_MAP, "symbols.misid_risk")
        else:
            # names — pass through as-is
            expanded[full_key] = value
    return expanded


def _expand_text_chars(text_chars: Dict[str, Any]) -> Dict[str, Any]:
    """Expand the text_chars sub-object."""
    _value_maps: Dict[str, Dict[str, str]] = {
        "present": BINARY_VALUE_MAP,
        "type": TEXT_TYPE_VALUE_MAP,
        "legib": LEGIBILITY_VALUE_MAP,
    }
    expanded: Dict[str, Any] = {}
    for short_key, value in text_chars.items():
        if short_key not in TEXT_CHARS_KEY_MAP:
            raise ValueError(f"Unknown text_chars key {short_key!r}")
        full_key = TEXT_CHARS_KEY_MAP[short_key]
        value_map = _value_maps.get(short_key)
        if value_map and isinstance(value, str):
            expanded[full_key] = _map_value(value, value_map, f"text_chars.{short_key}")
        else:
            expanded[full_key] = value
    return expanded
