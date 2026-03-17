"""Expands abbreviated review assessment keys and values to their full forms.

The review LLM produces a compact review assessment with shortened keys and
enum values to reduce output token usage. This module maps them back to full,
human-readable forms used by the rest of the application.
"""
from typing import Any, Dict, List

# ── Key mappings (short → full) ──────────────────────────────────────────────

REVIEW_KEY_MAP: Dict[str, str] = {
    "bias": "biased_language",
    "stereo": "stereotyping",
    "val_judg": "value_judgments",
    "contra_btwn": "contradictions_between_texts",
    "contra_within": "contradictions_within_description",
    "offensive": "offensive_language",
    "incon_demog": "inconsistent_demographics",
    "euphemism": "euphemistic_language",
    "ppl_first": "people_first_language",
    "unsup_infer": "unsupported_inferential_claims",
    "safety_consist": "safety_assessment_consistency",
    "concerns": "concerns_for_review",
    "src_warn": "source_content_warnings",
}

# ── Value mappings (short → full) ────────────────────────────────────────────

TERNARY_NPY_VALUE_MAP: Dict[str, str] = {
    "N": "NO",
    "P": "POSSIBLY",
    "Y": "YES",
}

BINARY_NY_VALUE_MAP: Dict[str, str] = {
    "N": "NO",
    "Y": "YES",
}

PEOPLE_FIRST_VALUE_MAP: Dict[str, str] = {
    "U": "USED",
    "NU": "NOT_USED",
    "NA": "N/A",
}

CONSISTENCY_VALUE_MAP: Dict[str, str] = {
    "CON": "CONSISTENT",
    "INCON": "INCONSISTENT",
}

# Map each short key to its value expansion map
_FIELD_VALUE_MAPS: Dict[str, Dict[str, str]] = {
    "bias": TERNARY_NPY_VALUE_MAP,
    "stereo": TERNARY_NPY_VALUE_MAP,
    "val_judg": TERNARY_NPY_VALUE_MAP,
    "contra_btwn": BINARY_NY_VALUE_MAP,
    "contra_within": TERNARY_NPY_VALUE_MAP,
    "offensive": BINARY_NY_VALUE_MAP,
    "incon_demog": BINARY_NY_VALUE_MAP,
    "euphemism": TERNARY_NPY_VALUE_MAP,
    "ppl_first": PEOPLE_FIRST_VALUE_MAP,
    "unsup_infer": TERNARY_NPY_VALUE_MAP,
    "safety_consist": CONSISTENCY_VALUE_MAP,
}


def _map_value(value: str, value_map: Dict[str, str], field: str) -> str:
    """Look up value in map, raising ValueError with context if not found."""
    if value not in value_map:
        raise ValueError(f"Unexpected value {value!r} for field {field!r}")
    return value_map[value]


def expand_review_form(short_form: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand abbreviated review assessment keys and values to full forms.

    Args:
        short_form: Review assessment dict with abbreviated keys/values
                    as produced by the LLM.

    Returns:
        Dictionary with full-length keys and values matching the ReviewAssessment model.

    Raises:
        ValueError: If a short key or value is not in the mapping.
    """
    expanded: Dict[str, Any] = {}

    for short_key, value in short_form.items():
        if short_key not in REVIEW_KEY_MAP:
            raise ValueError(f"Unknown short key {short_key!r}")
        full_key = REVIEW_KEY_MAP[short_key]

        value_map = _FIELD_VALUE_MAPS.get(short_key)
        if value_map and isinstance(value, str):
            expanded[full_key] = _map_value(value, value_map, short_key)
        else:
            # concerns list — pass through as-is
            expanded[full_key] = value

    return expanded
