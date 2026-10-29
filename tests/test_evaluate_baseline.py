from pathlib import Path

import json
import argparse

import pytest

from evaluate_baseline import (
    build_api_url,
    get_comparable_result,
    get_image_paths,
    load_baseline_result,
    positive_integer,
)


def test_build_api_url_uses_configured_host_and_port():
    assert build_api_url("localhost", "8000") == "http://localhost:8000/api/v1/describe/upload"
    assert build_api_url("https://describes.example", "443") == (
        "https://describes.example:443/api/v1/describe/upload"
    )


def test_get_image_paths_skips_hidden_files_and_directories(tmp_path: Path):
    visible_image = tmp_path / "visible.jpg"
    visible_image.touch()
    (tmp_path / ".hidden.png").touch()
    (tmp_path / "nested").mkdir()

    assert get_image_paths(tmp_path) == [visible_image]


def test_get_image_paths_applies_limit_in_filename_order(tmp_path: Path):
    first_image = tmp_path / "a.jpg"
    first_image.touch()
    (tmp_path / "b.jpg").touch()

    assert get_image_paths(tmp_path, limit=1) == [first_image]


def test_positive_integer_rejects_zero_and_negative_values():
    assert positive_integer("1") == 1
    with pytest.raises(argparse.ArgumentTypeError, match="greater than zero"):
        positive_integer("0")


def test_load_baseline_result_extracts_comparison_fields(tmp_path: Path):
    baseline_path = tmp_path / "image.jpg.json"
    baseline_path.write_text(
        json.dumps(
            {
                "result": {
                    "full_description": "Description",
                    "alt_text": "Alt text",
                    "transcript": "Transcript",
                    "safety_assessment": {"people_visible": "NO"},
                    "review_assessment": {"biased_language": "NO"},
                }
            }
        ),
        encoding="utf-8",
    )

    assert load_baseline_result(baseline_path) == {
        "full_description": "Description",
        "alt_text": "Alt text",
        "transcript": "Transcript",
        "safety_assessment": {"people_visible": "NO"},
    }


def test_get_comparable_result_rejects_missing_fields():
    with pytest.raises(ValueError, match="missing fields: transcript"):
        get_comparable_result(
            {
                "result": {
                    "full_description": "Description",
                    "alt_text": "Alt text",
                    "safety_assessment": {},
                }
            }
        )
