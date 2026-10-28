from pathlib import Path

from evaluate_baseline import build_api_url, get_image_paths


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
