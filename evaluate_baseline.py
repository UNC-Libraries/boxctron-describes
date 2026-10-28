"""Generate description results for every image in a fixtures project.

Usage:
    python evaluate_baseline.py [fixtures_project] [--suffix SUFFIX]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FIXTURES_PROJECT = PROJECT_ROOT.parent / "boxctron-describes-fixtures"


def build_api_url(host: str, port: str) -> str:
    """Build the upload endpoint URL from the configured host and port."""
    if "://" not in host:
        host = f"http://{host}"

    parsed_host = urlparse(host)
    if not parsed_host.hostname:
        raise ValueError("HOST must include a hostname.")

    return f"{parsed_host.scheme}://{parsed_host.hostname}:{port}/api/v1/describe/upload"


def get_image_paths(images_directory: Path) -> list[Path]:
    """Return the non-hidden files directly within an images directory."""
    return sorted(
        path for path in images_directory.iterdir() if path.is_file() and not path.name.startswith(".")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit fixture images to a running boxctron-describes instance."
    )
    parser.add_argument(
        "fixtures_project",
        nargs="?",
        type=Path,
        default=DEFAULT_FIXTURES_PROJECT,
        help=f"Fixtures project path (default: {DEFAULT_FIXTURES_PROJECT})",
    )
    parser.add_argument(
        "--suffix",
        help="Optional label appended to the timestamped output directory name.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_directory = args.fixtures_project / "images"
    if not images_directory.is_dir():
        print(f"Images directory does not exist: {images_directory}", file=sys.stderr)
        return 1

    if args.suffix and Path(args.suffix).name != args.suffix:
        print("Suffix must not contain a path separator.", file=sys.stderr)
        return 1

    environment = dotenv_values(PROJECT_ROOT / ".env")
    host = environment.get("HOST")
    port = environment.get("PORT")
    api_keys = environment.get("API_KEYS")
    if not host or not port or not api_keys:
        print(".env must define HOST, PORT, and API_KEYS.", file=sys.stderr)
        return 1

    try:
        api_url = build_api_url(host, port)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    if args.suffix:
        run_id = f"{run_id}_{args.suffix}"
    output_directory = PROJECT_ROOT / "output" / run_id

    image_paths = get_image_paths(images_directory)
    if not image_paths:
        print(f"No images found in: {images_directory}", file=sys.stderr)
        return 1

    if output_directory.exists():
        print(f"Output directory already exists: {output_directory}", file=sys.stderr)
        return 1
    output_directory.mkdir(parents=True)

    api_key = api_keys.split(",", maxsplit=1)[0].strip()
    failures = 0
    with httpx.Client(timeout=None, headers={"X-API-Key": api_key}) as client:
        for image_path in image_paths:
            print(f"Describing {image_path.name}")
            with image_path.open("rb") as image_file:
                response = client.post(
                    api_url,
                    files={"file": (image_path.name, image_file)},
                )

            try:
                content = response.json()
            except json.JSONDecodeError:
                content = {"status_code": response.status_code, "response_text": response.text}

            output_path = output_directory / f"{image_path.name}.json"
            output_path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            if response.is_error:
                failures += 1
                print(f"  API returned HTTP {response.status_code}", file=sys.stderr)

    print(f"Wrote results to: {output_directory}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
