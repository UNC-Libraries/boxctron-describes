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

from app.config import Settings
from app.services.evaluation_service import EvaluationService


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


def get_image_paths(images_directory: Path, limit: int | None = None) -> list[Path]:
    """Return the non-hidden files directly within an images directory."""
    image_paths = sorted(
        path for path in images_directory.iterdir() if path.is_file() and not path.name.startswith(".")
    )
    return image_paths[:limit]


def positive_integer(value: str) -> int:
    """Parse a command-line argument that must be greater than zero."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def get_comparable_result(response: dict) -> dict:
    """Extract the fields compared against a fixture baseline response."""
    if not isinstance(response, dict):
        raise ValueError("Response does not contain an object.")
    result = response.get("result")
    if not isinstance(result, dict):
        raise ValueError("Response does not contain a result object.")

    fields = ("full_description", "alt_text", "transcript", "safety_assessment")
    missing = [field for field in fields if field not in result]
    if missing:
        raise ValueError(f"Response result is missing fields: {', '.join(missing)}")

    return {field: result[field] for field in fields}


def load_baseline_result(baseline_path: Path) -> dict:
    """Read and validate the comparison fields from a fixture baseline response."""
    if not baseline_path.is_file():
        raise ValueError(f"Baseline file does not exist: {baseline_path}")
    try:
        with baseline_path.open(encoding="utf-8") as baseline_file:
            response = json.load(baseline_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Baseline file contains invalid JSON: {baseline_path}") from error

    if not isinstance(response, dict):
        raise ValueError(f"Baseline file does not contain an object: {baseline_path}")
    return get_comparable_result(response)


def write_json(path: Path, content: dict) -> None:
    """Write JSON output in a stable, readable form."""
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
    parser.add_argument(
        "--limit",
        type=positive_integer,
        help="Process only the first N images in filename order.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_directory = args.fixtures_project / "images"
    baseline_directory = args.fixtures_project / "baseline_json"
    if not images_directory.is_dir():
        print(f"Images directory does not exist: {images_directory}", file=sys.stderr)
        return 1
    if not baseline_directory.is_dir():
        print(f"Baseline directory does not exist: {baseline_directory}", file=sys.stderr)
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

    image_paths = get_image_paths(images_directory, args.limit)
    if not image_paths:
        print(f"No images found in: {images_directory}", file=sys.stderr)
        return 1

    if output_directory.exists():
        print(f"Output directory already exists: {output_directory}", file=sys.stderr)
        return 1
    output_directory.mkdir(parents=True)
    comparisons_directory = output_directory / "comparisons"
    comparisons_directory.mkdir()

    api_key = api_keys.split(",", maxsplit=1)[0].strip()
    failures = 0
    completed_comparisons = []
    evaluation_service = EvaluationService(Settings(_env_file=PROJECT_ROOT / ".env"))
    with httpx.Client(timeout=None, headers={"X-API-Key": api_key}) as client:
        for image_path in image_paths:
            print(f"Describing {image_path.name}")
            output_path = output_directory / f"{image_path.name}.json"
            comparison_path = comparisons_directory / f"{image_path.name}.json"
            # Submit image for generation of description using current boxctron-describes states
            try:
                with image_path.open("rb") as image_file:
                    response = client.post(
                        api_url,
                        files={"file": (image_path.name, image_file)},
                    )
            except httpx.HTTPError as error:
                failures += 1
                error_message = f"Upload failed: {error}"
                write_json(output_path, {"error": error_message})
                write_json(
                    comparison_path,
                    {"filename": image_path.name, "status": "error", "error": error_message},
                )
                print(f"  {error_message}", file=sys.stderr)
                continue

            try:
                content = response.json()
            except json.JSONDecodeError:
                content = {"status_code": response.status_code, "response_text": response.text}

            output_path = output_path
            write_json(output_path, content)

            if response.is_error:
                failures += 1
                error_message = f"API returned HTTP {response.status_code}"
                write_json(
                    comparison_path,
                    {"filename": image_path.name, "status": "error", "error": error_message},
                )
                print(f"  {error_message}", file=sys.stderr)
                continue

            # Load the baseline description and generate a comparison between it and the new output
            try:
                baseline_result = load_baseline_result(baseline_directory / f"{image_path.name}.json")
                generated_result = get_comparable_result(content)
                comparison = evaluation_service.compare_results(baseline_result, generated_result)
            except Exception as error:
                failures += 1
                error_message = f"Comparison model failed: {error}"
                write_json(
                    comparison_path,
                    {"filename": image_path.name, "status": "error", "error": error_message},
                )
                print(f"  {error_message}", file=sys.stderr)
                continue

            comparison_record = {"filename": image_path.name, "status": "success", **comparison}
            write_json(comparison_path, comparison_record)
            completed_comparisons.append(comparison_record)
            print(f"  {comparison['verdict']}: {comparison['summary']}")

    overall_assessment_path = output_directory / "overall_assessment.txt"
    if completed_comparisons:
        try:
            overall_assessment = evaluation_service.generate_overall_assessment(completed_comparisons)
            overall_assessment_path.write_text(overall_assessment + "\n", encoding="utf-8")
            print("\nOverall assessment:\n" + overall_assessment)
        except Exception as error:
            failures += 1
            error_message = f"Overall assessment model failed: {error}"
            overall_assessment_path.write_text(error_message + "\n", encoding="utf-8")
            print(error_message, file=sys.stderr)
    else:
        overall_assessment_path.write_text("No successful comparisons were generated.\n", encoding="utf-8")

    print(f"Wrote results to: {output_directory}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
