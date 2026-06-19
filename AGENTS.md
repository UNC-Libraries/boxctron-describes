# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Commands

```bash
# Run all tests (with coverage by default via pytest.ini)
pytest

# Run a single test file
pytest tests/test_describe.py

# Run a single test function
pytest tests/test_describe.py::test_function_name

# Run tests without coverage (faster)
pytest --no-cov

# Start dev server
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Architecture

**boxctron-describes** is a FastAPI microservice that generates descriptive metadata (alt text, full description, transcript) from images using LLM vision models via [LiteLLM](https://github.com/BerriAI/litellm).

### Request flow

```
POST /api/v1/describe/upload  (multipart file)
POST /api/v1/describe/uri     (JSON with URI + metadata)
    ↓
app/api/routes/describe.py    — validates request, handles temp file lifecycle
    ↓
app/services/describe_image_workflow.py  (DescribeImageWorkflow)
    ↓
  1. ImageNormalizer            — resize/normalize → base64 data URL
  2. ImageDescriptionService    — LLM call → FULL_DESCRIPTION, ALT_TEXT, TRANSCRIPT, SAF (safety form), SAR (safety reasoning)
     └─ safety_form_expander    — expand abbreviated LLM keys/values to full forms
  3. SafetyRiskScoringService   — weighted score from SafetyAssessment fields (0–100)
  4. SafetyInconsistencyService — count logical inconsistencies in safety fields
  5. ReviewAssessmentService    — second LLM call reviewing generated content for bias/quality issues
     └─ review_form_expander    — expand abbreviated review keys/values
  6. ReviewRiskScoringService   — weighted score from ReviewAssessment fields (0–100)
    ↓
DescriptionResult (overall_risk_score = avg of safety + review scores)
```

### LLM token efficiency pattern

Both LLM services (`ImageDescriptionService`, `ReviewAssessmentService`) use abbreviated JSON keys and enum values in the structured output schema to reduce token usage (e.g., `"people"` instead of `"people_visible"`, `"Y"/"N"` instead of `"YES"/"NO"`). The `*_form_expander` modules translate these back to full forms before the rest of the app sees them.

### Three-model design

- **Full description model** (`litellm_full_desc_model`, default `azure/gpt-4o`): Vision model that sees the image; generates description + safety assessment in one call.
- **Transcription model** (`litellm_transcribe_model`, default None): Vision model that is used to generate a higher quality transcript from the image. If not configred, this step is skipped. Only triggered if the image contains significant text that is difficult to read according to the full description model.
- **Review model** (`litellm_review_model`, default `azure/gpt-4o`): Text-only model that reviews the generated content for bias and quality issues. This step can be skipped via `review_skip_threshold` if the safety risk score is below the threshold.

### Risk scoring

Both `safety_risk_scoring_service.py` and `review_risk_scoring_service.py` use weighted field tables. Scores are normalized against `_EFFECTIVE_MAX_SCORE = _MAX_POSSIBLE_SCORE * _PRACTICAL_MAX_FRACTION` (not the theoretical maximum), so real-world scores spread across the 0–100 range more usefully. **Update `_MAX_POSSIBLE_SCORE` comments** whenever weight tables change.

### Configuration

All settings live in `app/config.py` (pydantic-settings). Loaded from `.env`. Key groups:
- `LITELLM_FULL_DESC_*` — first LLM call (vision)
- `LITELLM_TRANSCRIBE_*` — second LLM call (vision)
- `LITELLM_REVIEW_*` — third LLM call (review)
- `REVIEW_SKIP_THRESHOLD` — skip review step if safety score is below this
- `AUTH_ENABLED` / `API_KEYS` / `AUTH_USERNAME` / `AUTH_PASSWORD` — hybrid auth (API key header or HTTP Basic)

### Prompts

Prompt templates are `.txt` files in `app/prompts/`. They are loaded once at service initialization time (not per-request).
