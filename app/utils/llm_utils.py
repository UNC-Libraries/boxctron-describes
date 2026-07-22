"""Shared utilities for LLM service interactions."""
import json
import logging


_MAX_LOGGED_LLM_CONTENT_CHARS = 2000


def log_token_usage(log: logging.Logger, step: str, usage: object) -> None:
    """Log prompt, completion, and total token counts for an LLM call."""
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        log.info(
            f"Token usage [{step}]: "
            f"prompt={prompt_tokens}, "
            f"completion={completion_tokens}, "
            f"total={total_tokens}"
        )


def format_llm_response_diagnostics(response: object) -> str:
    """Build a compact, response-only diagnostic string for malformed LLM outputs."""
    if response is None:
        return "response=None"

    parts = []

    model = getattr(response, "model", None)
    if model:
        parts.append(f"model={model}")

    choices = getattr(response, "choices", None) or []
    parts.append(f"choices={len(choices)}")

    if choices:
        first_choice = choices[0]
        finish_reason = getattr(first_choice, "finish_reason", None)
        if finish_reason:
            parts.append(f"finish_reason={finish_reason}")

        message = getattr(first_choice, "message", None)
        refusal = getattr(message, "refusal", None) if message else None
        if refusal:
            parts.append(f"refusal={_format_preview(refusal)}")

        content = getattr(message, "content", None) if message else None
        parts.append(f"content_preview={_format_preview(content)}")

    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        usage_parts = []
        if prompt_tokens is not None:
            usage_parts.append(f"prompt={prompt_tokens}")
        if completion_tokens is not None:
            usage_parts.append(f"completion={completion_tokens}")
        if total_tokens is not None:
            usage_parts.append(f"total={total_tokens}")
        if usage_parts:
            parts.append(f"usage=({', '.join(usage_parts)})")

    return ", ".join(parts)


def _format_preview(value: object, max_chars: int = _MAX_LOGGED_LLM_CONTENT_CHARS) -> str:
    """Render an escaped, truncated preview suitable for logs."""
    if value is None:
        return "None"

    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, default=str)
        except TypeError:
            text = str(value)

    if len(text) > max_chars:
        text = f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"

    return json.dumps(text)
