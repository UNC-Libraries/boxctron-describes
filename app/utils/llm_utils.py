"""Shared utilities for LLM service interactions."""
import logging


def log_token_usage(log: logging.Logger, step: str, usage: object) -> None:
    """Log prompt, completion, and total token counts for an LLM call."""
    if usage:
        log.info(
            f"Token usage [{step}]: "
            f"prompt={usage.prompt_tokens}, "
            f"completion={usage.completion_tokens}, "
            f"total={usage.total_tokens}"
        )
