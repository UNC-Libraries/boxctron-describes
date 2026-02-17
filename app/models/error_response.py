"""Error response models for API documentation."""
from typing import Any, List
from pydantic import BaseModel, ConfigDict


class ValidationErrorDetail(BaseModel):
    """Detail of a single validation error."""
    model_config = ConfigDict(extra="forbid")

    type: str
    loc: List[str | int]
    msg: str
    input: Any


class ValidationErrorResponse(BaseModel):
    """Response model for validation errors (422)."""
    model_config = ConfigDict(extra="forbid")

    detail: List[ValidationErrorDetail]


class HTTPErrorResponse(BaseModel):
    """Response model for HTTP errors (400, 413, etc)."""
    model_config = ConfigDict(extra="forbid")

    detail: str
