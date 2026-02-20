"""
Describe endpoint - processes images and returns descriptive information.
"""
from typing import Annotated
from datetime import datetime, timezone
from urllib.parse import urlparse
import logging
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status, Depends
from fastapi.responses import HTMLResponse

from app.models import (
    DescribeUriRequest,
    DescribeResponse,
    ValidationErrorResponse,
    HTTPErrorResponse
)
from app.config import settings
from app.dependencies import get_describe_workflow
from app.services import DescribeImageWorkflow
from app.utils import stream_upload_to_temp, get_path_from_uri

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["describe"])

@router.get("/describe/upload/form", response_class=HTMLResponse, tags=["describe"])
async def upload_form():
    """Display an HTML form for uploading images."""
    template_path = Path(__file__).parent.parent.parent / "templates" / "upload_form.html"
    return HTMLResponse(content=template_path.read_text())


@router.post(
    "/describe/upload",
    response_model=DescribeResponse,
    status_code=status.HTTP_200_OK,
    summary="Describe an uploaded image",
    description="""
    Process an uploaded image file and return descriptive information.

    Submit the image as multipart/form-data along with optional context.
    """,
    responses={
        400: {"model": HTTPErrorResponse, "description": "Invalid MIME type or missing filename"},
        413: {"model": HTTPErrorResponse, "description": "File size exceeds maximum"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"}
    }
)
async def describe_uploaded_image(
    file: UploadFile = File(..., description="Image file to process"),
    context: Annotated[str | None, Form()] = None,
    workflow: DescribeImageWorkflow = Depends(get_describe_workflow)
) -> DescribeResponse:
    """
    Describe an uploaded image using AI vision models.

    Args:
        file: Uploaded image file
        context: Optional context string to guide description generation

    Returns:
        DescribeResponse with description results or error information

    Raises:
        HTTPException: If validation fails
    """
    # Extract filename and mimetype from uploaded file
    filename = file.filename or "uploaded_image"
    mimetype = file.content_type or "application/octet-stream"

    logger.info(f"Received request to describe uploaded file {filename}")

    # Validate MIME type
    if not mimetype.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"MIME type must be for an image (received: {mimetype})"
        )

    # Stream uploaded file to temporary location while checking size
    temp_path = await stream_upload_to_temp(file, filename, settings.max_upload_size)

    try:
        # Process the image through the workflow
        start_time = datetime.now(timezone.utc)
        result = await workflow.process_image(
            image_path=temp_path,
            filename=filename,
            mimetype=mimetype,
            context=context
        )
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return DescribeResponse(
            success=True,
            filename=filename,
            result=result,
            processing_time_ms=processing_time
        )
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()

@router.post(
    "/describe/uri",
    response_model=DescribeResponse,
    status_code=status.HTTP_200_OK,
    summary="Describe an image from URI",
    description="""
    Process an image from a URI and return descriptive information.

    Provide the URI to an accessible image along with metadata in JSON format.
    """,
    responses={
        400: {"model": HTTPErrorResponse, "description": "Invalid URI format or scheme"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"}
    }
)
async def describe_image_from_uri(
    request: DescribeUriRequest,
    workflow: DescribeImageWorkflow = Depends(get_describe_workflow)
) -> DescribeResponse:
    """
    Describe an image from a URI using AI vision models.

    Args:
        request: Request containing URI and metadata

    Returns:
        DescribeResponse with description results or error information

    Raises:
        HTTPException: If validation fails
    """
    logger.info(f"Received request to describe file URI {request.filename}")

    # Validate URI format
    try:
        parsed_uri = urlparse(request.uri)
        # Check that the URI has a scheme and it's one we support
        if not parsed_uri.scheme:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="URI must include a scheme (e.g., file://, http://, https://)"
            )

        # Validate supported schemes
        allowed_schemes = ["file", "http", "https"]
        if parsed_uri.scheme not in allowed_schemes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"URI scheme must be one of: {', '.join(allowed_schemes)}"
            )

        # Ensure there's a path
        if not parsed_uri.path or parsed_uri.path == "/":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="URI must include a path"
            )

        # For http/https URIs, ensure there's a netloc (domain)
        if parsed_uri.scheme in ["http", "https"] and not parsed_uri.netloc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="HTTP/HTTPS URI must include a domain"
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid URI format"
        )

    # Get file path from URI (downloads if http/https, uses local path if file://)
    temp_path = await get_path_from_uri(
        uri=request.uri,
        max_size=settings.max_upload_size,
        filename=request.filename
    )

    # Determine if we need to clean up (only for downloaded files)
    should_cleanup = parsed_uri.scheme in ["http", "https"]

    try:
        # Process the image through the workflow
        start_time = datetime.now(timezone.utc)
        result = await workflow.process_image(
            image_path=temp_path,
            filename=request.filename,
            mimetype=request.mimetype,
            context=request.context
        )
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return DescribeResponse(
            success=True,
            filename=request.filename,
            result=result,
            processing_time_ms=processing_time
        )
    finally:
        # Clean up temporary file only if we downloaded it
        if should_cleanup and temp_path.exists():
            temp_path.unlink()
