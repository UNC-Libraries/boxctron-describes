"""
Describe endpoint - processes images and returns descriptive information.
"""
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse

from app.models import DescribeResponse, DescriptionResult
from app.config import settings

router = APIRouter(prefix="/api/v1", tags=["describe"])


@router.post(
    "/describe",
    response_model=DescribeResponse,
    status_code=status.HTTP_200_OK,
    summary="Describe an image",
    description="""
    Process an image and return descriptive information.
    
    The endpoint accepts either:
    - An uploaded file via multipart/form-data
    - A file URI pointing to an image
    
    At least one of these must be provided.
    """
)
async def describe_image(
    file: Optional[UploadFile] = File(None, description="Image file to process"),
    file_uri: Optional[str] = Form(None, description="URI to an image file"),
    context: Optional[str] = Form(None, description="Optional context for description"),
    filename: str = Form(..., description="Filename of the image"),
    mimetype: str = Form(..., description="MIME type of the image")
) -> DescribeResponse:
    """
    Describe an image using AI vision models.
    
    Args:
        file: Uploaded image file (optional if file_uri is provided)
        file_uri: URI to an image file (optional if file is provided)
        context: Optional context string to guide description generation
        filename: Name of the file being processed
        mimetype: MIME type of the image
        
    Returns:
        DescribeResponse with description results or error information
        
    Raises:
        HTTPException: If validation fails or neither file nor file_uri is provided
    """
    
    # Validate that at least one input method is provided
    if not file and not file_uri:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'file' or 'file_uri' must be provided"
        )
    
    # Validate MIME type
    if not mimetype.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MIME type must be for an image"
        )
    
    # Validate MIME type against allowed types
    if mimetype not in settings.allowed_mime_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"MIME type '{mimetype}' is not supported. Allowed types: {settings.allowed_mime_types}"
        )
    
    # Validate uploaded file if provided
    if file:
        # Check file size
        file_content = await file.read()
        if len(file_content) > settings.max_upload_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum allowed size of {settings.max_upload_size} bytes"
            )
        # Reset file pointer for future processing
        await file.seek(0)
    
    # TODO: Implement actual image processing logic
    # For now, return a placeholder response
    return DescribeResponse(
        success=True,
        filename=filename,
        result=DescriptionResult(
            description="[Placeholder] Image description will be generated here",
            confidence=None,
            tags=None,
            metadata={
                "status": "not_implemented",
                "note": "Actual implementation pending"
            }
        ),
        processing_time_ms=0.0
    )
