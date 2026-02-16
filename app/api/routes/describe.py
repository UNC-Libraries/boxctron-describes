"""
Describe endpoint - processes images and returns descriptive information.
"""
from typing import Annotated
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status, Depends

from app.models import DescribeUploadRequest, DescribeUriRequest, DescribeResponse, DescriptionResult
from app.config import settings

router = APIRouter(prefix="/api/v1", tags=["describe"])


@router.post(
    "/describe/upload",
    response_model=DescribeResponse,
    status_code=status.HTTP_200_OK,
    summary="Describe an uploaded image",
    description="""
    Process an uploaded image file and return descriptive information.
    
    Submit the image as multipart/form-data along with metadata fields.
    """
)
async def describe_uploaded_image(
    file: UploadFile = File(..., description="Image file to process"),
    context: Annotated[str | None, Form()] = None,
    filename: Annotated[str, Form()] = ...,
    mimetype: Annotated[str, Form()] = ...
) -> DescribeResponse:
    """
    Describe an uploaded image using AI vision models.
    
    Args:
        file: Uploaded image file
        context: Optional context string to guide description generation
        filename: Name of the file being processed
        mimetype: MIME type of the image
        
    Returns:
        DescribeResponse with description results or error information
        
    Raises:
        HTTPException: If validation fails
    """
    
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
    
    # Validate uploaded file
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
                "note": "Actual implementation pending",
                "source": "upload"
            }
        ),
        processing_time_ms=0.0
    )


@router.post(
    "/describe/uri",
    response_model=DescribeResponse,
    status_code=status.HTTP_200_OK,
    summary="Describe an image from URI",
    description="""
    Process an image from a URI and return descriptive information.
    
    Provide the URI to an accessible image along with metadata in JSON format.
    """
)
async def describe_image_from_uri(
    request: DescribeUriRequest
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
    
    # Validate MIME type against allowed types
    if request.mimetype not in settings.allowed_mime_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"MIME type '{request.mimetype}' is not supported. Allowed types: {settings.allowed_mime_types}"
        )
    
    # TODO: Implement actual image processing logic
    # - Download image from URI
    # - Validate image content
    # - Process with LLM
    # For now, return a placeholder response
    return DescribeResponse(
        success=True,
        filename=request.filename,
        result=DescriptionResult(
            description="[Placeholder] Image description will be generated here",
            confidence=None,
            tags=None,
            metadata={
                "status": "not_implemented",
                "note": "Actual implementation pending",
                "source": "uri",
                "uri": request.uri
            }
        ),
        processing_time_ms=0.0
    )
