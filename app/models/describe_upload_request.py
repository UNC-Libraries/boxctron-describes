"""
Request model for the describe/upload endpoint.
"""
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DescribeUploadRequest(BaseModel):
    """
    Request model for the describe/upload endpoint.
    
    Contains metadata for file upload. The actual file is handled separately
    as a multipart/form-data file parameter.
    """
    
    context: Optional[str] = Field(
        None,
        description="Optional context string to help guide the description generation",
        max_length=5000
    )
    
    filename: str = Field(
        ...,
        description="Filename of the uploaded file",
        max_length=1024
    )
    
    mimetype: str = Field(
        ...,
        description="MIME type of the file being processed",
        pattern=r"^image/.+$"
    )
    
    @field_validator("mimetype")
    @classmethod
    def validate_mimetype(cls, v: str) -> str:
        """Validate that mimetype starts with 'image/'."""
        if not v.startswith("image/"):
            raise ValueError("MIME type must be for an image")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "context": "This is a product photo for an e-commerce listing",
                "filename": "photo.jpg",
                "mimetype": "image/jpeg"
            }
        }
    )
