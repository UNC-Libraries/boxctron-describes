"""
Request model for the describe/uri endpoint.
"""
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DescribeUriRequest(BaseModel):
    """
    Request model for the describe/uri endpoint.
    
    Contains all information needed to process an image from a URI.
    """
    
    uri: str = Field(
        ...,
        description="URI to an image file to process",
        max_length=2048
    )
    
    context: Optional[str] = Field(
        None,
        description="Optional context string to help guide the description generation",
        max_length=5000
    )
    
    filename: str = Field(
        ...,
        description="Filename of the image",
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
                "uri": "https://example.com/images/photo.jpg",
                "context": "This is a product photo for an e-commerce listing",
                "filename": "photo.jpg",
                "mimetype": "image/jpeg"
            }
        }
    )
