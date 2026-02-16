"""
Request models for the API endpoints.
"""
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from fastapi import UploadFile


class DescribeRequest(BaseModel):
    """
    Request model for the describe endpoint.
    
    Either file_uri or an uploaded file must be provided.
    """
    
    file_uri: Optional[str] = Field(
        None,
        description="URI to an image file to process"
    )
    
    context: Optional[str] = Field(
        None,
        description="Optional context string to help guide the description generation",
        max_length=2000
    )
    
    filename: str = Field(
        ...,
        description="Filename of the uploaded file",
        max_length=255
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_uri": "https://example.com/images/photo.jpg",
                "context": "This is a product photo for an e-commerce listing",
                "filename": "photo.jpg",
                "mimetype": "image/jpeg"
            }
        }
