"""
Response models for the API endpoints.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class DescriptionResult(BaseModel):
    """Contains the generated description and metadata."""
    
    description: str = Field(
        ...,
        description="The generated description of the image"
    )
    
    confidence: Optional[float] = Field(
        None,
        description="Confidence score of the description (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    tags: Optional[list[str]] = Field(
        None,
        description="List of tags/keywords extracted from the image"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata about the processing"
    )


class DescribeResponse(BaseModel):
    """
    Response model for the describe endpoint.
    """
    
    success: bool = Field(
        ...,
        description="Indicates whether the request was successful"
    )
    
    filename: str = Field(
        ...,
        description="The filename that was processed"
    )
    
    result: Optional[DescriptionResult] = Field(
        None,
        description="The description result, present if success is True"
    )
    
    error: Optional[str] = Field(
        None,
        description="Error message if success is False"
    )
    
    processing_time_ms: Optional[float] = Field(
        None,
        description="Time taken to process the request in milliseconds"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "filename": "photo.jpg",
                "result": {
                    "description": "A scenic mountain landscape with snow-capped peaks",
                    "confidence": 0.95,
                    "tags": ["mountain", "landscape", "nature", "snow"],
                    "metadata": {
                        "model": "gpt-4o",
                        "provider": "azure"
                    }
                },
                "processing_time_ms": 1250.5
            }
        }
