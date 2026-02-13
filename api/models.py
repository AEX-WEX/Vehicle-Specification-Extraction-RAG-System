"""Pydantic models for API requests and responses."""

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for specification query."""

    query: str = Field(..., description="Natural language query for specifications")
    top_k: Optional[int] = Field(None, description="Number of results to retrieve")
    return_contexts: bool = Field(
        False, description="Whether to return retrieved contexts"
    )


class Specification(BaseModel):
    """Model for extracted specification."""

    component: str
    spec_type: str
    value: str
    unit: str
    page_number: Optional[int] = None
    source_chunk_id: Optional[str] = None
    confidence: float = Field(default=0.8, description="Confidence score (0.0-1.0)")


class QueryResponse(BaseModel):
    """Response model for specification query."""

    query: str
    specifications: List[Specification]
    num_results: int
    extraction_method: Optional[str] = Field(None, description="Method used (ollama or rule_based)")
    average_confidence: float = Field(default=0.0, description="Average confidence of results (0.0-1.0)")
    message: Optional[str] = None


class IndexRequest(BaseModel):
    """Request model for index building."""

    pdf_path: str = Field(..., description="Path to PDF file")
    force_rebuild: bool = Field(False, description="Force rebuild of index")


class IndexResponse(BaseModel):
    """Response model for index building."""

    status: str
    message: str
    num_chunks: Optional[int] = None


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    pipeline_initialized: bool
    index_loaded: bool
    total_chunks: Optional[int] = None


class ContextItem(BaseModel):
    """Retrieved context item."""

    text: str
    chunk_id: str
    page_number: int
    score: float
    distance: float
    metadata: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None
