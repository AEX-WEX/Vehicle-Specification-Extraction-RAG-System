"""
FastAPI application for vehicle specification extraction.
"""

__version__ = "1.0.0"
__author__ = "Vehicle Spec RAG Team"
__description__ = "REST API for vehicle specification extraction from service manuals"
# Expose API models
from api.models import (
    QueryRequest,
    QueryResponse,
    Specification,
    IndexRequest,
    IndexResponse,
    HealthResponse,
    ContextItem,
    ErrorResponse,
)

__all__ = [
    "QueryRequest",
    "QueryResponse",
    "Specification",
    "IndexRequest",
    "IndexResponse",
    "HealthResponse",
    "ContextItem",
    "ErrorResponse",
]