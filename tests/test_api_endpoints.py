"""
Tests for API endpoints and request models.
"""

import pytest
from api.app import QueryRequest, Specification, QueryResponse, HealthResponse


class TestAPIModels:
    """Test API request/response models."""
    
    def test_query_request_creation(self):
        """Test QueryRequest model creation."""
        request = QueryRequest(
            query="What is the brake caliper torque?",
            return_contexts=False
        )
        
        assert request.query == "What is the brake caliper torque?"
        assert request.return_contexts is False
    
    def test_specification_model(self):
        """Test Specification model creation."""
        spec = Specification(
            component="Brake Caliper",
            spec_type="Torque",
            value="24",
            unit="Nm",
            page_number=145,
            source_chunk_id="chunk_2847"
        )
        
        assert spec.component == "Brake Caliper"
        assert spec.spec_type == "Torque"
        assert spec.value == "24"
        assert spec.unit == "Nm"
    
    def test_query_response_creation(self):
        """Test QueryResponse model creation."""
        specs = [
            Specification(
                component="Brake Caliper",
                spec_type="Torque",
                value="24",
                unit="Nm"
            )
        ]
        
        response = QueryResponse(
            query="brake torque",
            specifications=specs,
            num_results=1
        )
        
        assert response.query == "brake torque"
        assert response.num_results == 1
        assert len(response.specifications) == 1
    
    def test_health_response_creation(self):
        """Test HealthResponse model creation."""
        response = HealthResponse(
            status="healthy",
            pipeline_initialized=True,
            index_loaded=True,
            total_chunks=1234
        )
        
        assert response.status == "healthy"
        assert response.pipeline_initialized is True
        assert response.index_loaded is True
        assert response.total_chunks == 1234
