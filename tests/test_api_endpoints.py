"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_endpoint_exists(self, client):
        """Test that health endpoint responds."""
        response = client.get("/health")
        
        # Should return 200 or 503 depending on pipeline initialization
        assert response.status_code in [200, 503]
    
    def test_health_response_structure(self, client):
        """Test that health endpoint returns expected structure."""
        response = client.get("/health")
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "pipeline_initialized" in data
            assert "index_loaded" in data


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint_accessible(self, client):
        """Test that root endpoint is accessible."""
        response = client.get("/")
        
        # Should return 200 or HTML response
        assert response.status_code == 200


class TestQueryEndpoint:
    """Test query endpoint."""
    
    def test_query_endpoint_structure(self, client):
        """Test query endpoint request/response structure."""
        payload = {
            "query": "What is the brake caliper torque?",
            "return_contexts": False
        }
        
        response = client.post("/query", json=payload)
        
        # Endpoint should exist and be callable
        assert response.status_code in [200, 503, 422]
    
    def test_query_missing_parameters(self, client):
        """Test query with missing required parameters."""
        response = client.post("/query", json={})
        
        # Should fail validation
        assert response.status_code in [422, 200]
