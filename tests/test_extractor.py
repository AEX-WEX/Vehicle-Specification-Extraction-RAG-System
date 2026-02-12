"""
Tests for specification extraction module.
"""

import pytest
from src.extractor import ExtractedSpec


class TestExtractedSpec:
    """Test ExtractedSpec dataclass."""
    
    def test_spec_creation_basic(self):
        """Test basic specification creation."""
        spec = ExtractedSpec(
            component="Brake Caliper",
            spec_type="Torque",
            value="24",
            unit="Nm"
        )
        
        assert spec.component == "Brake Caliper"
        assert spec.spec_type == "Torque"
        assert spec.value == "24"
        assert spec.unit == "Nm"
        assert spec.confidence == 1.0
    
    def test_spec_with_metadata(self):
        """Test specification with page and source information."""
        spec = ExtractedSpec(
            component="Engine Oil",
            spec_type="Capacity",
            value="5.5",
            unit="L",
            confidence=0.95,
            page_number=42,
            source_chunk_id="chunk_00123"
        )
        
        assert spec.page_number == 42
        assert spec.source_chunk_id == "chunk_00123"
        assert spec.confidence == 0.95
    
    def test_spec_default_confidence(self):
        """Test that confidence defaults to 1.0."""
        spec = ExtractedSpec(
            component="Component",
            spec_type="Type",
            value="Value",
            unit="Unit"
        )
        
        assert spec.confidence == 1.0
    
    def test_spec_optional_fields(self):
        """Test that optional fields can be None."""
        spec = ExtractedSpec(
            component="Component",
            spec_type="Type",
            value="Value",
            unit="Unit",
            page_number=None,
            source_chunk_id=None
        )
        
        assert spec.page_number is None
        assert spec.source_chunk_id is None
