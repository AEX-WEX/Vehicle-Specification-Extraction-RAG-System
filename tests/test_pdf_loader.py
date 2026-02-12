"""Tests for PDF loading functionality."""

import sys
sys.path.insert(0, '.')

from src.pdf_loader import load_pdf


def test_pdf_loading():
    """Test PDF loading from sample file."""
    try:
        pages = load_pdf("data/sample-service-manual_1.pdf", loader_type="pymupdf")

        print(f"Successfully loaded {len(pages)} pages")
        
        # Check first 5 pages
        for i, page in enumerate(pages[:5], 1):
            text = page['text']
            print(f"\n{'='*60}")
            print(f"PAGE {i} - Length: {len(text)} chars")
            print(f"{'='*60}")
            print(text[:500])  # First 500 chars
            print("...")
    except FileNotFoundError:
        print("Note: Sample PDF not found. Check data/ directory.")


if __name__ == "__main__":
    test_pdf_loading()
