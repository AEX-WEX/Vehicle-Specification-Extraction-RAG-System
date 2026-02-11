from src.pdf_loader import load_pdf

# Load PDF
pages = load_pdf("data/sample-service-manual_1.pdf", loader_type="pymupdf")

# Check first 5 pages
for i, page in enumerate(pages[:5], 1):
    text = page['text']
    print(f"\n{'='*60}")
    print(f"PAGE {i} - Length: {len(text)} chars")
    print(f"{'='*60}")
    print(text[:500])  # First 500 chars
    print("...")
