"""Integration tests for the RAG pipeline."""

import sys
sys.path.insert(0, '.')

import logging
from src.pipeline import create_pipeline

logging.basicConfig(level=logging.ERROR)


def test_query_examples():
    """Test basic query examples."""
    pipeline = create_pipeline()
    try:
        pipeline.load_index()
    except Exception as e:
        print(f"Note: Could not load existing index: {e}")
        return

    tests = [
        'What is the torque for brake caliper bolts?',
        'Engine oil capacity',
        'Coolant capacity',
        'Tire pressure'
    ]

    for i, query in enumerate(tests, 1):
        result = pipeline.query(query)
        print(f'Test {i}: {query}')
        print(f'  Found: {result["num_results"]} specs')
        for spec in result['specifications'][:2]:
            print(f'    - {spec["component"]}: {spec["value"]} {spec["unit"]}')
        print()


if __name__ == "__main__":
    test_query_examples()
