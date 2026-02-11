
import sys
sys.path.insert(0, '.')
from src.pipeline import create_pipeline
import logging
logging.basicConfig(level=logging.ERROR)

pipeline = create_pipeline()
pipeline.load_index()

tests = [
    'What is the torque for brake caliper bolts?',
    'Engine oil capacity',
    'Coolant capacity',
    'Tire pressure'
]

for i, q in enumerate(tests, 1):
    r = pipeline.query(q)
    print(f'Test {i}: {q}')
    print(f'  Found: {r[\"num_results\"]} specs')
    for s in r['specifications'][:2]:
        print(f'    - {s[\"component\"]}: {s[\"value\"]} {s[\"unit\"]}')
    print()
