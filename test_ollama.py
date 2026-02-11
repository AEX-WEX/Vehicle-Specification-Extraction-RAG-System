"""Test Ollama extraction directly."""

import requests
import json

# Test Ollama API directly
def test_ollama():
    prompt = """Extract vehicle specifications from this text:

Text: "Brake caliper mounting bolts: 35 Nm"

Output JSON only:
[
  {
    "component": "Brake Caliper Bolt",
    "spec_type": "Torque", 
    "value": "35",
    "unit": "Nm"
  }
]"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 500
            }
        },
        timeout=60
    )
    
    result = response.json()
    print("=" * 80)
    print("OLLAMA RESPONSE:")
    print("=" * 80)
    print(result.get('response', ''))
    print("=" * 80)

if __name__ == "__main__":
    test_ollama()
