"""Tests for LLM integration (Ollama, OpenAI, etc)."""

import requests
import json


def test_ollama_api():
    """Test Ollama API connectivity and basic extraction."""
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

    try:
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
    except requests.exceptions.ConnectionError:
        print("Note: Ollama is not running. Start it with: ollama serve")
    except Exception as e:
        print(f"Error testing Ollama: {e}")


if __name__ == "__main__":
    test_ollama_api()
