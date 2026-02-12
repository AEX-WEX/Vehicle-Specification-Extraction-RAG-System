"""
LLM Extraction Module

Handles structured extraction using LLMs.
"""

import json
import logging
import re
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from src.extractor_diagnostics import get_diagnostics

load_dotenv()

@dataclass
class ExtractedSpec:
    """Represents an extracted specification."""
    component: str
    spec_type: str
    value: str
    unit: str
    confidence: float = 1.0
    source_chunk_id: Optional[str] = None
    page_number: Optional[int] = None


class LLMExtractor:
    """
    Extracts structured specifications using LLM.
    """
    
    def __init__(self,
                 provider: str = "openai",
                 model: str = "gpt-3.5-turbo",
                 temperature: float = 0.0,
                 max_tokens: int = 500,
                 api_key: Optional[str] = None):
        """
        Initialize LLM extractor.
        
        Args:
            provider: LLM provider ("openai", "anthropic", "cohere")
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            api_key: API key (if not in environment)
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize client
        if provider == "openai":
            import openai
            if api_key:
                openai.api_key = api_key
            self.client = openai.OpenAI()
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"Initialized LLM extractor: {provider}/{model}")
    
    def extract(self,
                query: str,
                contexts: List[Dict],
                validate: bool = True) -> List[ExtractedSpec]:
        """
        Extract specifications from contexts.
        
        Args:
            query: User query
            contexts: Retrieved context dictionaries
            validate: Whether to validate extracted JSON
            
        Returns:
            List of ExtractedSpec objects
        """
        logger.info(f"Extracting specifications for query: {query}")
        
        # Build prompt
        prompt = self._build_extraction_prompt(query, contexts)
        
        # Call LLM
        try:
            response_text = self._call_llm(prompt)
            logger.debug(f"LLM response: {response_text}")
            
            # Parse response
            specs = self._parse_response(response_text, contexts, validate)
            
            logger.info(f"Extracted {len(specs)} specifications")
            return specs
            
        except Exception as e:
            logger.error(f"Extraction error: {str(e)}")
            return []
    
    def _build_extraction_prompt(self, query: str, contexts: List[Dict]) -> str:
        """
        Build extraction prompt.

        Args:
            query: User query
            contexts: Context chunks

        Returns:
            Formatted prompt
        """
        # Combine context
        context_text = "\n\n---\n\n".join([
            f"[Chunk {ctx['chunk_id']} - Page {ctx['page_number']}]\n{ctx['text']}"
            for ctx in contexts
        ])

        prompt = f"""You are a precise automotive specification extractor. Extract vehicle specifications from the provided service manual excerpts.

**User Query:** {query}

**Service Manual Excerpts:**
{context_text}

**Instructions:**
1. Extract ONLY specifications that directly answer the user's query
2. For each specification found, extract: component name, specification type, numeric value, and unit
3. Return ONLY valid specifications with all four fields populated
4. Do NOT hallucinate or infer information not present in the text
5. If you cannot find relevant specifications, return an empty list
6. Be precise with units (Nm, ft-lb, liters, psi, etc.)
7. ALWAYS return a valid JSON array, even if empty

**JSON Output Format:**
[
  {{"component": "Brake Caliper Bolt", "spec_type": "Torque", "value": "35", "unit": "Nm"}},
  {{"component": "Brake Fluid Capacity", "spec_type": "Capacity", "value": "1.2", "unit": "liters"}}
]

**IMPORTANT:**
- Return ONLY the JSON array, no other text
- Ensure "value" and "unit" fields are NEVER empty
- Each specification MUST have all 4 required fields with non-empty values

**Response:**"""

        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API."""
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise data extraction assistant. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        
        elif self.provider == "groq":
            # Groq uses OpenAI-compatible API
            import os
            from openai import OpenAI
            groq_client = OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1"
            )
            response = groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise data extraction assistant. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _parse_response(self,
                        response_text: str,
                        contexts: List[Dict],
                        validate: bool) -> List[ExtractedSpec]:
        """
        Parse LLM response into structured specs.

        Args:
            response_text: Raw LLM response
            contexts: Original contexts
            validate: Whether to validate

        Returns:
            List of ExtractedSpec objects
        """
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```json\s*(\[.*?\])\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON array directly - use greedy matching for closing bracket
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.warning("No JSON found in response")
                logger.debug(f"Response text: {response_text[:500]}")
                return []

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.debug(f"Failed to parse JSON: {json_str[:200]}")
            return []

        # Convert to ExtractedSpec objects
        specs = []
        for item in data:
            if validate and not self._validate_spec(item):
                logger.debug(f"Spec failed validation: {item}")
                continue

            spec = ExtractedSpec(
                component=item.get("component", "Unknown"),
                spec_type=item.get("spec_type", "Unknown"),
                value=str(item.get("value", "")),
                unit=item.get("unit", ""),
                source_chunk_id=contexts[0]['chunk_id'] if contexts else None,
                page_number=contexts[0]['page_number'] if contexts else None
            )
            specs.append(spec)

        return specs

    def _validate_spec(self, spec_dict: Dict) -> bool:
        """
        Validate specification dictionary.

        Args:
            spec_dict: Specification dictionary

        Returns:
            True if valid (has required fields and non-empty values)
        """
        required_fields = ["component", "spec_type", "value", "unit"]

        # Check all fields present
        for field in required_fields:
            if field not in spec_dict:
                logger.warning(f"Missing required field '{field}' in spec: {spec_dict}")
                return False

        # Component and spec_type must be non-empty
        if not spec_dict.get("component") or not spec_dict.get("spec_type"):
            logger.warning(f"Empty component or spec_type in spec: {spec_dict}")
            return False

        # Value and unit CAN be empty (models sometimes return incomplete specs)
        # But at least one should have content for the spec to be useful
        value = str(spec_dict.get("value", "")).strip()
        unit = spec_dict.get("unit", "").strip()

        if not value and not unit:
            logger.warning(f"Both value and unit are empty in spec: {spec_dict}")
            return False

        # Valid spec (even if incomplete)
        return True

class OllamaExtractor:
    """
    Extracts structured specifications using local Ollama models.
    """
    
    def __init__(self,
                 model: str = "llama3",
                 temperature: float = 0.0,
                 max_tokens: int = 500,
                 base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama extractor.
        
        Args:
            model: Ollama model name (llama3, mistral, etc.)
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            base_url: Ollama server URL
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        
        logger.info(f"Initialized Ollama extractor: {model}")
    
    def extract(self,
                query: str,
                contexts: List[Dict],
                validate: bool = True) -> List[ExtractedSpec]:
        """
        Extract specifications from contexts using Ollama.

        Args:
            query: User query
            contexts: Retrieved context dictionaries
            validate: Whether to validate extracted JSON

        Returns:
            List of ExtractedSpec objects
        """
        logger.info(f"Extracting specifications for query: {query}")
        start_time = time.time()
        diagnostics = get_diagnostics()

        # Build prompt
        prompt = self._build_extraction_prompt(query, contexts)

        # Call Ollama
        try:
            response_text = self._call_ollama(prompt)
            logger.debug(f"Ollama response: {response_text}")

            # Parse response
            specs = self._parse_response(response_text, contexts, validate)

            # If Ollama found components but no values, use hybrid extraction
            if specs and any(not spec.value and not spec.unit for spec in specs):
                logger.info("Ollama found components but incomplete values. Using hybrid extraction...")
                specs = self._hybrid_extraction(specs, contexts, query)

            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            confidence_scores = [s.confidence for s in specs]
            diagnostics.record_extraction(
                method="ollama",
                query=query,
                num_specs=len(specs),
                execution_time_ms=execution_time,
                success=True,
                confidence_scores=confidence_scores,
            )
            
            logger.info(f"Extracted {len(specs)} specifications using Ollama")
            return specs

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            diagnostics.record_extraction(
                method="ollama",
                query=query,
                num_specs=0,
                execution_time_ms=execution_time,
                success=False,
                error_message=str(e),
            )
            logger.error(f"Ollama extraction error: {str(e)}")
            return []

    def _hybrid_extraction(self, partial_specs: List[ExtractedSpec],
                         contexts: List[Dict], query: str) -> List[ExtractedSpec]:
        """
        Hybrid extraction: Use Ollama's component/spec_type and fill values from context.

        Args:
            partial_specs: Specs with component/spec_type but empty value/unit
            contexts: Original context chunks
            query: User query

        Returns:
            List of specs with values filled in
        """
        enhanced_specs = []

        for partial_spec in partial_specs:
            # Try to find matching value in contexts using patterns
            value, unit = self._extract_value_from_context(
                partial_spec.component,
                partial_spec.spec_type,
                contexts,
                query
            )

            if value or unit:
                enhanced_spec = ExtractedSpec(
                    component=partial_spec.component,
                    spec_type=partial_spec.spec_type,
                    value=value,
                    unit=unit,
                    source_chunk_id=partial_spec.source_chunk_id,
                    page_number=partial_spec.page_number,
                    confidence=0.8
                )
                enhanced_specs.append(enhanced_spec)
            else:
                # Keep original if no values found in context
                enhanced_specs.append(partial_spec)

        return enhanced_specs

    def _extract_value_from_context(self, component: str, spec_type: str,
                                   contexts: List[Dict], query: str):
        """
        Extract numeric value and unit from context for a given component/spec_type.

        Args:
            component: Component name from Ollama
            spec_type: Specification type from Ollama
            contexts: Context chunks
            query: User query

        Returns:
            Tuple of (value, unit)
        """
        # Common unit patterns by spec type
        unit_patterns = {
            'Torque': r'(Nm|N\s*m|ft-lb|ft\s*lb|lb-ft|lb\s*ft)',
            'Capacity': r'(liters?|gallons?|quarts?|qts?|l|gal|ml|cc)',
            'Pressure': r'(bar|psi|ps?i|kPa|atm|atmospheres?)',
            'Temperature': r'(°C|°F|C|F|Celsius|Fahrenheit)',
            'Speed': r'(rpm|k?m/h|km/h|mph|per minute)',
            'Power': r'(hp|kW|w|watts?|horsepower)',
            'Voltage': r'(V|volts?)',
            'Current': r'(A|amps?|amperes?)',
        }

        # Get unit pattern for this spec type
        unit_pattern = unit_patterns.get(spec_type, r'([A-Za-z/%]+)')

        # Combine all context text
        combined_text = '\n'.join([ctx['text'] for ctx in contexts])

        # Search for component mention followed by spec info
        component_keywords = [kw.strip() for kw in component.lower().split() if len(kw.strip()) > 2]

        for ctx in contexts:
            text = ctx['text']

            # Look for context around component
            for keyword in component_keywords:
                if keyword in text.lower():
                    # Find lines containing component
                    lines = text.split('\n')
                    for i, line in enumerate(lines):
                        if keyword in line.lower():
                            # Search this line and nearby lines for values
                            search_start = max(0, i-2)
                            search_end = min(len(lines), i+4)
                            search_text = '\n'.join(lines[search_start:search_end])

                            # Try different value patterns based on spec type
                            # Pattern 1: "label: value unit" (colon separator)
                            pattern1 = r':\s*(\d+\.?\d*)\s+(' + unit_pattern + r')\b'
                            matches = re.findall(pattern1, search_text, re.IGNORECASE)
                            if matches and matches[0][0]:
                                value, unit = matches[0]
                                logger.debug(f"Hybrid: Found via pattern1: {value} {unit}")
                                return str(value).strip(), str(unit).strip()

                            # Pattern 2: "value unit" (space separator)
                            pattern2 = r'(\d+\.?\d*)\s+(' + unit_pattern + r')\b'
                            matches = re.findall(pattern2, search_text, re.IGNORECASE)
                            if matches and matches[0][0]:
                                value, unit = matches[0]
                                logger.debug(f"Hybrid: Found via pattern2: {value} {unit}")
                                return str(value).strip(), str(unit).strip()

                            # Pattern 3: Just a number (last resort)
                            pattern3 = r'(\d+\.?\d*)'
                            matches = re.findall(pattern3, search_text)
                            if matches:
                                # Take the first reasonable number found
                                for val in matches:
                                    val_float = float(val)
                                    # Filter out suspiciously large/small numbers
                                    if 0.1 < val_float < 100000:
                                        logger.debug(f"Hybrid: Found via pattern3: {val}")
                                        return str(val).strip(), ''

        logger.debug(f"Hybrid: No value found for {component} ({spec_type})")
        return '', ''

    def _clean_value_unit(self, value: str, unit: str) -> tuple:
        """
        Clean and separate value from unit.

        Handles cases where value contains embedded units like "1.66L liters" or "35 Nm".

        Args:
            value: Raw value string (may contain unit)
            unit: Raw unit string

        Returns:
            Tuple of (cleaned_value, cleaned_unit)

        Examples:
            "1.66L liters" → ("1.66", "liters")
            "35 Nm" → ("35", "Nm")
            " Nm" → ("", "Nm")
        """
        if not value or not value.strip():
            return "", unit.strip() if unit else ""

        value = value.strip()

        # Pattern: extract number and optional embedded unit
        # Matches: "1.66L", "35 Nm", "2.84L liters"
        match = re.match(r'^\s*([\d.]+)\s*([A-Za-z/°]+)?\s*(.*)$', value)

        if match and match.group(1):
            number = match.group(1)  # "1.66" or "35"
            embedded_unit = match.group(2) or ""  # "L" or "Nm"
            remaining = match.group(3) or ""  # "liters" or ""

            # Determine final unit (prefer explicit unit parameter)
            final_unit = ""
            if unit and unit.strip():
                final_unit = unit.strip()  # Use explicit unit parameter
            elif remaining.strip():
                final_unit = remaining.strip()  # Use remaining text
            elif embedded_unit:
                final_unit = embedded_unit  # Use embedded unit
            else:
                final_unit = ""

            logger.debug(f"Cleaned: value='{number}', unit='{final_unit}' (from '{value}')")
            return number, final_unit

        # If no number found, return as-is
        logger.debug(f"Could not parse number from value: '{value}'")
        return value, unit.strip() if unit else ""


    def _build_extraction_prompt(self, query: str, contexts: List[Dict]) -> str:
        """Build extraction prompt optimized for Ollama."""
        # Combine context
        context_text = "\n\n---\n\n".join([
            f"[Chunk {ctx['chunk_id']} - Page {ctx['page_number']}]\n{ctx['text']}"
            for ctx in contexts
        ])

        prompt = f"""You are extracting vehicle specifications from a service manual.

USER QUESTION: {query}

MANUAL TEXT:
{context_text}

TASK:
Extract specifications with these EXACT fields:
- component: name of the part (e.g., "Brake Caliper Bolt")
- spec_type: type (Torque, Capacity, Pressure, Temperature, Speed, Power, etc.)
- value: ONLY the number (e.g., "35" NOT "35 Nm")
- unit: ONLY the unit (e.g., "Nm" NOT "35 Nm" or "35")

CRITICAL RULES:
1. Separate the number from the unit ALWAYS
2. If value is "1.66L liters", output value="1.66" and unit="liters"
3. If value contains text like "35 Nm", output value="35" and unit="Nm"
4. Output ONLY clean numeric values in the "value" field
5. If you cannot find the exact numeric value, DO NOT include that specification
6. Return ONLY valid JSON, no explanations or text

CORRECT EXAMPLES:
Input text: "Brake caliper bolt: 35 Nm"
       -> {{"component": "Brake Caliper Bolt", "spec_type": "Torque", "value": "35", "unit": "Nm"}}

Input text: "Engine oil capacity: 4.5 liters"
       -> {{"component": "Engine Oil", "spec_type": "Capacity", "value": "4.5", "unit": "liters"}}

Input text: "Tire pressure: 2.4 bar"
       -> {{"component": "Tire", "spec_type": "Pressure", "value": "2.4", "unit": "bar"}}

INCORRECT EXAMPLES (DO NOT DO THIS):
- WRONG: {{"value": "35 Nm"}} - value should NOT contain unit
- WRONG: {{"unit": "35"}} - unit should NOT contain number
- WRONG: {{"value": "1.66L liters"}} - value should only be number

JSON OUTPUT (array only, no text before or after):"""

        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Ollama response text
        """
        import requests
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '')
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise
    
    def _parse_response(self,
                        response_text: str,
                        contexts: List[Dict],
                        validate: bool) -> List[ExtractedSpec]:
        """
        Parse Ollama response into structured specs with error recovery.

        Args:
            response_text: Raw Ollama response
            contexts: Original contexts
            validate: Whether to validate

        Returns:
            List of ExtractedSpec objects
        """
        # Strategy 1: Look for ```json block
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            logger.debug("Found JSON in code block")
        else:
            # Strategy 2: Look for JSON array pattern
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.debug("Found JSON array pattern")
            else:
                # Strategy 3: Try first line that looks like JSON
                json_str = None
                for line in response_text.split('\n'):
                    line = line.strip()
                    if line.startswith('[') and '{' in line:
                        json_str = line
                        logger.debug("Found JSON on line")
                        break

                if not json_str:
                    logger.warning("No JSON array found in response")
                    logger.debug(f"Response preview: {response_text[:300]}")
                    return []

        # Try parsing with error recovery
        try:
            data = json.loads(json_str)
            logger.debug("JSON parsed successfully")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            logger.debug(f"Failed JSON: {json_str[:200]}")

            # Try to fix common issues
            try:
                # Remove trailing commas
                fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                # Remove extra data after array closing bracket
                fixed = re.sub(r'(\])\s*\S.*$', r'\1', fixed, flags=re.DOTALL)
                data = json.loads(fixed)
                logger.info("Fixed JSON with cleanup regex")
            except Exception as fix_error:
                logger.error(f"Could not repair JSON: {fix_error}")
                logger.debug(f"Attempted fix: {fixed[:200] if 'fixed' in locals() else 'N/A'}")
                return []

        # Convert to ExtractedSpec objects
        specs = []
        for item in data:
            # For Ollama, be lenient with validation - we may do hybrid extraction later
            # Just check that we have component and spec_type
            if not item.get("component") or not item.get("spec_type"):
                logger.debug(f"Skipping spec without component or spec_type: {item}")
                continue

            # Get raw values
            value_raw = str(item.get("value", "")).strip()
            unit_raw = str(item.get("unit", "")).strip()

            # Clean the value/unit pair
            value_clean, unit_clean = self._clean_value_unit(value_raw, unit_raw)

            # Skip specs with empty values (after cleaning)
            if not value_clean or not value_clean.strip():
                logger.debug(f"Skipping spec with empty value after cleaning: {item}")
                continue

            spec = ExtractedSpec(
                component=item.get("component", "Unknown"),
                spec_type=item.get("spec_type", "Unknown"),
                value=value_clean,
                unit=unit_clean,
                source_chunk_id=contexts[0]['chunk_id'] if contexts else None,
                page_number=contexts[0]['page_number'] if contexts else None
            )
            specs.append(spec)
            logger.debug(f"Parsed spec: {spec.component} = {spec.value} {spec.unit}")

        return specs

    def _validate_spec(self, spec_dict: Dict) -> bool:
        """
        Validate specification dictionary.

        Args:
            spec_dict: Specification dictionary

        Returns:
            True if valid (has required fields and non-empty values)
        """
        required_fields = ["component", "spec_type", "value", "unit"]

        # Check all fields present
        for field in required_fields:
            if field not in spec_dict:
                logger.warning(f"Missing required field '{field}' in spec: {spec_dict}")
                return False

        # Component and spec_type must be non-empty
        if not spec_dict.get("component") or not spec_dict.get("spec_type"):
            logger.warning(f"Empty component or spec_type in spec: {spec_dict}")
            return False

        # Value and unit CAN be empty (Ollama sometimes returns incomplete specs)
        # But at least one should have content for the spec to be useful
        value = str(spec_dict.get("value", "")).strip()
        unit = spec_dict.get("unit", "").strip()

        if not value and not unit:
            logger.warning(f"Both value and unit are empty in spec: {spec_dict}")
            return False

        # Valid spec (even if incomplete)
        return True



class RuleBasedExtractor:
    """
    Rule-based specification extractor using regex patterns.
    Fallback when LLM extraction fails.
    """

    def __init__(self):
        """Initialize with comprehensive regex patterns."""
        # Enhanced patterns for different specification types
        self.patterns = [
            # TORQUE PATTERNS
            # "Brake caliper bolt: 35 Nm"
            (r'([^:]+\bbolt[^:]*?):\s*(\d+\.?\d*)\s*(Nm|N·m|ft-lb|lb-ft|kgf·m)', 'Torque'),
            # "Torque specification: 35 Nm" or just numbers with units
            (r'([^:]+):\s*(\d+\.?\d*)\s*(Nm|N·m|ft-lb|lb-ft)', 'Torque'),
            # Standalone number + unit patterns
            (r'(\w+(?:\s+\w+){0,2})\s+(\d+\.?\d*)\s+(Nm|ft-lb|lb-ft)', 'Torque'),

            # CAPACITY PATTERNS
            # "Oil capacity: 4.5 liters"
            (r'([^:]+\bcapacity[^:]*?):\s*(\d+\.?\d*)\s*(L|literss?|litres?|qt|quarts?|gal|gallons?|ml|cc)', 'Capacity'),
            # "Fill: 2.84L liters"
            (r'([^:]+\bfill[^:]*?):\s*(\d+\.?\d*)\s*[L]?\s*(literss?|litres?|quarts?|gallons?)?', 'Capacity'),
            # Oil/fluid/coolant specifications
            (r'(oil|fluid|coolant|brake)(\s+\w+)*\s*(?:capacity|fill)?\s*:?\s*(\d+\.?\d*)\s*(L|literss?|qt|gal)', 'Capacity'),

            # PRESSURE PATTERNS
            # "Tire pressure: 2.4 bar"
            (r'([^:]+\bpressure[^:]*?):\s*(\d+\.?\d*)\s*(bar|psi|kPa|MPa|atm)', 'Pressure'),
            (r'(tire|tyre)s?(\s+\w+)?\s*:?\s*(\d+\.?\d*)\s*(bar|psi)', 'Pressure'),

            # TEMPERATURE PATTERNS
            # "Operating temperature: 90°C"
            (r'([^:]+\btemperature[^:]*?):\s*(?:up\s+to\s+)?(\d+\.?\d*)\s*(°C|°F|Celsius|Fahrenheit)', 'Temperature'),

            # SPEED/RPM PATTERNS
            # "Maximum speed: 2000 rpm"
            (r'([^:]+\b(?:speed|rpm)[^:]*?):\s*(\d+\.?\d*)\s*(rpm|k?m/h|mph)', 'Speed'),
        ]

    def extract(self, contexts: List[Dict], query: str = "") -> List[ExtractedSpec]:
        """
        Extract specifications using regex patterns.

        Args:
            contexts: List of context dictionaries with 'text', 'chunk_id', 'page_number'
            query: User query (for logging/diagnostics)

        Returns:
            List of ExtractedSpec objects
        """
        start_time = time.time()
        diagnostics = get_diagnostics()
        specs = []
        seen = set()  # Track seen specs to avoid duplicates

        try:
            for ctx in contexts:
                text = ctx['text']

                for pattern, spec_type in self.patterns:
                    try:
                        matches = re.finditer(pattern, text, re.IGNORECASE)

                        for match in matches:
                            groups = match.groups()

                            # Parse based on group count
                            if spec_type == 'Capacity' and len(groups) == 3:
                                component = groups[0].strip()
                                value = groups[2].strip() if groups[2] else ""
                                unit = groups[2] if groups[2] else ""
                            elif len(groups) >= 3:
                                component = groups[0].strip()
                                value = groups[1].strip() if groups[1] else ""
                                unit = groups[2].strip() if groups[2] else ""
                            else:
                                continue

                            # Validate
                            if not component or not value or not unit:
                                continue

                            # Create unique key to avoid duplicates
                            key = (component.lower(), value, unit.lower())
                            if key in seen:
                                continue
                            seen.add(key)

                            try:
                                # Validate value is numeric
                                float(value)
                            except ValueError:
                                logger.debug(f"Rule-based: Skipping non-numeric value: {value}")
                                continue

                            spec = ExtractedSpec(
                                component=component,
                                spec_type=spec_type,
                                value=value,
                                unit=unit,
                                source_chunk_id=ctx['chunk_id'],
                                page_number=ctx['page_number'],
                                confidence=0.7  # Lower confidence for rule-based extraction
                            )
                            specs.append(spec)
                            logger.debug(f"Rule-based: Found {component} ({spec_type}): {value} {unit}")

                    except Exception as e:
                        logger.debug(f"Rule-based: Pattern error for {spec_type}: {e}")
                        continue
            
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            confidence_scores = [s.confidence for s in specs]
            diagnostics.record_extraction(
                method="rule_based",
                query=query,
                num_specs=len(specs),
                execution_time_ms=execution_time,
                success=True,
                confidence_scores=confidence_scores,
            )

            logger.info(f"Rule-based extraction found {len(specs)} specifications")
            return specs
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            diagnostics.record_extraction(
                method="rule_based",
                query=query,
                num_specs=len(specs),
                execution_time_ms=execution_time,
                success=False,
                error_message=str(e),
            )
            logger.error(f"Rule-based extraction error: {str(e)}")
            return specs


class SmartExtractor:
    """
    Smart extractor that:
    1. Checks if Ollama is available
    2. Tries Ollama first if available
    3. Falls back to rule-based extraction if Ollama fails or is unavailable
    4. Tracks performance and provides diagnostics
    """

    def __init__(self):
        """Initialize smart extractor."""
        self.diagnostics = get_diagnostics()
        self.ollama_extractor = OllamaExtractor()
        self.rule_extractor = RuleBasedExtractor()
        
        logger.info(f"SmartExtractor initialized (Ollama available: {self.diagnostics.ollama_status})")

    def extract(self, query: str, contexts: List[Dict]) -> List[ExtractedSpec]:
        """
        Intelligently extract specifications.

        Priority order:
        1. Try Ollama if available
        2. Fall back to rule-based if Ollama fails/unavailable
        3. Return best results

        Args:
            query: User query
            contexts: Context dictionaries

        Returns:
            List of ExtractedSpec objects
        """
        logger.info(f"SmartExtractor: Processing query '{query[:50]}...'")
        logger.info(f"SmartExtractor: Ollama status: {'Available' if self.diagnostics.ollama_status else 'Unavailable'}")

        # Try Ollama first if available
        if self.diagnostics.ollama_status:
            logger.info("SmartExtractor: Attempting Ollama extraction...")
            try:
                specs = self.ollama_extractor.extract(query, contexts, validate=True)
                
                if specs:
                    logger.info(f"SmartExtractor: ✓ Ollama succeeded with {len(specs)} specs")
                    return specs
                else:
                    logger.warning("SmartExtractor: Ollama returned 0 specs, trying rule-based...")
            except Exception as e:
                logger.warning(f"SmartExtractor: Ollama failed ({type(e).__name__}), trying rule-based...")
        else:
            logger.info("SmartExtractor: Ollama unavailable, using rule-based extraction")

        # Fall back to rule-based
        logger.info("SmartExtractor: Using rule-based extraction...")
        try:
            specs = self.rule_extractor.extract(contexts, query=query)
            logger.info(f"SmartExtractor: ✓ Rule-based succeeded with {len(specs)} specs")
            return specs
        except Exception as e:
            logger.error(f"SmartExtractor: Rule-based extraction also failed: {e}")
            return []

    def get_diagnostics_report(self) -> Dict:
        """Get performance diagnostics report."""
        return self.diagnostics.get_recommendation()

    def print_diagnostics(self) -> None:
        """Print diagnostics report."""
        self.diagnostics.print_report()


if __name__ == "__main__":
    # Test extraction
    sample_contexts = [
        {
            "text": "Brake caliper mounting bolts should be torqued to 35 Nm.",
            "chunk_id": "chunk_00001",
            "page_number": 42,
            "score": 0.95
        }
    ]
    
    print("Extractor module - requires API keys for testing")
