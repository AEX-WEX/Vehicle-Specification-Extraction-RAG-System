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
    1. Runs both Ollama (if available) and rule-based extraction
    2. Compares results and returns the BEST one
    3. Tracks which method performed better
    4. Provides detailed diagnostics for UI display
    """

    def __init__(self):
        """Initialize smart extractor."""
        self.diagnostics = get_diagnostics()
        self.ollama_extractor = OllamaExtractor()
        self.rule_extractor = RuleBasedExtractor()
        self.last_method_used = None  # Track which method was best
        
        logger.info(f"SmartExtractor initialized (Ollama available: {self.diagnostics.ollama_status})")

    def extract(self, query: str, contexts: List[Dict]) -> List[ExtractedSpec]:
        """
        Intelligently extract specifications by comparing methods.

        Strategy:
        1. If Ollama available: try Ollama AND rule-based
        2. Compare results: number of specs, confidence scores, relevance
        3. Return the BEST results
        4. If Ollama unavailable: use rule-based only

        Args:
            query: User query
            contexts: Context dictionaries

        Returns:
            List of ExtractedSpec objects (from best method)
        """
        logger.info(f"SmartExtractor: Processing query '{query[:50]}...'")
        logger.info(f"SmartExtractor: Ollama status: {'Available' if self.diagnostics.ollama_status else 'Unavailable'}")

        ollama_specs = []
        rule_specs = []
        
        # Try Ollama if available
        if self.diagnostics.ollama_status:
            logger.info("SmartExtractor: Attempting Ollama extraction...")
            try:
                ollama_specs = self.ollama_extractor.extract(query, contexts, validate=True)
                logger.info(f"SmartExtractor: Ollama returned {len(ollama_specs)} specs")
            except Exception as e:
                logger.warning(f"SmartExtractor: Ollama failed ({type(e).__name__}): {e}")
                ollama_specs = []
        else:
            logger.info("SmartExtractor: Ollama unavailable")

        # Always try rule-based for comparison
        logger.info("SmartExtractor: Attempting rule-based extraction...")
        try:
            rule_specs = self.rule_extractor.extract(contexts, query=query)
            logger.info(f"SmartExtractor: Rule-based returned {len(rule_specs)} specs")
        except Exception as e:
            logger.error(f"SmartExtractor: Rule-based extraction failed: {e}")
            rule_specs = []

        # Compare and pick the best
        best_specs = self._compare_and_select(ollama_specs, rule_specs, query)
        
        logger.info(f"SmartExtractor: ✓ Selected {self.last_method_used} with {len(best_specs)} specs")
        return best_specs

    def _compare_and_select(self, ollama_specs: List[ExtractedSpec], 
                           rule_specs: List[ExtractedSpec],
                           query: str) -> List[ExtractedSpec]:
        """
        Compare extraction results and select the best one.

        Scoring criteria (in order):
        1. More specs count (quantity)
        2. Higher average confidence scores (quality)
        3. Query relevance (semantic match with component names)

        Args:
            ollama_specs: Results from Ollama
            rule_specs: Results from rule-based
            query: Original query

        Returns:
            Best extraction results
        """
        # If only one method returned results
        if not ollama_specs and rule_specs:
            self.last_method_used = "rule_based"
            logger.info("SmartExtractor: Selecting rule-based (Ollama had 0 results)")
            return rule_specs
        
        if not rule_specs and ollama_specs:
            self.last_method_used = "ollama"
            logger.info("SmartExtractor: Selecting Ollama (rule-based had 0 results)")
            return ollama_specs
        
        if not ollama_specs and not rule_specs:
            self.last_method_used = "none"
            logger.warning("SmartExtractor: No results from any method")
            return []

        # Both methods returned results - compare them
        ollama_score = self._score_results(ollama_specs, query)
        rule_score = self._score_results(rule_specs, query)

        logger.info(f"SmartExtractor: Ollama score: {ollama_score:.2f} (ratio: {ollama_score:.0%})")
        logger.info(f"SmartExtractor: Rule-based score: {rule_score:.2f} (ratio: {rule_score:.0%})")

        if rule_score > ollama_score:
            self.last_method_used = "rule_based"
            logger.info(f"SmartExtractor: Rule-based WINS (score diff: +{rule_score - ollama_score:.2f})")
            return rule_specs
        elif ollama_score > rule_score:
            self.last_method_used = "ollama"
            logger.info(f"SmartExtractor: Ollama WINS (score diff: +{ollama_score - rule_score:.2f})")
            return ollama_specs
        else:
            # Same score - prefer rule-based (faster, deterministic)
            self.last_method_used = "rule_based"
            logger.info("SmartExtractor: Scores tied - selecting rule-based (faster)")
            return rule_specs

    def _score_results(self, specs: List[ExtractedSpec], query: str) -> float:
        """
        Score extraction results on quality and quantity.

        Scoring formula (RAG-focused):
        - Quantity score: (num_specs / 6) * 0.6      [0-0.6] (60% weight)
        - Quality score: avg(confidence) * 0.4       [0-0.4] (40% weight)
        - Total: [0.0 - 1.0]
        
        Rationale: 60% quantity emphasizes comprehensive extraction (RAG goal),
        while 40% quality prevents low-confidence specs. This ensures rule-based
        fallback with more specs wins over fewer high-confidence results.

        Args:
            specs: List of extracted specs
            query: Original query (for potential relevance scoring)

        Returns:
            Score between 0.0 and 1.0
        """
        if not specs:
            return 0.0

        # Component 1: Quantity score (normalized) - 60% weight
        # More specs = higher score, but diminishing returns after 6 specs
        quantity_score = min(len(specs) / 6.0, 1.0) * 0.6

        # Component 2: Quality score (average confidence) - 40% weight
        avg_confidence = sum(s.confidence for s in specs) / len(specs)
        quality_score = avg_confidence * 0.4

        total_score = quantity_score + quality_score
        
        logger.debug(f"Result scoring: {len(specs)} specs, "
                    f"avg_conf={avg_confidence:.2f}, "
                    f"qty_score={quantity_score:.2f}, "
                    f"qual_score={quality_score:.2f}, "
                    f"total={total_score:.2f}")
        
        return total_score

    def extract_with_metadata(self, query: str, contexts: List[Dict]) -> Dict:
        """
        Extract specifications and return with metadata about which method was used.

        Useful for UI to show which extraction method was selected.

        Returns:
            {
                "specs": List[ExtractedSpec],
                "method_used": str ("ollama" or "rule_based"),
                "confidence": float (0.0-1.0),
                "message": str (explanation for UI)
            }
        """
        best_specs = self.extract(query, contexts)
        
        if not best_specs:
            avg_confidence = 0.0
            message = "⚠️ No specifications found"
        else:
            avg_confidence = sum(s.confidence for s in best_specs) / len(best_specs)
            message = f"✓ Found {len(best_specs)} specs using {self.last_method_used}"
        
        return {
            "specs": best_specs,
            "method_used": self.last_method_used,
            "num_specs": len(best_specs),
            "average_confidence": avg_confidence,
            "message": message
        }

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
