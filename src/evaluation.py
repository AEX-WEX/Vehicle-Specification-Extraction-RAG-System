"""
Evaluation Module

Evaluates extraction accuracy and retrieval quality.
"""

import logging
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import csv

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthSpec:
    """Ground truth specification for evaluation."""
    query: str
    component: str
    spec_type: str
    value: str
    unit: str
    page_number: Optional[int] = None


@dataclass
class EvaluationResult:
    """Evaluation result for a single query."""
    query: str
    num_extracted: int
    num_expected: int
    num_correct: int
    precision: float
    recall: float
    f1_score: float
    extracted_specs: List[Dict]
    expected_specs: List[Dict]
    errors: List[str]


class SpecificationEvaluator:
    """
    Evaluates extraction accuracy against ground truth.
    """
    
    def __init__(self, 
                 strict_matching: bool = False,
                 value_tolerance: float = 0.01):
        """
        Initialize evaluator.
        
        Args:
            strict_matching: Require exact string matches
            value_tolerance: Tolerance for numeric value matching
        """
        self.strict_matching = strict_matching
        self.value_tolerance = value_tolerance
    
    def evaluate_query(self,
                       extracted_specs: List[Dict],
                       ground_truth: List[GroundTruthSpec]) -> EvaluationResult:
        """
        Evaluate extraction for a single query.
        
        Args:
            extracted_specs: Extracted specifications
            ground_truth: Expected specifications
            
        Returns:
            EvaluationResult object
        """
        if not ground_truth:
            logger.warning("No ground truth provided")
            return EvaluationResult(
                query="",
                num_extracted=len(extracted_specs),
                num_expected=0,
                num_correct=0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                extracted_specs=extracted_specs,
                expected_specs=[],
                errors=["No ground truth provided"]
            )
        
        query = ground_truth[0].query
        
        # Match extracted specs with ground truth
        matches = self._match_specs(extracted_specs, ground_truth)
        num_correct = len(matches)
        
        # Calculate metrics
        precision = num_correct / len(extracted_specs) if extracted_specs else 0.0
        recall = num_correct / len(ground_truth) if ground_truth else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Identify errors
        errors = []
        for gt in ground_truth:
            if not any(gt == matched_gt for _, matched_gt in matches):
                errors.append(f"Missing: {gt.component} - {gt.spec_type}")
        
        for spec in extracted_specs:
            if not any(spec == matched_spec for matched_spec, _ in matches):
                errors.append(f"Incorrect: {spec.get('component', 'Unknown')} - {spec.get('spec_type', 'Unknown')}")
        
        return EvaluationResult(
            query=query,
            num_extracted=len(extracted_specs),
            num_expected=len(ground_truth),
            num_correct=num_correct,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            extracted_specs=extracted_specs,
            expected_specs=[asdict(gt) for gt in ground_truth],
            errors=errors
        )
    
    def _match_specs(self,
                     extracted: List[Dict],
                     ground_truth: List[GroundTruthSpec]) -> List[Tuple[Dict, GroundTruthSpec]]:
        """
        Match extracted specs with ground truth.
        
        Args:
            extracted: Extracted specifications
            ground_truth: Ground truth specifications
            
        Returns:
            List of (extracted_spec, ground_truth_spec) pairs
        """
        matches = []
        used_extracted = set()
        used_gt = set()
        
        for i, gt in enumerate(ground_truth):
            for j, spec in enumerate(extracted):
                if j in used_extracted or i in used_gt:
                    continue
                
                if self._is_match(spec, gt):
                    matches.append((spec, gt))
                    used_extracted.add(j)
                    used_gt.add(i)
                    break
        
        return matches
    
    def _is_match(self, spec: Dict, gt: GroundTruthSpec) -> bool:
        """
        Check if extracted spec matches ground truth.
        
        Args:
            spec: Extracted specification
            gt: Ground truth specification
            
        Returns:
            True if match
        """
        # Compare component
        if not self._compare_text(spec.get('component', ''), gt.component):
            return False
        
        # Compare spec type
        if not self._compare_text(spec.get('spec_type', ''), gt.spec_type):
            return False
        
        # Compare value
        if not self._compare_value(spec.get('value', ''), gt.value):
            return False
        
        # Compare unit
        if not self._compare_unit(spec.get('unit', ''), gt.unit):
            return False
        
        return True
    
    def _compare_text(self, text1: str, text2: str) -> bool:
        """Compare text fields."""
        if self.strict_matching:
            return text1.strip() == text2.strip()
        else:
            return text1.strip().lower() == text2.strip().lower()
    
    def _compare_value(self, val1: str, val2: str) -> bool:
        """Compare numeric values with tolerance."""
        try:
            num1 = float(val1)
            num2 = float(val2)
            return abs(num1 - num2) <= self.value_tolerance
        except (ValueError, TypeError):
            return self._compare_text(val1, val2)
    
    def _compare_unit(self, unit1: str, unit2: str) -> bool:
        """Compare units (with common aliases)."""
        unit_aliases = {
            'nm': ['nm', 'n-m', 'newton-meter'],
            'ft-lb': ['ft-lb', 'ft-lbs', 'lb-ft', 'lbf-ft'],
            'liter': ['liter', 'liters', 'l', 'lt'],
            'bar': ['bar', 'bars'],
            'psi': ['psi', 'lb/in2'],
        }
        
        unit1_normalized = unit1.strip().lower()
        unit2_normalized = unit2.strip().lower()
        
        # Check direct match
        if unit1_normalized == unit2_normalized:
            return True
        
        # Check aliases
        for standard, aliases in unit_aliases.items():
            if unit1_normalized in aliases and unit2_normalized in aliases:
                return True
        
        return False
    
    def evaluate_batch(self,
                       test_cases: List[Tuple[str, List[Dict], List[GroundTruthSpec]]]) -> Dict:
        """
        Evaluate multiple queries.
        
        Args:
            test_cases: List of (query, extracted_specs, ground_truth) tuples
            
        Returns:
            Aggregate evaluation metrics
        """
        results = []
        
        for query, extracted, gt in test_cases:
            result = self.evaluate_query(extracted, gt)
            results.append(result)
        
        # Aggregate metrics
        avg_precision = sum(r.precision for r in results) / len(results) if results else 0.0
        avg_recall = sum(r.recall for r in results) / len(results) if results else 0.0
        avg_f1 = sum(r.f1_score for r in results) / len(results) if results else 0.0
        
        total_extracted = sum(r.num_extracted for r in results)
        total_expected = sum(r.num_expected for r in results)
        total_correct = sum(r.num_correct for r in results)
        
        return {
            "num_queries": len(results),
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1_score": avg_f1,
            "total_extracted": total_extracted,
            "total_expected": total_expected,
            "total_correct": total_correct,
            "micro_precision": total_correct / total_extracted if total_extracted > 0 else 0.0,
            "micro_recall": total_correct / total_expected if total_expected > 0 else 0.0,
            "results": results
        }
    
    def save_results(self, results: Dict, output_path: str):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
        
        # Also save CSV summary
        csv_path = output_path.with_suffix('.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Query', 'Extracted', 'Expected', 'Correct', 'Precision', 'Recall', 'F1'])
            
            for result in results.get('results', []):
                if isinstance(result, EvaluationResult):
                    writer.writerow([
                        result.query,
                        result.num_extracted,
                        result.num_expected,
                        result.num_correct,
                        f"{result.precision:.3f}",
                        f"{result.recall:.3f}",
                        f"{result.f1_score:.3f}"
                    ])
        
        logger.info(f"CSV summary saved to {csv_path}")


class RetrievalEvaluator:
    """
    Evaluates retrieval quality.
    """
    
    def __init__(self):
        """Initialize retrieval evaluator."""
        pass
    
    def evaluate_retrieval(self,
                          retrieved_contexts: List[Dict],
                          relevant_page_numbers: List[int]) -> Dict:
        """
        Evaluate retrieval quality.
        
        Args:
            retrieved_contexts: Retrieved context chunks
            relevant_page_numbers: Page numbers containing relevant info
            
        Returns:
            Retrieval metrics
        """
        if not relevant_page_numbers:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "map": 0.0,
                "mrr": 0.0
            }
        
        retrieved_pages = [ctx['page_number'] for ctx in retrieved_contexts]
        
        # Calculate metrics
        relevant_retrieved = sum(1 for page in retrieved_pages if page in relevant_page_numbers)
        
        precision = relevant_retrieved / len(retrieved_pages) if retrieved_pages else 0.0
        recall = relevant_retrieved / len(relevant_page_numbers) if relevant_page_numbers else 0.0
        
        # Mean Average Precision (simplified)
        precisions_at_k = []
        relevant_count = 0
        for k, page in enumerate(retrieved_pages, 1):
            if page in relevant_page_numbers:
                relevant_count += 1
                precisions_at_k.append(relevant_count / k)
        
        map_score = sum(precisions_at_k) / len(relevant_page_numbers) if relevant_page_numbers else 0.0
        
        # Mean Reciprocal Rank
        mrr = 0.0
        for k, page in enumerate(retrieved_pages, 1):
            if page in relevant_page_numbers:
                mrr = 1.0 / k
                break
        
        return {
            "precision": precision,
            "recall": recall,
            "map": map_score,
            "mrr": mrr,
            "num_retrieved": len(retrieved_pages),
            "num_relevant": len(relevant_page_numbers),
            "num_relevant_retrieved": relevant_retrieved
        }


def load_ground_truth(file_path: str) -> List[GroundTruthSpec]:
    """
    Load ground truth from JSON file.
    
    Args:
        file_path: Path to ground truth JSON
        
    Returns:
        List of GroundTruthSpec objects
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    ground_truth = []
    for item in data:
        gt = GroundTruthSpec(**item)
        ground_truth.append(gt)
    
    return ground_truth


if __name__ == "__main__":
    # Test evaluation
    evaluator = SpecificationEvaluator()
    
    extracted = [
        {"component": "Brake Caliper Bolt", "spec_type": "Torque", "value": "35", "unit": "Nm"}
    ]
    
    ground_truth = [
        GroundTruthSpec(
            query="Brake caliper torque",
            component="Brake Caliper Bolt",
            spec_type="Torque",
            value="35",
            unit="Nm"
        )
    ]
    
    result = evaluator.evaluate_query(extracted, ground_truth)
    print(f"Precision: {result.precision:.3f}")
    print(f"Recall: {result.recall:.3f}")
    print(f"F1: {result.f1_score:.3f}")
