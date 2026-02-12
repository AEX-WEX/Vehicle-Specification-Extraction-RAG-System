"""
Extractor Diagnostics & Performance Tracking

Monitors which extraction method is being used (Ollama vs Rule-based),
checks if Ollama is running, and compares performance metrics.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import json
import requests

logger = logging.getLogger(__name__)


@dataclass
class ExtractionMetrics:
    """Metrics for a single extraction attempt."""
    timestamp: str
    method: str  # "ollama", "rule_based", or "hybrid"
    query: str
    num_specs_extracted: int
    execution_time_ms: float
    success: bool
    error_message: Optional[str] = None
    ollama_available: Optional[bool] = None
    confidence_scores: List[float] = field(default_factory=list)


class ExtractorDiagnostics:
    """Monitor and diagnose extractor performance."""

    def __init__(self, stats_file: Optional[str] = None):
        """
        Initialize diagnostics tracker.

        Args:
            stats_file: Path to file for persisting stats (default: logs/extraction_stats.json)
        """
        self.stats_file = Path(stats_file or "logs/extraction_stats.json")
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)

        self.metrics: List[ExtractionMetrics] = []
        self.ollama_status = self._check_ollama_status()

        # Load existing stats
        self._load_existing_stats()

        logger.info(f"Extractor diagnostics initialized")
        logger.info(f"Ollama available: {self.ollama_status}")

    def _check_ollama_status(self, base_url: str = "http://localhost:11434") -> bool:
        """
        Check if Ollama is running.

        Args:
            base_url: Ollama server URL

        Returns:
            True if Ollama is running and responsive
        """
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            is_available = response.status_code == 200
            logger.info(f"Ollama status check: {'✓ Running' if is_available else '✗ Not running'}")
            return is_available
        except requests.exceptions.RequestException as e:
            logger.debug(f"Ollama unreachable: {type(e).__name__}: {str(e)}")
            return False

    def record_extraction(
        self,
        method: str,
        query: str,
        num_specs: int,
        execution_time_ms: float,
        success: bool,
        error_message: Optional[str] = None,
        confidence_scores: Optional[List[float]] = None,
    ) -> None:
        """
        Record metrics for an extraction attempt.

        Args:
            method: Extraction method used ("ollama", "rule_based", "hybrid")
            query: User query
            num_specs: Number of specs extracted
            execution_time_ms: Execution time in milliseconds
            success: Whether extraction succeeded
            error_message: Error message if extraction failed
            confidence_scores: Confidence scores for extracted specs
        """
        metric = ExtractionMetrics(
            timestamp=datetime.now().isoformat(),
            method=method,
            query=query,
            num_specs_extracted=num_specs,
            execution_time_ms=execution_time_ms,
            success=success,
            error_message=error_message,
            ollama_available=self.ollama_status,
            confidence_scores=confidence_scores or [],
        )

        self.metrics.append(metric)
        self._save_stats()

        log_level = logging.INFO if success else logging.WARNING
        logger.log(
            log_level,
            f"[{method.upper()}] Query: {query[:50]}... | "
            f"Specs: {num_specs} | Time: {execution_time_ms:.0f}ms | "
            f"Success: {success}",
        )

    def get_performance_summary(self) -> Dict:
        """
        Get performance comparison between methods.

        Returns:
            Dictionary with metrics for each extraction method
        """
        if not self.metrics:
            return {"message": "No extraction metrics recorded yet"}

        summary = {}

        for method in set(m.method for m in self.metrics):
            method_metrics = [m for m in self.metrics if m.method == method]

            successful = [m for m in method_metrics if m.success]
            failed = [m for m in method_metrics if not m.success]

            avg_specs = (
                sum(m.num_specs_extracted for m in successful) / len(successful)
                if successful
                else 0
            )
            avg_time = (
                sum(m.execution_time_ms for m in successful) / len(successful)
                if successful
                else 0
            )
            avg_confidence = []
            for m in successful:
                avg_confidence.extend(m.confidence_scores)
            avg_confidence = (
                sum(avg_confidence) / len(avg_confidence)
                if avg_confidence
                else 0
            )

            success_rate = len(successful) / len(method_metrics) if method_metrics else 0

            summary[method] = {
                "total_attempts": len(method_metrics),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": f"{success_rate * 100:.1f}%",
                "avg_specs_extracted": f"{avg_specs:.1f}",
                "avg_execution_time_ms": f"{avg_time:.1f}",
                "avg_confidence_score": f"{avg_confidence:.2f}",
            }

        return summary

    def get_recommendation(self) -> Dict:
        """
        Get recommendation on which extraction method is better.

        Returns:
            Dictionary with recommendation and reasoning
        """
        summary = self.get_performance_summary()

        if not summary or "message" in summary:
            return {
                "recommendation": "Insufficient data",
                "reasoning": "Not enough extraction attempts to make a recommendation",
                "ollama_available": self.ollama_status,
            }

        # Compare methods
        best_method = None
        best_score = -1

        for method, stats in summary.items():
            try:
                success_rate = float(stats["success_rate"].rstrip("%")) / 100
                avg_specs = float(stats["avg_specs_extracted"])
                avg_time = float(stats["avg_execution_time_ms"])

                # Scoring: success rate (50%) + avg specs (30%) - time penalty (20%)
                # Normalize time: treat anything over 1000ms as penalty
                time_penalty = min(avg_time / 1000, 1.0)
                score = (success_rate * 0.5) + (min(avg_specs / 5, 1.0) * 0.3) - (
                    time_penalty * 0.2
                )

                if score > best_score:
                    best_score = score
                    best_method = method

            except (ValueError, KeyError):
                continue

        if not best_method:
            return {
                "recommendation": "Unable to determine",
                "reasoning": "Could not parse performance metrics",
                "ollama_available": self.ollama_status,
            }

        # Generate reasoning
        ollama_stats = summary.get("ollama", {})
        rule_stats = summary.get("rule_based", {})

        reasoning = []
        reasoning.append(f"Best method: {best_method.upper()}")

        if best_method == "ollama" and ollama_stats:
            reasoning.append(
                f"✓ Ollama success rate: {ollama_stats.get('success_rate', 'N/A')}"
            )
            reasoning.append(
                f"✓ Average specs extracted: {ollama_stats.get('avg_specs_extracted', 'N/A')}"
            )
            if rule_stats:
                reasoning.append(
                    f"  vs Rule-based: {rule_stats.get('avg_specs_extracted', 'N/A')} specs"
                )
        elif best_method == "rule_based" and rule_stats:
            reasoning.append(
                f"✓ Rule-based success rate: {rule_stats.get('success_rate', 'N/A')}"
            )
            if ollama_stats:
                reasoning.append(f"  (Ollama available: {self.ollama_status})")
                if not self.ollama_status:
                    reasoning.append("  Ollama is currently offline")

        return {
            "recommendation": best_method,
            "score": f"{best_score:.2f}",
            "reasoning": ", ".join(reasoning),
            "ollama_available": self.ollama_status,
            "metrics_summary": summary,
        }

    def print_report(self) -> None:
        """Print a formatted performance report."""
        print("\n" + "=" * 80)
        print("EXTRACTOR DIAGNOSTICS REPORT")
        print("=" * 80)

        # Ollama status
        status_icon = "✓" if self.ollama_status else "✗"
        print(f"\n[Ollama Status]: {status_icon} {'Running' if self.ollama_status else 'Not running'}")

        # Performance summary
        summary = self.get_performance_summary()
        if summary and "message" not in summary:
            print("\n[Performance Metrics]:")
            print("-" * 80)
            for method, stats in summary.items():
                print(f"\n  {method.upper()}:")
                print(f"    Total attempts:        {stats['total_attempts']}")
                print(f"    Successful:            {stats['successful']}")
                print(f"    Failed:                {stats['failed']}")
                print(f"    Success rate:          {stats['success_rate']}")
                print(f"    Avg specs extracted:   {stats['avg_specs_extracted']}")
                print(f"    Avg execution time:    {stats['avg_execution_time_ms']}ms")
                print(f"    Avg confidence:        {stats['avg_confidence_score']}")

        # Recommendation
        print("\n[Recommendation]:")
        print("-" * 80)
        rec = self.get_recommendation()
        print(f"  Best method:  {rec['recommendation'].upper()}")
        if "reasoning" in rec:
            print(f"  Reasoning:    {rec['reasoning']}")
        if "ollama_available" in rec:
            print(f"  Ollama:       {'Available' if rec['ollama_available'] else 'Unavailable'}")

        print("\n" + "=" * 80 + "\n")

    def _save_stats(self) -> None:
        """Save metrics to file."""
        try:
            stats_data = [asdict(m) for m in self.metrics]
            with open(self.stats_file, "w") as f:
                json.dump(stats_data, f, indent=2)
            logger.debug(f"Stats saved to {self.stats_file}")
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")

    def _load_existing_stats(self) -> None:
        """Load metrics from file."""
        try:
            if self.stats_file.exists():
                with open(self.stats_file, "r") as f:
                    stats_data = json.load(f)
                    for data in stats_data:
                        metric = ExtractionMetrics(**data)
                        self.metrics.append(metric)
                logger.info(f"Loaded {len(self.metrics)} existing metrics")
        except Exception as e:
            logger.warning(f"Failed to load existing stats: {e}")

    def reset_stats(self) -> None:
        """Clear all metrics."""
        self.metrics = []
        self._save_stats()
        logger.info("Metrics cleared")

    def export_stats_csv(self, csv_file: Optional[str] = None) -> str:
        """
        Export metrics as CSV.

        Args:
            csv_file: Path to CSV file

        Returns:
            Path to exported file
        """
        import csv

        csv_file = Path(csv_file or "logs/extraction_stats.csv")
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "timestamp",
                    "method",
                    "query",
                    "num_specs_extracted",
                    "execution_time_ms",
                    "success",
                    "error_message",
                    "ollama_available",
                ])
                writer.writeheader()
                for metric in self.metrics:
                    writer.writerow({
                        "timestamp": metric.timestamp,
                        "method": metric.method,
                        "query": metric.query,
                        "num_specs_extracted": metric.num_specs_extracted,
                        "execution_time_ms": metric.execution_time_ms,
                        "success": metric.success,
                        "error_message": metric.error_message or "",
                        "ollama_available": metric.ollama_available,
                    })
            logger.info(f"Stats exported to {csv_file}")
            return str(csv_file)
        except Exception as e:
            logger.error(f"Failed to export stats: {e}")
            raise


# Global diagnostics instance
_diagnostics: Optional[ExtractorDiagnostics] = None


def get_diagnostics() -> ExtractorDiagnostics:
    """Get or create global diagnostics instance."""
    global _diagnostics
    if _diagnostics is None:
        _diagnostics = ExtractorDiagnostics()
    return _diagnostics
