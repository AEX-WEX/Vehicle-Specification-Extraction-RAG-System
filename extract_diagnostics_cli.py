"""
Extractor Diagnostics CLI

Check Ollama status and view extraction performance metrics.

Usage:
    python extract_diagnostics_cli.py status          # Check Ollama status
    python extract_diagnostics_cli.py report          # View performance report
    python extract_diagnostics_cli.py reset           # Clear all metrics
    python extract_diagnostics_cli.py export          # Export to CSV
    python extract_diagnostics_cli.py recommend       # Get recommendation
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.extractor_diagnostics import get_diagnostics, ExtractorDiagnostics


def cmd_status(args):
    """Check Ollama status."""
    diagnostics = get_diagnostics()
    
    print("\n" + "=" * 80)
    print("OLLAMA STATUS")
    print("=" * 80)
    
    status_icon = "✓" if diagnostics.ollama_status else "✗"
    status_text = "RUNNING" if diagnostics.ollama_status else "NOT RUNNING"
    
    print(f"\n  Status: {status_icon} {status_text}")
    print(f"  URL:    http://localhost:11434")
    
    if diagnostics.ollama_status:
        print("\n  ✓ Ollama is ready for extraction!")
    else:
        print("\n  ✗ Ollama is not running.")
        print("    Start it with: ollama serve")
        print("    Or run extractions with SmartExtractor for rule-based fallback")
    
    print("\n" + "=" * 80 + "\n")


def cmd_report(args):
    """View performance metrics."""
    diagnostics = get_diagnostics()
    diagnostics.print_report()


def cmd_reset(args):
    """Reset all metrics."""
    diagnostics = get_diagnostics()
    diagnostics.reset_stats()
    print("\n✓ Metrics cleared\n")


def cmd_export(args):
    """Export metrics to CSV."""
    diagnostics = get_diagnostics()
    csv_file = diagnostics.export_stats_csv()
    print(f"\n✓ Exported to {csv_file}\n")


def cmd_recommend(args):
    """Get recommendation on best extraction method."""
    diagnostics = get_diagnostics()
    
    print("\n" + "=" * 80)
    print("EXTRACTION METHOD RECOMMENDATION")
    print("=" * 80)
    
    rec = diagnostics.get_recommendation()
    
    if "message" in rec:
        print(f"\n  {rec['message']}")
    else:
        print(f"\n  Recommended method: {rec['recommendation'].upper()}")
        print(f"  Score:              {rec.get('score', 'N/A')}")
        print(f"  Ollama available:   {'Yes' if rec['ollama_available'] else 'No'}")
        
        if "reasoning" in rec:
            print(f"\n  Reasoning:")
            for line in rec['reasoning'].split(","):
                print(f"    • {line.strip()}")
        
        if "metrics_summary" in rec:
            print(f"\n  Detailed Metrics:")
            for method, metrics in rec['metrics_summary'].items():
                print(f"    {method.upper()}:")
                print(f"      Success rate: {metrics['success_rate']}")
                print(f"      Avg specs:    {metrics['avg_specs_extracted']}")
                print(f"      Avg time:     {metrics['avg_execution_time_ms']}ms")
    
    print("\n" + "=" * 80 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extractor Diagnostics & Performance Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_diagnostics_cli.py status          # Check Ollama
  python extract_diagnostics_cli.py report          # View metrics
  python extract_diagnostics_cli.py recommend       # Best method?
  python extract_diagnostics_cli.py export          # Save to CSV
  python extract_diagnostics_cli.py reset           # Clear stats
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Status command
    status_parser = subparsers.add_parser('status', help='Check Ollama status')
    status_parser.set_defaults(func=cmd_status)

    # Report command
    report_parser = subparsers.add_parser('report', help='View performance report')
    report_parser.set_defaults(func=cmd_report)

    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Clear all metrics')
    reset_parser.set_defaults(func=cmd_reset)

    # Export command
    export_parser = subparsers.add_parser('export', help='Export metrics to CSV')
    export_parser.set_defaults(func=cmd_export)

    # Recommend command
    rec_parser = subparsers.add_parser('recommend', help='Get recommendation')
    rec_parser.set_defaults(func=cmd_recommend)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
