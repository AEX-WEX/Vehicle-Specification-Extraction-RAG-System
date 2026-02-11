"""
Command-Line Interface

CLI for vehicle specification extraction.
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

from src.pipeline import VehicleSpecRAGPipeline

# Load environment variables
load_dotenv()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def cmd_index(args):
    """Build index from PDF."""
    print(f"Building index from: {args.pdf_path}")
    print("=" * 80)
    
    pipeline = VehicleSpecRAGPipeline(config_path=args.config)
    
    try:
        pipeline.build_index(
            pdf_path=args.pdf_path,
            force_rebuild=args.force
        )
        
        stats = pipeline.get_status()
        print("\n" + "=" * 80)
        print("✓ Index built successfully!")
        print(f"  Total chunks: {stats.get('total_chunks', 'N/A')}")
        print(f"  Index directory: {pipeline.config['vector_store']['persist_directory']}")
        
    except Exception as e:
        print(f"\n✗ Index building failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


def cmd_query(args):
    """Query for specifications."""
    print(f"Query: {args.query}")
    print("=" * 80)
    
    pipeline = VehicleSpecRAGPipeline(config_path=args.config)
    
    try:
        # Load index
        pipeline.load_index()
        
        # Execute query
        result = pipeline.query(
            query_text=args.query,
            return_contexts=args.show_contexts
        )
        
        print("\nResults:")
        print("-" * 80)
        
        if result['specifications']:
            for i, spec in enumerate(result['specifications'], 1):
                print(f"\n{i}. {spec['component']}")
                print(f"   Type: {spec['spec_type']}")
                print(f"   Value: {spec['value']} {spec['unit']}")
                print(f"   Page: {spec.get('page_number', 'N/A')}")
                print(f"   Source: {spec.get('source_chunk_id', 'N/A')}")
        else:
            print("\nNo specifications found.")
            if result.get('message'):
                print(f"  {result['message']}")
        
        # Show contexts if requested
        if args.show_contexts and 'contexts' in result:
            print("\n" + "=" * 80)
            print("Retrieved Contexts:")
            print("-" * 80)
            for i, ctx in enumerate(result['contexts'], 1):
                print(f"\n[Context {i}] (Page {ctx['page_number']}, Score: {ctx['score']:.3f})")
                print(f"{ctx['text'][:300]}...")
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            
            if args.format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n✓ Results saved to {output_path}")
            
            elif args.format == 'csv':
                import csv
                with open(output_path, 'w', newline='') as f:
                    if result['specifications']:
                        writer = csv.DictWriter(f, fieldnames=result['specifications'][0].keys())
                        writer.writeheader()
                        writer.writerows(result['specifications'])
                    print(f"\n✓ Results saved to {output_path}")
    
    except FileNotFoundError:
        print("\n✗ No index found. Please build an index first using the 'index' command.", file=sys.stderr)
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Query failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


def cmd_status(args):
    """Show pipeline status."""
    pipeline = VehicleSpecRAGPipeline(config_path=args.config)
    
    try:
        pipeline.load_index()
        stats = pipeline.get_status()
        
        print("Pipeline Status:")
        print("=" * 80)
        print(f"  Initialized: {stats['initialized']}")
        print(f"  Embedding Model: {stats['embedding_model_loaded']}")
        print(f"  Index Loaded: {stats['index_loaded']}")
        print(f"  Extractor: {stats['extractor_initialized']}")
        
        if stats['index_loaded']:
            print(f"\nIndex Statistics:")
            print(f"  Total Chunks: {stats.get('total_chunks', 'N/A')}")
            print(f"  Embedding Dim: {stats.get('embedding_dim', 'N/A')}")
            print(f"  Index Type: {stats.get('index_type', 'N/A')}")
    
    except Exception as e:
        print(f"Status check failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


def cmd_server(args):
    """Start API server."""
    import uvicorn
    
    print(f"Starting API server on {args.host}:{args.port}")
    print("=" * 80)
    print(f"  API docs: http://{args.host}:{args.port}/docs")
    print(f"  Reload: {args.reload}")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80)
    
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vehicle Specification Extraction RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index from PDF
  python main.py index data/service-manual.pdf
  
  # Query for specifications
  python main.py query "What is the torque for brake caliper bolts?"
  
  # Save results to JSON
  python main.py query "Engine oil capacity" --output results.json
  
  # Start API server
  python main.py server --port 8000
  
  # Check status
  python main.py status
        """
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Build index from PDF')
    index_parser.add_argument('pdf_path', help='Path to service manual PDF')
    index_parser.add_argument('--force', '-f', action='store_true', help='Force rebuild index')
    index_parser.set_defaults(func=cmd_index)
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query for specifications')
    query_parser.add_argument('query', help='Natural language query')
    query_parser.add_argument('--output', '-o', help='Output file path')
    query_parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Output format')
    query_parser.add_argument('--show-contexts', '-c', action='store_true', help='Show retrieved contexts')
    query_parser.set_defaults(func=cmd_query)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show pipeline status')
    status_parser.set_defaults(func=cmd_status)
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='127.0.0.1', help='Server host')
    server_parser.add_argument('--port', type=int, default=8000, help='Server port')
    server_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    server_parser.set_defaults(func=cmd_server)
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


if __name__ == "__main__":
    main()
