"""
FastAPI Application

REST API for vehicle specification extraction.
"""

import os
import logging
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from src.pipeline import VehicleSpecRAGPipeline
from src.utils import load_indexed_pdf_metadata
from api.models import (
    QueryRequest,
    QueryResponse,
    Specification,
    IndexRequest,
    IndexResponse,
    HealthResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global pipeline instance
pipeline: Optional[VehicleSpecRAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global pipeline
    
    # Startup
    logger.info("Initializing pipeline...")
    try:
        pipeline = VehicleSpecRAGPipeline(config_path="config.yaml")
        
        # Try to load existing index
        try:
            pipeline.load_index()
            logger.info("Loaded existing index")
        except Exception as e:
            logger.warning(f"No existing index found: {e}")
    
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        pipeline = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Vehicle Specification Extraction API",
    description="RAG-based system for extracting vehicle specifications from service manuals",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint - serve UI."""
    static_dir = Path(__file__).parent.parent / "static"
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return {
        "message": "Vehicle Specification Extraction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and readiness.
    """
    if pipeline is None:
        return HealthResponse(
            status="unhealthy",
            pipeline_initialized=False,
            index_loaded=False
        )
    
    status_info = pipeline.get_status()
    
    return HealthResponse(
        status="healthy" if status_info['index_loaded'] else "ready",
        pipeline_initialized=status_info['initialized'],
        index_loaded=status_info['index_loaded'],
        total_chunks=status_info.get('total_chunks')
    )


@app.post("/index_pdf", response_model=IndexResponse)
async def index_pdf(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Index a PDF service manual.
    
    This endpoint builds the vector index from a PDF file.
    For large PDFs, this may take several minutes.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    pdf_path = Path(request.pdf_path)
    
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF file not found: {request.pdf_path}")
    
    try:
        logger.info(f"Building index from {request.pdf_path}")
        
        # Run in background for large files
        if pdf_path.stat().st_size > 10 * 1024 * 1024:  # > 10 MB
            background_tasks.add_task(pipeline.build_index, request.pdf_path, request.force_rebuild)
            return IndexResponse(
                status="processing",
                message="Index building started in background",
                num_chunks=None
            )
        else:
            pipeline.build_index(request.pdf_path, request.force_rebuild)
            stats = pipeline.get_status()
            return IndexResponse(
                status="success",
                message="Index built successfully",
                num_chunks=stats.get('total_chunks')
            )
    
    except Exception as e:
        logger.error(f"Index building failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Index building failed: {str(e)}")


@app.post("/upload_pdf", response_model=IndexResponse)
async def upload_pdf(file: UploadFile = File(...), force_rebuild: bool = False):
    """
    Upload and index a PDF file.

    This endpoint accepts a PDF upload and automatically handles index building.
    If the uploaded PDF is different from the previously indexed PDF, the index
    is automatically rebuilt. If the same PDF is uploaded again, the existing
    index is reused.

    Args:
        file: PDF file to upload
        force_rebuild: Force rebuild of index even if PDF matches (default: False)

    Returns:
        IndexResponse with status and chunk count
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    pdf_path = data_dir / file.filename

    try:
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Saved uploaded file to {pdf_path}")

        # Build index (will auto-detect if PDF is different)
        # force_rebuild parameter allows manual override
        pipeline.build_index(str(pdf_path), force_rebuild)
        stats = pipeline.get_status()

        return IndexResponse(
            status="success",
            message=f"PDF uploaded and indexed successfully (force_rebuild={force_rebuild})",
            num_chunks=stats.get('total_chunks')
        )

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        if pdf_path.exists():
            pdf_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/indexed_pdf_info")
async def get_indexed_pdf_info():
    """
    Get information about the currently indexed PDF.

    Returns metadata about which PDF is currently indexed, when it was indexed,
    and other relevant information.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    index_dir = pipeline.config['vector_store']['persist_directory']
    metadata = load_indexed_pdf_metadata(index_dir)

    if metadata is None:
        return {
            "status": "not_indexed",
            "message": "No PDF has been indexed yet"
        }

    return {
        "status": "indexed",
        "pdf_filename": metadata.get("pdf_filename"),
        "pdf_path": metadata.get("pdf_path"),
        "pdf_hash": metadata.get("pdf_hash"),
        "indexed_at": metadata.get("indexed_at"),
        "num_chunks": metadata.get("num_chunks"),
        "embedding_model": metadata.get("embedding_model"),
        "index_type": metadata.get("index_type")
    }


@app.post("/query", response_model=QueryResponse)
async def query_specifications(request: QueryRequest):
    """
    Query for vehicle specifications.
    
    Extracts structured specifications from the indexed service manual
    based on a natural language query.
    
    Example queries:
    - "What is the torque for brake caliper bolts?"
    - "Engine oil capacity"
    - "Tire pressure specifications"
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not pipeline.get_status()['index_loaded']:
        raise HTTPException(
            status_code=400,
            detail="No index loaded. Please index a PDF first using /index_pdf endpoint"
        )
    
    try:
        result = pipeline.query(
            query_text=request.query,
            return_contexts=request.return_contexts
        )
        
        # Convert to response model
        specs = [
            Specification(**spec)
            for spec in result['specifications']
        ]
        
        return QueryResponse(
            query=result['query'],
            specifications=specs,
            num_results=result['num_results'],
            message=result.get('message')
        )
    
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/stats", response_model=dict)
async def get_stats():
    """Get pipeline statistics."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return pipeline.get_status()


# Run server
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vehicle Spec Extraction API")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
