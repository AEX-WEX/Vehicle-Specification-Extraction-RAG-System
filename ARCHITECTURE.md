# System Architecture

## Overview

The Vehicle Specification Extraction RAG (Retrieval-Augmented Generation) system combines modern NLP techniques with local language models to extract structured specifications from PDF documents.

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface                            │
│  ┌──────────────────┐         ┌──────────────────────────┐  │
│  │   Web UI (SPA)   │         │    REST API (FastAPI)    │  │
│  │  - Upload        │         │  - /upload_pdf           │  │
│  │  - Query         │         │  - /query                │  │
│  │  - Results       │         │  - /health               │  │
│  └──────────────────┘         └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  RAG Pipeline (Pipeline.py)                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Orchestrates all components of the extraction flow  │   │
│  │  - Manages state                                     │   │
│  │  - Coordinates subsystems                           │   │
│  │  - Handles errors and fallbacks                     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │              │              │              │
    ┌────▼──┐    ┌─────▼─────┐   ┌────▼──┐    ┌────▼──────┐
    │  PDF  │    │  Vector   │   │  LLM  │    │ Embedding │
    │ Loader│    │  Store    │   │Module │    │  Module   │
    └───┬──┘    └─────┬─────┘   └────┬──┘    └────┬──────┘
        │             │              │              │
        ▼             ▼              ▼              ▼
    ┌────────────────────────────────────────────────────┐
    │           Core Processing Modules                  │
    │  - Chunking    - Retrieval    - Extraction        │
    └────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Document Processing Layer

#### PDF Loader (`src/pdf_loader.py`)
```python
PDFLoader
├── Loaders: PyMuPDF, PDFMiner
├── Processing Pipeline:
│   ├── Document Loading
│   ├── Text Extraction
│   ├── Metadata Extraction
│   ├── Table Detection (optional)
│   └── Image Extraction (optional)
└── Outputs: RawDocument with pages and metadata
```

**Workflow**:
1. Load PDF file using specified loader
2. Extract text from each page
3. Extract metadata (page numbers, section titles)
4. Normalize and clean text
5. Return structured document

**Configuration**:
- `loader`: pymupdf | pdfminer
- `extract_tables`: bool
- `extract_images`: bool

---

### 2. Text Processing Layer

#### Chunking (`src/chunking.py`)
```
Raw Document
    │
    ├─► Normalize Whitespace
    │
    ├─► Remove Headers/Footers
    │
    ├─► Apply Chunking Strategy
    │   ├── Semantic Chunking
    │   ├── Fixed-size Chunking
    │   └── Recursive Chunking
    │
    ├─► Add Overlap
    │
    └─► Add Metadata (page numbers, source)
            │
            ▼
        Document Chunks
```

**Chunking Strategies**:

1. **Semantic Chunking** (Recommended)
   - Splits on natural boundaries (paragraphs, sentences)
   - Preserves semantic coherence
   - Configurable separator hierarchy

2. **Fixed-size Chunking**
   - Splits every N characters
   - Consistent chunk sizes
   - Simple and deterministic

3. **Recursive Chunking**
   - Attempts sentence-level splits first
   - Falls back to word/character splits
   - Balances semantic preservation with consistency

**Parameters**:
- `min_chunk_size`: Minimum characters per chunk
- `max_chunk_size`: Maximum characters per chunk
- `chunk_overlap`: Characters to overlap between chunks

---

### 3. Embedding Layer

#### Embeddings (`src/embeddings.py`)
```
Document Chunks
    │
    ├─► Model Initialization
    │   └── Load: sentence-transformers model
    │
    ├─► Batch Processing
    │   ├─► Tokenization
    │   ├─► Encoding
    │   └─► Normalization (L2)
    │
    └─► Output: 384-768 dimension vectors
```

**Embedding Models**:

| Model | Dimensions | Size | Speed | Quality |
|-------|-----------|------|-------|---------|
| all-MiniLM-L6-v2 | 384 | ~22MB | Fast | Good |
| all-mpnet-base-v2 | 768 | ~420MB | Slow | Excellent |
| BAAI/bge-small-en-v1.5 | 384 | ~30MB | Fast | Excellent |

**Configuration**:
- `model_name`: HuggingFace model identifier
- `batch_size`: Batch size for inference
- `device`: cpu | cuda | mps
- `normalize`: L2 normalization flag

---

### 4. Vector Storage Layer

#### Vector Store (`src/vector_store.py`)
```
Embedding Vectors
    │
    └─► FAISS Index
        ├── Index Type
        │   ├── IndexFlatL2 (exact search)
        │   └── IndexIVFFlat (approximate search)
        │
        ├── Storage
        │   ├── Memory
        │   └── Disk (.faiss file)
        │
        ├── Metadata
        │   └── Chunk metadata (chunk_id -> chunk)
        │
        └── Operations
            ├── Add vectors
            ├── Search (k-NN)
            ├── Save/Load
            └── Statistics
```

**Index Operations**:
1. **Index Creation**: Build FAISS index from embeddings
2. **Storage**: Persist to disk in `./index/` directory
3. **Retrieval**: Fast k-NN search for query vectors
4. **Metadata**: Map vector indices to original chunks

**Configuration**:
- `index_type`: IndexFlatL2 | IndexIVFFlat
- `persist_directory`: Path to index files
- `index_name`: Identifier for index

---

### 5. Retrieval Layer

#### Retriever (`src/retriever.py`)
```
Query Text
    │
    ├─► Embed Query
    │   └── Same embedder as corpus
    │
    ├─► Search FAISS Index
    │   ├─► Top-k nearest neighbors
    │   └─► Similarity scores
    │
    └─► Output: Retrieved Chunks
        ├── Chunk text
        ├── Similarity score
        ├── Page number
        └── Metadata
```

**Retrieval Process**:
1. Embed user query using same model
2. Perform k-NN search in FAISS index
3. Retrieve top-k similar chunks
4. Apply optional score threshold
5. Format as context for LLM

**Parameters**:
- `top_k`: Number of chunks to retrieve (default: 5)
- `score_threshold`: Minimum similarity score
- `max_context_length`: Maximum total context size

---

### 6. LLM Layer

#### Extraction (`src/extractor.py`)
```
Retrieved Context + Query
    │
    ├─► Format Prompt
    │   ├── System prompt
    │   ├── Retrieved chunks
    │   ├── Query
    │   └── JSON schema
    │
    ├─► Call LLM (Ollama)
    │   ├── Temperature: 0.0 (deterministic)
    │   ├── Max tokens: 500
    │   └── Timeout: 60s
    │
    ├─► Parse Response
    │   ├── Extract JSON
    │   ├── Validate schema
    │   └── Handle errors
    │
    ├─► Fallback Strategy
    │   ├── Retry on timeout
    │   ├── Rule-based extraction
    │   └── Return empty if all fail
    │
    └─► Output: Specifications
        ├── Component
        ├── Type
        ├── Value
        ├── Unit
        └── Metadata
```

**LLM Providers Supported**:

1. **Ollama** (Only Supported Provider)
   - Local inference
   - Model: Llama3
   - No API costs
   - Complete privacy

**Prompt Template**:
```
You are an expert technical specification extractor.
Extract vehicle specifications from the given context.

Context:
{context}

Query: {query}

Return a JSON array with specifications. Each item should have:
- component: Component name
- spec_type: Type of specification
- value: Numeric or text value
- unit: Measurement unit

Only return valid specifications found in context.
If no specifications match, return empty array.
```

---

### 7. Pipeline Orchestration

#### Main Pipeline (`src/pipeline.py`)
```
User Request
    │
    ├─► Initialize Pipeline
    │   ├── Load config
    │   ├── Initialize embedder
    │   ├── Load vector index
    │   └── Initialize LLM
    │
    ├─► Build Index (from PDF)
    │   ├── Load PDF (PDFLoader)
    │   ├── Chunk text (Chunking)
    │   ├── Embed chunks (Embeddings)
    │   ├── Store in FAISS (VectorStore)
    │   └── Save to disk
    │
    ├─► Query (Search)
    │   ├── Embed query (Embeddings)
    │   ├── Retrieve chunks (Retriever)
    │   ├── Extract specs (Extractor)
    │   └── Format response
    │
    └─► Return Results
        ├── Specifications array
        ├── Query text
        ├── Result count
        └── Optional contexts
```

**Pipeline Methods**:
```python
class VehicleSpecRAGPipeline:
    def __init__(config_path: str)
    def build_index(pdf_path: str, force_rebuild: bool)
    def load_index()
    def query(query_text: str, return_contexts: bool) -> Dict
    def get_status() -> Dict
```

---

### 8. API Layer

#### FastAPI Application (`api/app.py`)
```
HTTP Request
    │
    ├─► Route Handler
    │   ├── /health → Health Check
    │   ├── /upload_pdf → Upload & Index
    │   ├── /index_pdf → Index from Path
    │   ├── /query → Extract Specs
    │   └── /stats → Statistics
    │
    ├─► Pipeline Call
    │   └── Execute extraction
    │
    ├─► Format Response
    │   ├── Pydantic models
    │   ├── JSON serialization
    │   └── Error handling
    │
    └─► HTTP Response
        ├── Status code
        ├── JSON body
        └── Headers
```

**Endpoints**:

| Method | Path | Purpose | async |
|--------|------|---------|-------|
| GET | /health | System status | Yes |
| POST | /upload_pdf | Upload & index PDF | Yes |
| POST | /index_pdf | Index from file path | Yes |
| POST | /query | Query specifications | Yes |
| GET | /stats | Pipeline statistics | Yes |

**Background Processing**:
- Large PDFs (>10MB) indexed in background
- Non-blocking user experience
- Polling via /stats for completion

---

### 9. Web UI Layer

#### Frontend (`static/index.html`)
```
User Interface
    │
    ├─► Upload Section
    │   ├── Drag & drop
    │   ├── File selection
    │   ├── Progress tracking
    │   └── Status display
    │
    ├─► Query Section
    │   ├── Input field
    │   ├── Search button
    │   └── Message display
    │
    ├─► Results Display
    │   ├── Specification cards
    │   ├── Metadata display
    │   └── Export buttons
    │
    └─► API Calls
        ├── /health (health check)
        ├── /upload_pdf (upload)
        ├── /query (search)
        └── /stats (status)
```

**Features**:
- Single-page application (no frameworks)
- Vanilla JavaScript
- Real-time health monitoring
- Responsive design
- CSV/JSON export

---

## Data Flow Diagrams

### Building an Index

```
PDF File
    │
    ▼
┌─────────────────┐
│   PDF Loader    │  Extract text from pages
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Chunking      │  Split into overlapping chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │  Generate embeddings for chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FAISS Index   │  Store vectors and metadata
└────────┬────────┘
         │
         ▼
    Disk Storage (/index/)
```

**Performance**: 852-page manual → ~15-20 minutes (CPU), ~5-10 minutes (GPU)

---

### Querying for Specifications

```
User Query
    │
    ▼
┌──────────────────┐
│  Embed Query     │  Create embedding vector
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  FAISS Search    │  Find top-k similar chunks
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Format Context   │  Combine chunks into prompt
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Call LLM       │  Extract specifications from context
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Parse Output    │  Validate JSON response
└────────┬─────────┘
         │
         ▼
  Specifications Array

Duration: 2-5 seconds (includes network latency to Ollama)
```

---

## Concurrency & Performance

### Threading Model
```
API Server (uvicorn)
    │
    ├─► Thread Pool (for sync operations)
    │   └── File I/O, CPU-bound tasks
    │
    ├─► Async/Await
    │   ├── API handlers
    │   ├── Background tasks
    │   └── External API calls
    │
    └─► Global Pipeline Instance
        ├── Thread-safe (no concurrent indexing)
        └── Safe for concurrent queries
```

### Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Load PDF | 10-30s | Depends on file size |
| Chunk text | 5-10s | Configurable strategy |
| Generate embeddings | 2-5 min | Batch of 1000+ chunks |
| Build FAISS index | 1-2 min | Index type dependent |
| Query (embed → search) | 0.5-1s | Fast retrieval |
| LLM extraction | 1-3s | Network + inference time |
| **Total Query** | **2-5s** | End-to-end |

---

## Error Handling & Resilience

### Error Recovery Strategy
```
Operation Fails
    │
    ├─► Log Error
    │
    ├─► Try Fallback
    │   ├── Alternative method
    │   ├── Cached result
    │   ├── Default response
    │   └── Partial result
    │
    ├─► Return Best Result
    │   └── Empty if no fallback
    │
    └─► User Notification
        └── Error message in response
```

### Key Safeguards
1. **PDF Parsing**: Try multiple loaders (PyMuPDF → pdfminer)
2. **Embedding Failures**: Skip problematic chunks
3. **LLM Timeouts**: Return empty specifications
4. **Index Corruption**: Auto-rebuild on load failure
5. **Out of Memory**: Reduce batch sizes automatically

---

## Configuration Hierarchy

```
Defaults (hardcoded)
    ▼
config.yaml (project-level)
    ▼
.env (environment-level)
    ▼
Command-line arguments
    ▼
Environment variables (override all)
```

**Example**: `max_chunk_size`
```
Hardcoded: 800
config.yaml: max_chunk_size: 1000
.env: (not applicable)
CLI/ENV: (can override)
Final value: 1000
```

---

## Security Considerations

### Input Validation
- PDF file type validation
- Query length limits
- JSON schema validation
- File path sanitization

### Output Sanitization
- HTML escape for web UI
- JSON serialization safety
- CSV escaping for export

### Resource Limits
- Max file upload size: 100MB
- Max query length: 1000 characters
- Timeout limits on all operations
- Memory limits via configuration

### Access Control
- CORS configured for same-origin by default
- API key support (future enhancement)
- No authentication by default (for internal use)

---

## Scalability Considerations

### Current Architecture Limits
- Single PDF index at a time
- No query queuing
- No distributed indexing
- Single machine only

### Scaling Strategies
1. **Horizontal Scaling**
   - Load balancer in front
   - Multiple API instances
   - Shared vector store (network drive)
   - Distributed embeddings

2. **Vertical Scaling**
   - GPU acceleration
   - Larger machines
   - Increased batch sizes

3. **Caching Layer**
   - Redis for query results
   - LRU cache for embeddings
   - Memoization of common queries

---

## Future Architecture Enhancements

### Near-term
- Query result caching
- Multi-document indexes
- Batch query processing
- Advanced filtering

### Long-term
- Distributed processing
- Real-time indexing
- Specification validation DB
- Multi-language support
- GraphQL API
- WebSocket streaming

---

## Testing Architecture

```
Unit Tests (src/, api/)
    ├── Component tests
    └── Function tests

Integration Tests
    ├── Pipeline tests
    └── API endpoint tests

End-to-End Tests
    ├── Full workflow
    └── UI tests

Performance Tests
    ├── Throughput
    ├── Latency
    └── Memory usage
```

---

**Last Updated**: February 2025
**Version**: 1.0.0
