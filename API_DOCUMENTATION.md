# REST API Documentation

The Vehicle Specification Extraction RAG System provides a comprehensive REST API built with FastAPI for programmatic access to all features.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. For production deployment, implement API key or JWT authentication.

---

## Health Check

Check system status and readiness.

### Request

```http
GET /health HTTP/1.1
Host: localhost:8000
```

### Response

**Status**: 200 OK

```json
{
  "status": "healthy",
  "pipeline_initialized": true,
  "index_loaded": true,
  "total_chunks": 12456
}
```

**Status Values**:
- `healthy` - System fully operational with index loaded
- `ready` - System ready but no PDF indexed
- `unhealthy` - System error or not initialized

### Curl Example

```bash
curl http://localhost:8000/health
```

---

## Upload PDF

Upload and index a PDF file for specification extraction.

### Request

```http
POST /upload_pdf HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data

file=@service-manual.pdf&force_rebuild=false
```

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | Yes | PDF file to upload (max 100MB) |
| force_rebuild | Boolean | No | Force rebuild of index (default: false) |

### Response

**Status**: 200 OK

```json
{
  "status": "success",
  "message": "PDF uploaded and indexed successfully",
  "num_chunks": 3245
}
```

**Status Codes**:
- `200 OK` - Successfully indexed
- `400 Bad Request` - Invalid file type or size
- `413 Payload Too Large` - File exceeds 100MB
- `500 Internal Server Error` - Processing failure

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| status | String | "success" or error description |
| message | String | Human-readable message |
| num_chunks | Integer | Number of text chunks created |

### Curl Example

```bash
curl -X POST \
  -F "file=@brake-specs-manual.pdf" \
  -F "force_rebuild=false" \
  http://localhost:8000/upload_pdf
```

### Python Example

```python
import requests

with open('manual.pdf', 'rb') as f:
    files = {'file': f}
    data = {'force_rebuild': False}
    response = requests.post(
        'http://localhost:8000/upload_pdf',
        files=files,
        data=data
    )
    print(response.json())
```

---

## Index PDF

Index a PDF from a file path (alternative to upload).

### Request

```http
POST /index_pdf HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "pdf_path": "/path/to/manual.pdf",
  "force_rebuild": false
}
```

### Request Body

```json
{
  "pdf_path": "string",
  "force_rebuild": "boolean (optional)"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| pdf_path | String | Yes | Absolute or relative path to PDF |
| force_rebuild | Boolean | No | Force rebuild (default: false) |

### Response

**Status**: 200 OK

```json
{
  "status": "success",
  "message": "Index built successfully",
  "num_chunks": 5432
}
```

Or for large files (>10MB):

```json
{
  "status": "processing",
  "message": "Index building started in background",
  "num_chunks": null
}
```

### Status Codes

- `200 OK` - Indexed successfully
- `202 Accepted` - Processing in background (large file)
- `404 Not Found` - PDF file not found
- `500 Internal Server Error` - Indexing failed

### Curl Example

```bash
curl -X POST http://localhost:8000/index_pdf \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "./data/vehicle-manual.pdf",
    "force_rebuild": false
  }'
```

### Python Example

```python
import requests

response = requests.post(
    'http://localhost:8000/index_pdf',
    json={
        'pdf_path': './data/manual.pdf',
        'force_rebuild': False
    }
)
print(response.json())
```

---

## Query Specifications

Extract specifications from indexed document using natural language query.

### Request

```http
POST /query HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "query": "What is the torque for brake caliper bolts?",
  "top_k": 5,
  "return_contexts": false
}
```

### Request Body

```json
{
  "query": "string",
  "top_k": "integer (optional)",
  "return_contexts": "boolean (optional)"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| query | String | Yes | - | Natural language specification query |
| top_k | Integer | No | 5 | Number of results to retrieve |
| return_contexts | Boolean | No | false | Include retrieved context chunks |

### Response

**Status**: 200 OK

```json
{
  "query": "What is the torque for brake caliper bolts?",
  "specifications": [
    {
      "component": "Brake Caliper Bolt",
      "spec_type": "Torque",
      "value": "24",
      "unit": "Nm",
      "page_number": 145,
      "source_chunk_id": "chunk_2847"
    },
    {
      "component": "Brake Caliper Bolt",
      "spec_type": "Torque Angle",
      "value": "90",
      "unit": "degrees",
      "page_number": 145,
      "source_chunk_id": "chunk_2847"
    }
  ],
  "num_results": 2,
  "message": null
}
```

### Specification Object

| Field | Type | Description |
|-------|------|-------------|
| component | String | Component name (e.g., "Brake Caliper Bolt") |
| spec_type | String | Type of specification (e.g., "Torque", "Pressure") |
| value | String | Specification value |
| unit | String | Measurement unit (e.g., "Nm", "bar") |
| page_number | Integer \| Null | Page number in source document |
| source_chunk_id | String \| Null | ID of source text chunk |

### Status Codes

- `200 OK` - Query successful
- `400 Bad Request` - No index loaded or invalid query
- `500 Internal Server Error` - Query processing failed
- `503 Service Unavailable` - Pipeline not initialized

### Query Examples

```
"What is the torque for brake calipers?"
"Engine oil capacity"
"Tire pressure specifications for all wheels"
"Coolant type and capacity"
"Spark plug gap specification"
```

### Curl Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "engine oil capacity",
    "top_k": 5,
    "return_contexts": false
  }'
```

### Python Example

```python
import requests

response = requests.post(
    'http://localhost:8000/query',
    json={
        'query': 'What is the brake fluid type?',
        'top_k': 5,
        'return_contexts': False
    }
)

data = response.json()
print(f"Found {data['num_results']} specifications:")
for spec in data['specifications']:
    print(f"  {spec['component']}: {spec['value']} {spec['unit']}")
```

### With Contexts

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "coolant capacity",
    "top_k": 5,
    "return_contexts": true
  }'
```

Returns additional `contexts` array:

```json
{
  "contexts": [
    {
      "text": "Coolant capacity: 8.5 liters...",
      "page_number": 78,
      "score": 0.92
    }
  ]
}
```

---

## Statistics

Get pipeline statistics and current status.

### Request

```http
GET /stats HTTP/1.1
Host: localhost:8000
```

### Response

**Status**: 200 OK

```json
{
  "initialized": true,
  "embedding_model_loaded": true,
  "index_loaded": true,
  "extractor_initialized": true,
  "total_chunks": 12456,
  "embedding_dim": 384,
  "index_type": "IndexFlatL2",
  "index_size_mb": 45.2,
  "model_info": {
    "embeddings": "sentence-transformers/all-MiniLM-L6-v2",
    "llm": "llama3",
    "llm_provider": "ollama"
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| initialized | Boolean | Pipeline initialization status |
| embedding_model_loaded | Boolean | Embeddings available |
| index_loaded | Boolean | Vector index loaded |
| extractor_initialized | Boolean | LLM extractor ready |
| total_chunks | Integer | Number of indexed chunks |
| embedding_dim | Integer | Embedding vector dimensions |
| index_type | String | FAISS index type used |
| index_size_mb | Float | Index file size in MB |
| model_info | Object | Model configuration details |

### Curl Example

```bash
curl http://localhost:8000/stats
```

---

## Error Responses

All endpoints return standardized error responses.

### Format

```json
{
  "detail": "Error description"
}
```

### Common Error Codes

| Code | Message | Cause |
|------|---------|-------|
| 400 | No index loaded | PDF not indexed yet |
| 400 | PDF file not found | File path doesn't exist |
| 400 | Only PDF files supported | Wrong file type uploaded |
| 404 | Not Found | Invalid endpoint |
| 413 | Payload Too Large | File exceeds 100MB |
| 500 | Index building failed | PDF processing error |
| 500 | Query failed | LLM/extraction error |
| 503 | Pipeline not initialized | System startup incomplete |

### Example Error Response

```json
{
  "detail": "No index loaded. Please index a PDF first using /index_pdf endpoint"
}
```

---

## Rate Limiting

No rate limiting is currently implemented. For production, consider:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/query")
@limiter.limit("10/minute")
async def query_specifications(request: Request, ...):
    ...
```

---

## CORS Configuration

Current CORS allows all origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For production, restrict to specific origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
```

---

## Async Operations

Large file indexing (>10MB) is processed asynchronously. Check status via:

```bash
curl http://localhost:8000/health
```

The `index_loaded` field indicates when indexing is complete.

---

## Webhooks (Future)

Planned webhook support for async completion notifications:

```json
{
  "url": "https://yourdomain.com/webhook/index-complete",
  "events": ["index.complete", "index.failed"]
}
```

---

## Code Examples by Language

### Python (requests)

```python
import requests

# Query specifications
response = requests.post(
    'http://localhost:8000/query',
    json={'query': 'engine torque specifications'}
)

if response.status_code == 200:
    data = response.json()
    for spec in data['specifications']:
        print(f"{spec['component']}: {spec['value']} {spec['unit']}")
else:
    print(f"Error: {response.json()['detail']}")
```

### JavaScript (fetch)

```javascript
// Upload PDF
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const uploadResponse = await fetch(
  'http://localhost:8000/upload_pdf',
  {
    method: 'POST',
    body: formData
  }
);

// Query specifications
const queryResponse = await fetch(
  'http://localhost:8000/query',
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: 'brake specifications',
      top_k: 5
    })
  }
);

const data = await queryResponse.json();
console.log(`Found ${data.num_results} specifications`);
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Upload PDF
curl -X POST -F "file=@manual.pdf" http://localhost:8000/upload_pdf

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"engine capacity"}'

# Stats
curl http://localhost:8000/stats
```

---

## API Versioning

Future versions will be available at `/api/v2/`, `/api/v3/`, etc.

Current version is at `/` (root).

---

**Last Updated**: February 2025
**API Version**: 1.0
