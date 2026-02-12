# Vehicle Specification Extractor — Demo Walkthrough

This document demonstrates the end-to-end workflow of the Vehicle Specification Extraction system using Retrieval-Augmented Generation (RAG).

The system extracts structured vehicle specifications from service manuals and allows semantic search over them.

---

## 1. Application Interface

The application UI allows users to upload a vehicle service manual and query specifications.

![Application UI](assets/ui_overview.png)

Once a manual is uploaded, the system:
- Extracts text from the PDF
- Splits content into chunks
- Generates embeddings
- Stores them in the vector database

Example system status:
System ready (1392 chunks indexed)

---

## 2. Searching Specifications

Users can search using natural language queries.

Example query:
engine oil

The retriever performs similarity search on the vector database and sends relevant context to the LLM extraction pipeline.

![Search Interface](assets/search_example.png)

---

## 3. Extracted Specification Result

The system extracts structured specification data grounded in the source manual.

Example result:

- Component: Engine Oil
- Specification Type: Capacity
- Value: 1.66 liters
- Page Number: 515
- Source Chunk ID: chunk_00746

![Extraction Result](assets/extraction_result.png)

This demonstrates retrieval-grounded specification extraction.

---

## 4. JSON Export

The extracted specifications can be downloaded as a JSON file.

Example output:

```json
[
  {
    "component": "Engine Oil",
    "spec_type": "Capacity",
    "value": "1.66",
    "unit": "liters",
    "page_number": 515,
    "source_chunk_id": "chunk_00746"
  }
]
![JSON OUTPUT](assets/json_output.png)

## 5. CSV Export

The system also supports CSV export for spreadsheet workflows and downstream processing.

Columns: Component Type Value Unit Page Source

![CSV OUTPUT](assets/csv_output.png)