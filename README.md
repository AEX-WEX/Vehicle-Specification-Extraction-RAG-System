# Vehicle Specification Extraction RAG System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Extracts structured specifications (torque, fluid capacity, clearances) from vehicle service-manual PDFs using retrieval-augmented generation. Everything runs locally: Ollama (Llama 3) for the language model, FAISS and sentence-transformers for retrieval. No API cost and no document leaves the machine.

Every result carries the component, spec type, value, unit, **source page** and **source chunk ID**, so each number can be traced back to the manual.

## How it works

```
PDF ─► parse (PyMuPDF, table rows rebuilt from word positions)
    ─► table-aware chunking (each spec table is one chunk, prefixed with its manual section)
    ─► embed (all-MiniLM-L6-v2) ─► FAISS index
query ─► embedding search over a wide pool ─► lexical rerank ─► top-k chunks ─► SmartExtractor
                                   ├─ table parser   (deterministic: torque rows and fill capacities)
                                   ├─ Ollama / Llama 3 (prose and fluid capacities)
                                   └─ rule-based regex extractor (fallback)
```

`SmartExtractor` uses the table parser when a table row matches, since that is exact. Otherwise it runs Ollama and the regex extractor, requires the LLM to quote its evidence, keeps only answers that are traceable to the retrieved text and relevant to the question, and returns the better result. If nothing survives, or the retrieved text doesn't cover the question, it says so instead of guessing (see [docs/EVALUATION.md](docs/EVALUATION.md)). Each response says which method produced it.

**Where the page number comes from.** Every chunk is created from a single PDF page and keeps that 1-based page number. A result's page is the page of the chunk the value was read from (the table parser uses the chunk of the matched row; the LLM path matches the value back to the retrieved chunk that contains it). Pages are PDF page indices, not the section-relative "Page 2 sur 3" printed in the manual's header.

| Module | Role |
|---|---|
| `src/pdf_loader.py`, `src/chunking.py` | parse the PDF and split it into overlapping chunks with page metadata |
| `src/embeddings.py`, `src/vector_store.py` | embeddings and the FAISS index (persisted in `index/`) |
| `src/retriever.py` | top-k and hybrid (semantic + keyword) retrieval |
| `src/extractor.py` | Ollama, rule-based and combined extraction |
| `src/pipeline.py` | orchestration |
| `api/app.py`, `static/index.html` | FastAPI service and web UI |
| `eval/` | ground-truth set and scoring (see below) |

Design details are in [ARCHITECTURE.md](ARCHITECTURE.md); the REST API is in [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

## Quick start

Requires Python 3.9+ and [Ollama](https://ollama.ai) with the model pulled (`ollama pull llama3`).

```bash
git clone https://github.com/AEX-WEX/Vehicle-Specification-Extraction-RAG-System.git
cd Vehicle-Specification-Extraction-RAG-System
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

python main.py index data/sample-service-manual_1.pdf --force   # build the index (rebuild after any parsing/chunking change)
python main.py server --port 8000                       # web UI at http://localhost:8000
python main.py query "rear shock absorber nuts torque"  # or use the CLI
```

REST example:

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
     -d '{"query": "front brake caliper guide pin bolts torque"}'
```

Docker: `docker-compose up -d` starts the API together with Ollama. The sample manual is not copied into the image; mount your own `data/`.

## Evaluation

Three labeled sets from the 852-page 2014 Ford F-150 workshop manual (`data/sample-service-manual_1.pdf`) measure different things:

| Set | Size | Purpose |
|---|---|---|
| `eval/ground_truth.json` (dev) | 43 specs | regression checks; used while building the parser |
| `eval/heldout.json` | 25 specs | never used for tuning, short natural queries: **the number to quote** |
| `eval/unanswerable.json` | 12 queries | questions not in the manual; the system must return nothing |

```bash
python -m eval.run_eval                     # all sets -> eval/report.md
python -m eval.run_eval --sets heldout unanswerable --min-exact 0.90 --max-false-answer 0.10   # release gate
python -m eval.run_eval --predictions eval/predictions.json                                      # re-score saved output
```

It reports exact-match accuracy (component, value and unit), page accuracy, wrong-value rate, false-answer rate on unanswerable questions, and latency. Results from the latest run are in [`eval/report.md`](eval/report.md). Metric definitions, confidence caveats and the production checklist are in [docs/EVALUATION.md](docs/EVALUATION.md).

## Known limitations

- The "Material" (fill capacity) parser relies on Ford's specification-code format (WSL-, ESA-...); other manufacturers' capacity tables fall back to the LLM and need Ollama running.
- Only torque and fill-capacity tables are parsed deterministically; clearances and other "General Specifications" go through the LLM.
- Each evaluation set comes from one manual; a new manual needs its own labeled set (see docs/EVALUATION.md).
- Table rows that share a name across sections (for example "Shock absorber nuts" appears in several suspension and axle tables) can be attributed to the wrong page or unit. An early run returned `136 lb-in` for the spring shackle nut, where the manual says 185 Nm / 136 lb-ft.
- Rows with footnotes ("refer to the procedure in this section") have no value and are skipped.
- One manual per index; no cross-document search.
- There is no authentication, rate limiting or caching in the API. Treat it as a single-user, local tool.

## Development

```bash
pip install -r requirements.txt && pip install -e .
pytest tests/test_table_parser.py tests/test_eval_scoring.py   # fast unit tests (also run in CI)
pytest tests/                  # LLM and integration tests need Ollama running and an index built
black src api tests eval && flake8 src api tests eval
```

`.env.example` lists the configuration keys; `config.yaml` controls chunking, retrieval and LLM parameters.

## License

MIT, see [LICENSE](LICENSE). Author: Aditya Kumar Choudhary ([AEX-WEX](https://github.com/AEX-WEX)).
