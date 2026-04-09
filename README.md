# NexusSearch

NexusSearch is a public, local-first RAG system built to demonstrate modern retrieval engineering rather than a minimal chatbot demo.

## Design Goals

- Use recent, interview-relevant methods instead of outdated PDF-only parsing.
- Keep the stack practical for local development and public open-source usage.
- Show clear system thinking across parsing, chunking, retrieval, orchestration, and serving.

## Current Stack

- Parsing: `Docling` for rich documents, `Tree-sitter` for code-aware parsing when available
- Chunking: structure-aware chunking plus embedding-driven semantic chunking
- Storage: `LanceDB`
- Retrieval: dense retrieval + sparse BM25 + weighted `RRF`
- Reranking: `BGE-Reranker-v2-m3` when available
- Orchestration: `Self-RAG` loop with optional `LangGraph`
- Serving: `FastAPI` + `Typer`
- Local inference: `Ollama`

## What Makes This Project Stronger Than A Basic RAG Demo

- It treats parsing quality as a first-class problem.
- It distinguishes code files from text documents.
- It supports semantic chunk boundaries instead of fixed-size slicing only.
- It uses a retrieval-feedback loop with `retrieve -> grade -> rewrite -> generate`.
- It keeps optional dependencies graceful so the project is still runnable on a fresh machine.

## Repository Layout

```text
.
|-- api/
|   `-- server.py
|-- core/
|   |-- chunker.py
|   |-- config.py
|   |-- indexer.py
|   |-- llm.py
|   |-- parser.py
|   |-- retriever.py
|   |-- schemas.py
|   |-- service.py
|   `-- workflow.py
|-- tests/
|   |-- test_chunker.py
|   |-- test_retriever_fusion.py
|   `-- test_workflow.py
|-- main.py
|-- requirements.txt
`-- environment.yml
```

## Quick Start

```bash
conda env create -f environment.yml
conda activate NexusSearch
pip install -r requirements.txt
```

Ingest documents:

```bash
python main.py ingest data --overwrite
```

Ask with Self-RAG:

```bash
python main.py query "What retrieval strategy does this system use?" --workflow self-rag
```

Run the API:

```bash
python main.py serve --host 127.0.0.1 --port 8000
```

## Resume Framing

- Built an open-source local RAG engine with structure-aware parsing, semantic chunking, hybrid retrieval, and weighted reciprocal rank fusion.
- Implemented a Self-RAG workflow with retrieval grading and query rewriting, exposed through CLI and FastAPI interfaces.
- Designed graceful fallbacks for optional dependencies so the system remains usable across constrained local environments.

## Honest Status

- Implemented now: Docling integration, Tree-sitter-aware code parsing fallback path, semantic chunking, hybrid retrieval, reranking hooks, Self-RAG orchestration, CLI/API.
- Planned next: RAGAS evaluation, SSE streaming, polished web UI, and vLLM deployment mode for performance demos.
