from __future__ import annotations

import json

import typer

from core.config import NexusConfig
from core.service import NexusRAGService

app = typer.Typer(help="NexusSearch: local-first hybrid RAG system")


def _service() -> NexusRAGService:
    return NexusRAGService(NexusConfig())


@app.command()
def ingest(
    source: str = typer.Argument("data", help="File or directory to ingest"),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing vector table",
    ),
) -> None:
    service = _service()
    summary = service.ingest(source=source, overwrite=overwrite)
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(6, "--top-k", help="Final result count"),
    candidate_k: int = typer.Option(24, "--candidate-k", help="Candidate pool"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Use reranker"),
    workflow: str = typer.Option(
        "self-rag",
        "--workflow",
        help="Workflow type: vanilla or self-rag",
    ),
    generate: bool = typer.Option(
        True,
        "--generate/--no-generate",
        help="Use local LLM to generate final answer",
    ),
) -> None:
    service = _service()
    result = service.answer(
        query=question,
        top_k=top_k,
        candidate_k=candidate_k,
        use_rerank=rerank,
        generate=generate,
        workflow=workflow,
    )
    payload = {
        "query": result.query,
        "answer": result.answer,
        "trace": result.trace,
        "contexts": [
            {
                "source_path": chunk.source_path,
                "header_path": chunk.header_path,
                "score": round(chunk.score, 6),
                "metadata": chunk.metadata,
                "retrieval_scores": {
                    key: round(value, 6)
                    for key, value in chunk.retrieval_scores.items()
                },
                "text": chunk.text[:500],
            }
            for chunk in result.contexts
        ],
    }
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    import uvicorn

    uvicorn.run("api.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
