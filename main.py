from __future__ import annotations

import json

import typer

from core.config import NexusConfig
from core.evaluation import RagasEvaluator
from core.service import NexusRAGService
from core.schemas import RetrievalFilters

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
    year_from: int | None = typer.Option(None, "--year-from", help="Filter by minimum year"),
    year_to: int | None = typer.Option(None, "--year-to", help="Filter by maximum year"),
    file_type: str | None = typer.Option(None, "--file-type", help="Filter by file type"),
    language: str | None = typer.Option(None, "--language", help="Filter by language"),
    source_contains: str | None = typer.Option(
        None,
        "--source-contains",
        help="Filter by source path substring",
    ),
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
    filters = RetrievalFilters(
        year_from=year_from,
        year_to=year_to,
        file_type=file_type,
        language=language,
        source_contains=source_contains,
    )
    result = service.answer(
        query=question,
        top_k=top_k,
        candidate_k=candidate_k,
        use_rerank=rerank,
        generate=generate,
        workflow=workflow,
        filters=filters,
    )
    payload = {
        "query": result.query,
        "answer": result.answer,
        "trace": result.trace,
        "filters": {
            "year_from": year_from,
            "year_to": year_to,
            "file_type": file_type,
            "language": language,
            "source_contains": source_contains,
        },
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


@app.command()
def evaluate(
    dataset: str = typer.Argument(..., help="Path to evaluation JSONL file"),
    output: str = typer.Option(
        "reports/ragas_eval.json",
        "--output",
        help="Path to save evaluation report JSON",
    ),
) -> None:
    service = _service()
    evaluator = RagasEvaluator(service)
    report = evaluator.evaluate_file(dataset_path=dataset, output_path=output)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command()
def desktop() -> None:
    from ui.desktop import launch_desktop

    raise typer.Exit(launch_desktop())


if __name__ == "__main__":
    app()
