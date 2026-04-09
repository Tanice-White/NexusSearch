from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.config import NexusConfig
from core.schemas import RetrievalFilters

if TYPE_CHECKING:
    from core.service import NexusRAGService

app = FastAPI(
    title="NexusSearch API",
    version="1.0.0",
    description="Local-first hybrid RAG service",
)
_service: "NexusRAGService | None" = None


class IngestRequest(BaseModel):
    source: str = Field(..., description="File path or directory path")
    overwrite: bool = Field(default=False)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 6
    candidate_k: int = 24
    rerank: bool = True
    generate: bool = True
    workflow: str = "self-rag"
    year_from: int | None = None
    year_to: int | None = None
    file_type: str | None = None
    language: str | None = None
    source_contains: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict[str, int]:
    return _get_service().ingest(source=req.source, overwrite=req.overwrite)


@app.post("/query")
def query(req: QueryRequest) -> dict:
    filters = RetrievalFilters(
        year_from=req.year_from,
        year_to=req.year_to,
        file_type=req.file_type,
        language=req.language,
        source_contains=req.source_contains,
    )
    answer = _get_service().answer(
        query=req.query,
        top_k=req.top_k,
        candidate_k=req.candidate_k,
        use_rerank=req.rerank,
        generate=req.generate,
        workflow=req.workflow,
        filters=filters,
    )
    return {
        "query": answer.query,
        "answer": answer.answer,
        "trace": answer.trace,
        "filters": asdict(filters),
        "contexts": [
            {
                "source_path": c.source_path,
                "header_path": c.header_path,
                "score": c.score,
                "metadata": c.metadata,
                "retrieval_scores": c.retrieval_scores,
                "text": c.text,
            }
            for c in answer.contexts
        ],
    }


@app.post("/query/stream")
def query_stream(req: QueryRequest) -> StreamingResponse:
    filters = RetrievalFilters(
        year_from=req.year_from,
        year_to=req.year_to,
        file_type=req.file_type,
        language=req.language,
        source_contains=req.source_contains,
    )

    def event_stream():
        yield _sse_event(
            "start",
            {
                "query": req.query,
                "workflow": req.workflow,
                "filters": asdict(filters),
            },
        )
        for payload in _get_service().stream_answer(
            query=req.query,
            top_k=req.top_k,
            candidate_k=req.candidate_k,
            use_rerank=req.rerank,
            generate=req.generate,
            workflow=req.workflow,
            filters=filters,
        ):
            yield _sse_event(payload["event"], payload["data"])
        yield _sse_event("end", {"status": "completed"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _get_service() -> "NexusRAGService":
    global _service
    if _service is None:
        from core.service import NexusRAGService

        _service = NexusRAGService(NexusConfig())
    return _service
