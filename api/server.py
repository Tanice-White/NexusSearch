from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from core.config import NexusConfig
from core.service import NexusRAGService

app = FastAPI(
    title="NexusSearch API",
    version="1.0.0",
    description="Local-first hybrid RAG service",
)
service = NexusRAGService(NexusConfig())


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict[str, int]:
    return service.ingest(source=req.source, overwrite=req.overwrite)


@app.post("/query")
def query(req: QueryRequest) -> dict:
    answer = service.answer(
        query=req.query,
        top_k=req.top_k,
        candidate_k=req.candidate_k,
        use_rerank=req.rerank,
        generate=req.generate,
        workflow=req.workflow,
    )
    return {
        "query": answer.query,
        "answer": answer.answer,
        "trace": answer.trace,
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
