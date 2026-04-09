from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ParsedSection:
    title: str
    text: str
    kind: str = "section"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    source_path: str
    content: str
    file_type: str
    language: str | None = None
    parser_name: str = "plain"
    sections: list[ParsedSection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentChunk:
    id: str
    text: str
    source_path: str
    header_path: str
    section_index: int
    chunk_index: int
    token_count: int
    file_type: str = "document"
    language: str | None = None
    chunk_kind: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    id: str
    text: str
    source_path: str
    header_path: str
    score: float
    retrieval_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RAGAnswer:
    query: str
    answer: str
    contexts: list[RetrievedChunk]
    trace: list[dict[str, Any]] = field(default_factory=list)
