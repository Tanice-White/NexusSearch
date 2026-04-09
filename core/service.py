from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from core.chunker import HeaderAwareChunker, SemanticChunker
from core.config import NexusConfig
from core.indexer import NexusIndexer
from core.llm import OllamaGenerator
from core.parser import NexusParser
from core.retriever import HybridRetriever
from core.schemas import DocumentChunk, RAGAnswer, RetrievedChunk, RetrievalFilters
from core.workflow import SelfRAGWorkflow


class NexusRAGService:
    def __init__(self, config: NexusConfig | None = None) -> None:
        self.config = config or NexusConfig()
        self.parser = NexusParser()
        self.indexer = NexusIndexer(self.config)
        self.retriever = HybridRetriever(self.config, self.indexer)
        self.generator = OllamaGenerator(self.config)
        self.workflow = SelfRAGWorkflow(self.config, self.retriever, self.generator)
        self.chunker = self._build_chunker()

    def ingest(self, source: str, overwrite: bool = False) -> dict[str, int]:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source}")

        files = list(self.parser.iter_files(source_path))
        all_chunks: list[DocumentChunk] = []
        for file in files:
            document = self.parser.parse(file)
            chunks = self.chunker.chunk_document(document)
            all_chunks.extend(chunks)

        indexed_count = self.indexer.index_chunks(all_chunks, overwrite=overwrite)
        self.retriever.invalidate_cache()
        return {
            "files": len(files),
            "chunks": len(all_chunks),
            "indexed": indexed_count,
            "chunking_strategy": self.config.chunking_strategy,
        }

    def ingest_stream(
        self,
        source: str,
        overwrite: bool = False,
    ) -> Iterator[dict[str, Any]]:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source}")

        files = list(self.parser.iter_files(source_path))
        total_files = len(files)
        yield {
            "event": "start",
            "data": {
                "source": str(source_path),
                "total_files": total_files,
                "overwrite": overwrite,
            },
        }

        all_chunks: list[DocumentChunk] = []
        for index, file in enumerate(files, start=1):
            document = self.parser.parse(file)
            chunks = self.chunker.chunk_document(document)
            all_chunks.extend(chunks)
            yield {
                "event": "progress",
                "data": {
                    "current": index,
                    "total": total_files,
                    "file_path": str(file),
                    "chunks_in_file": len(chunks),
                    "chunks_accumulated": len(all_chunks),
                },
            }

        indexed_count = self.indexer.index_chunks(all_chunks, overwrite=overwrite)
        self.retriever.invalidate_cache()
        summary = {
            "files": total_files,
            "chunks": len(all_chunks),
            "indexed": indexed_count,
            "chunking_strategy": self.config.chunking_strategy,
        }
        yield {"event": "final", "data": summary}

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        use_rerank: bool | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            use_rerank=use_rerank,
            filters=filters,
        )

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        use_rerank: bool | None = None,
        generate: bool = True,
        workflow: str = "self-rag",
        filters: RetrievalFilters | None = None,
    ) -> RAGAnswer:
        if workflow == "self-rag":
            return self.workflow.run(
                query=query,
                top_k=top_k,
                candidate_k=candidate_k,
                use_rerank=use_rerank,
                filters=filters,
                generate=generate,
            )

        contexts = self.retrieve(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            use_rerank=use_rerank,
            filters=filters,
        )
        if generate:
            try:
                answer = self.generator.answer(query, contexts)
            except Exception:
                answer = self._extractive_fallback(query, contexts)
        else:
            answer = self._extractive_fallback(query, contexts)
        return RAGAnswer(query=query, answer=answer, contexts=contexts, trace=[])

    def stream_answer(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        use_rerank: bool | None = None,
        generate: bool = True,
        workflow: str = "self-rag",
        filters: RetrievalFilters | None = None,
    ) -> Iterator[dict[str, Any]]:
        prepared = self.answer(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            use_rerank=use_rerank,
            generate=False,
            workflow=workflow,
            filters=filters,
        )
        yield {"event": "retrieval", "data": {"query": prepared.query, "trace": prepared.trace}}
        for chunk in prepared.contexts:
            yield {
                "event": "context",
                "data": self._serialize_context(chunk),
            }

        if not generate:
            final_answer = self._extractive_fallback(prepared.query, prepared.contexts)
            yield {"event": "final", "data": self._serialize_answer(prepared, final_answer)}
            return

        pieces: list[str] = []
        try:
            for piece in self.generator.stream_answer(prepared.query, prepared.contexts):
                pieces.append(piece)
                yield {"event": "token", "data": {"text": piece}}
            final_answer = "".join(pieces).strip()
            if not final_answer:
                final_answer = self._extractive_fallback(prepared.query, prepared.contexts)
        except Exception:
            final_answer = self._extractive_fallback(prepared.query, prepared.contexts)
            yield {"event": "token", "data": {"text": final_answer}}

        yield {"event": "final", "data": self._serialize_answer(prepared, final_answer)}

    def _build_chunker(self) -> HeaderAwareChunker | SemanticChunker:
        if self.config.chunking_strategy == "semantic":
            return SemanticChunker(
                embed_texts=self.indexer.embed_texts,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                similarity_threshold=self.config.semantic_similarity_threshold,
                min_sentences=self.config.semantic_min_sentences,
            )
        return HeaderAwareChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    @staticmethod
    def _extractive_fallback(query: str, contexts: list[RetrievedChunk]) -> str:
        if not contexts:
            return f"No relevant context was retrieved for query: {query}"
        top = contexts[0]
        return (
            "Top retrieved context (generation model disabled or unavailable):\n"
            f"[1] {top.source_path} | {top.header_path}\n{top.text}"
        )

    @staticmethod
    def _serialize_context(chunk: RetrievedChunk) -> dict[str, Any]:
        return {
            "source_path": chunk.source_path,
            "header_path": chunk.header_path,
            "score": chunk.score,
            "metadata": chunk.metadata,
            "retrieval_scores": chunk.retrieval_scores,
            "text": chunk.text,
        }

    def _serialize_answer(self, prepared: RAGAnswer, answer_text: str) -> dict[str, Any]:
        return {
            "query": prepared.query,
            "answer": answer_text,
            "trace": prepared.trace,
            "contexts": [self._serialize_context(chunk) for chunk in prepared.contexts],
        }
