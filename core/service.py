from __future__ import annotations

from pathlib import Path

from core.chunker import HeaderAwareChunker, SemanticChunker
from core.config import NexusConfig
from core.indexer import NexusIndexer
from core.llm import OllamaGenerator
from core.parser import NexusParser
from core.retriever import HybridRetriever
from core.schemas import DocumentChunk, RAGAnswer, RetrievedChunk
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

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        use_rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            use_rerank=use_rerank,
        )

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        use_rerank: bool | None = None,
        generate: bool = True,
        workflow: str = "self-rag",
    ) -> RAGAnswer:
        if workflow == "self-rag":
            return self.workflow.run(
                query=query,
                top_k=top_k,
                candidate_k=candidate_k,
                use_rerank=use_rerank,
                generate=generate,
            )

        contexts = self.retrieve(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            use_rerank=use_rerank,
        )
        if generate:
            try:
                answer = self.generator.answer(query, contexts)
            except Exception:
                answer = self._extractive_fallback(query, contexts)
        else:
            answer = self._extractive_fallback(query, contexts)
        return RAGAnswer(query=query, answer=answer, contexts=contexts, trace=[])

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
