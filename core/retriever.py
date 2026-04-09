from __future__ import annotations

import json
import re
from dataclasses import dataclass
from math import log
from typing import TYPE_CHECKING, Any

try:
    from rank_bm25 import BM25Okapi
except Exception:
    class BM25Okapi:  # type: ignore[no-redef]
        """Fallback BM25 implementation for dependency-light environments."""

        def __init__(
            self,
            corpus: list[list[str]],
            k1: float = 1.5,
            b: float = 0.75,
        ) -> None:
            self.corpus = corpus
            self.k1 = k1
            self.b = b
            self.doc_len = [len(doc) for doc in corpus]
            self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
            self.doc_freqs: list[dict[str, int]] = []
            self.idf: dict[str, float] = {}
            self._build()

        def _build(self) -> None:
            freq: dict[str, int] = {}
            for doc in self.corpus:
                counts: dict[str, int] = {}
                for token in doc:
                    counts[token] = counts.get(token, 0) + 1
                self.doc_freqs.append(counts)
                for token in counts:
                    freq[token] = freq.get(token, 0) + 1

            n_docs = len(self.corpus)
            for token, df in freq.items():
                self.idf[token] = log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        def get_scores(self, query_tokens: list[str]) -> list[float]:
            scores = [0.0 for _ in self.corpus]
            for i, doc in enumerate(self.corpus):
                freqs = self.doc_freqs[i]
                dl = len(doc)
                for token in query_tokens:
                    if token not in freqs:
                        continue
                    tf = freqs[token]
                    idf = self.idf.get(token, 0.0)
                    denom = tf + self.k1 * (
                        1 - self.b + self.b * dl / max(self.avgdl, 1e-8)
                    )
                    scores[i] += idf * (tf * (self.k1 + 1) / denom)
            return scores

from core.config import NexusConfig
from core.schemas import RetrievedChunk, RetrievalFilters

if TYPE_CHECKING:
    from core.indexer import NexusIndexer

TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    weights: list[float],
    k: int = 60,
) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, (doc_id, _score) in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + (weight / (k + rank + 1))
    return fused


@dataclass(slots=True)
class _SparseCache:
    docs: list[dict[str, Any]]
    tokenized_docs: list[list[str]]
    bm25: BM25Okapi


class _CrossEncoderReranker:
    def __init__(self, model_name: str, enabled: bool) -> None:
        self.enabled = enabled
        self.model = None
        if not enabled:
            return
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
        except Exception:
            self.model = None

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if self.model is None or not chunks:
            return chunks
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)
        for chunk, rerank_score in zip(chunks, scores):
            chunk.retrieval_scores["rerank"] = float(rerank_score)
        return sorted(
            chunks,
            key=lambda x: x.retrieval_scores.get("rerank", float("-inf")),
            reverse=True,
        )


class HybridRetriever:
    def __init__(self, config: NexusConfig, indexer: "NexusIndexer") -> None:
        self.config = config
        self.indexer = indexer
        self._sparse_cache: _SparseCache | None = None
        self.reranker = _CrossEncoderReranker(
            model_name=self.config.reranker_model,
            enabled=self.config.enable_rerank,
        )

    def invalidate_cache(self) -> None:
        self._sparse_cache = None

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        use_rerank: bool | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.config.top_k
        candidate_k = candidate_k or self.config.candidate_k
        use_rerank = self.config.enable_rerank if use_rerank is None else use_rerank

        dense = self._dense_search(query, candidate_k, filters)
        sparse = self._sparse_search(query, candidate_k, filters)
        fused_scores = reciprocal_rank_fusion(
            ranked_lists=[dense, sparse],
            weights=[self.config.dense_weight, self.config.sparse_weight],
            k=self.config.rrf_k,
        )

        doc_map = self._doc_map()
        merged: list[RetrievedChunk] = []
        for doc_id, fused_score in sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        ):
            doc = doc_map.get(doc_id)
            if doc is None:
                continue
            dense_score = next((s for i, s in dense if i == doc_id), 0.0)
            sparse_score = next((s for i, s in sparse if i == doc_id), 0.0)
            merged.append(
                RetrievedChunk(
                    id=doc_id,
                    text=str(doc.get("text", "")),
                    source_path=str(doc.get("source_path", "")),
                    header_path=str(doc.get("header_path", "ROOT")),
                    score=float(fused_score),
                    retrieval_scores={
                        "dense": float(dense_score),
                        "sparse": float(sparse_score),
                        "rrf": float(fused_score),
                    },
                    metadata={
                        **self._decode_metadata(doc.get("metadata_json")),
                        "file_type": str(doc.get("file_type", "")),
                        "language": str(doc.get("language", "")),
                        "chunk_kind": str(doc.get("chunk_kind", "")),
                    },
                )
            )

        merged = merged[:top_k]
        if use_rerank:
            merged = self.reranker.rerank(query, merged)[:top_k]
        return merged

    def _dense_search(
        self,
        query: str,
        k: int,
        filters: RetrievalFilters | None,
    ) -> list[tuple[str, float]]:
        oversample_k = k
        if filters is not None:
            oversample_k = max(k * 10, 50)
        rows = self.indexer.vector_search(query, oversample_k)
        ranked: list[tuple[str, float]] = []
        for row in rows:
            if not self._matches_filters(row, filters):
                continue
            doc_id = str(row.get("id", ""))
            distance = float(row.get("_distance", 1.0))
            dense_score = 1.0 / (1.0 + max(distance, 0.0))
            if doc_id:
                ranked.append((doc_id, dense_score))
            if len(ranked) >= k:
                break
        return ranked

    def _sparse_search(
        self,
        query: str,
        k: int,
        filters: RetrievalFilters | None,
    ) -> list[tuple[str, float]]:
        cache = self._ensure_sparse_cache()
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        scores = cache.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        ranked: list[tuple[str, float]] = []
        for idx in ranked_indices:
            doc = cache.docs[idx]
            if not self._matches_filters(doc, filters):
                continue
            doc_id = str(doc.get("id", ""))
            score = float(scores[idx])
            if doc_id:
                ranked.append((doc_id, score))
            if len(ranked) >= k:
                break
        return ranked

    def _ensure_sparse_cache(self) -> _SparseCache:
        if self._sparse_cache is not None:
            return self._sparse_cache

        docs = self.indexer.all_records()
        tokenized_docs = [tokenize(str(doc.get("text", ""))) for doc in docs]
        bm25 = BM25Okapi(tokenized_docs if tokenized_docs else [["empty"]])
        self._sparse_cache = _SparseCache(
            docs=docs,
            tokenized_docs=tokenized_docs,
            bm25=bm25,
        )
        return self._sparse_cache

    def _doc_map(self) -> dict[str, dict[str, Any]]:
        docs = self._ensure_sparse_cache().docs
        return {str(doc.get("id", "")): doc for doc in docs}

    @staticmethod
    def _decode_metadata(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value:
            try:
                return json.loads(value)
            except Exception:
                return {"raw": value}
        return {}

    @staticmethod
    def _matches_filters(doc: dict[str, Any], filters: RetrievalFilters | None) -> bool:
        if filters is None:
            return True

        updated_year = int(doc.get("updated_year", -1))
        file_type = str(doc.get("file_type", ""))
        language = str(doc.get("language", ""))
        source_path = str(doc.get("source_path", ""))

        if filters.year_from is not None and updated_year < filters.year_from:
            return False
        if filters.year_to is not None and updated_year > filters.year_to:
            return False
        if filters.file_type is not None and file_type != filters.file_type:
            return False
        if filters.language is not None and language != filters.language:
            return False
        if filters.source_contains is not None and filters.source_contains.lower() not in source_path.lower():
            return False
        return True
