from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class NexusConfig:
    db_path: Path = Path(os.getenv("NEXUS_DB_PATH", "data/lancedb"))
    table_name: str = os.getenv("NEXUS_TABLE_NAME", "nexus_chunks")

    chunking_strategy: str = os.getenv("NEXUS_CHUNKING_STRATEGY", "semantic")
    chunk_size: int = int(os.getenv("NEXUS_CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("NEXUS_CHUNK_OVERLAP", "120"))
    semantic_similarity_threshold: float = float(
        os.getenv("NEXUS_SEMANTIC_SIMILARITY_THRESHOLD", "0.72")
    )
    semantic_min_sentences: int = int(os.getenv("NEXUS_SEMANTIC_MIN_SENTENCES", "2"))

    embedding_model: str = os.getenv("NEXUS_EMBED_MODEL", "BAAI/bge-m3")
    embedding_device: str = os.getenv("NEXUS_EMBED_DEVICE", "cpu")

    top_k: int = int(os.getenv("NEXUS_TOP_K", "6"))
    candidate_k: int = int(os.getenv("NEXUS_CANDIDATE_K", "24"))
    rrf_k: int = int(os.getenv("NEXUS_RRF_K", "60"))
    dense_weight: float = float(os.getenv("NEXUS_DENSE_WEIGHT", "0.7"))
    sparse_weight: float = float(os.getenv("NEXUS_SPARSE_WEIGHT", "0.3"))

    enable_rerank: bool = os.getenv("NEXUS_ENABLE_RERANK", "true").lower() == "true"
    reranker_model: str = os.getenv(
        "NEXUS_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"
    )

    ollama_base_url: str = os.getenv("NEXUS_OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("NEXUS_OLLAMA_MODEL", "qwen2.5:7b-instruct")
    ollama_timeout_sec: int = int(os.getenv("NEXUS_OLLAMA_TIMEOUT", "90"))
    self_rag_max_loops: int = int(os.getenv("NEXUS_SELF_RAG_MAX_LOOPS", "2"))
    relevance_threshold: float = float(os.getenv("NEXUS_RELEVANCE_THRESHOLD", "0.12"))
    enable_query_rewrite: bool = (
        os.getenv("NEXUS_ENABLE_QUERY_REWRITE", "true").lower() == "true"
    )
    prefer_langgraph: bool = (
        os.getenv("NEXUS_PREFER_LANGGRAPH", "true").lower() == "true"
    )

    def ensure_dirs(self) -> None:
        self.db_path.mkdir(parents=True, exist_ok=True)
