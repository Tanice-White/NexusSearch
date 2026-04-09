from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from core.config import NexusConfig


@dataclass(slots=True)
class DesktopSettings:
    db_path: str = "data/lancedb"
    table_name: str = "nexus_chunks"
    source_path: str = "data"
    chunking_strategy: str = "semantic"
    chunk_size: int = 900
    chunk_overlap: int = 120
    semantic_similarity_threshold: float = 0.72
    semantic_min_sentences: int = 2
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    top_k: int = 6
    candidate_k: int = 24
    enable_rerank: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    workflow: str = "self-rag"
    generate: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    self_rag_max_loops: int = 2
    prefer_langgraph: bool = True
    theme_mode: str = "sunset"
    window_width: int = 1480
    window_height: int = 940

    def to_nexus_config(self) -> NexusConfig:
        return NexusConfig(
            db_path=Path(self.db_path),
            table_name=self.table_name,
            chunking_strategy=self.chunking_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            semantic_similarity_threshold=self.semantic_similarity_threshold,
            semantic_min_sentences=self.semantic_min_sentences,
            embedding_model=self.embedding_model,
            embedding_device=self.embedding_device,
            top_k=self.top_k,
            candidate_k=self.candidate_k,
            enable_rerank=self.enable_rerank,
            reranker_model=self.reranker_model,
            ollama_base_url=self.ollama_base_url,
            ollama_model=self.ollama_model,
            self_rag_max_loops=self.self_rag_max_loops,
            prefer_langgraph=self.prefer_langgraph,
        )


class SettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (Path.home() / ".nexussearch" / "desktop_settings.json")

    def load(self) -> DesktopSettings:
        if not self.path.exists():
            settings = DesktopSettings()
            self.save(settings)
            return settings

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        defaults = asdict(DesktopSettings())
        defaults.update(payload)
        return DesktopSettings(**defaults)

    def save(self, settings: DesktopSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(asdict(settings), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
