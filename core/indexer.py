from __future__ import annotations

import json
from typing import Any

import lancedb
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from core.config import NexusConfig
from core.schemas import DocumentChunk


class _FallbackEmbedder:
    """Dependency-light fallback if sentence-transformers cannot be loaded."""

    def __init__(self, dim: int = 768) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32)


class NexusIndexer:
    def __init__(self, config: NexusConfig) -> None:
        self.config = config
        self.config.ensure_dirs()
        self.db = lancedb.connect(str(self.config.db_path))
        self.embedder = self._init_embedder()

    def _init_embedder(self):  # noqa: ANN001
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(
                self.config.embedding_model, device=self.config.embedding_device
            )
        except Exception:
            return _FallbackEmbedder()

    def index_chunks(self, chunks: list[DocumentChunk], overwrite: bool = False) -> int:
        if not chunks:
            return 0

        embeddings = self.embed_texts([chunk.text for chunk in chunks])
        records = []
        for chunk, embedding in zip(chunks, embeddings):
            updated_year = chunk.metadata.get("updated_year")
            updated_at = chunk.metadata.get("updated_at")
            contains_table = bool(chunk.metadata.get("contains_table", False))
            records.append(
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "source_path": chunk.source_path,
                    "header_path": chunk.header_path,
                    "section_index": chunk.section_index,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "file_type": chunk.file_type,
                    "language": chunk.language or "",
                    "chunk_kind": chunk.chunk_kind,
                    "updated_year": int(updated_year) if updated_year is not None else -1,
                    "updated_at": str(updated_at or ""),
                    "contains_table": contains_table,
                    "metadata_json": json.dumps(chunk.metadata, ensure_ascii=False),
                    "vector": embedding,
                }
            )

        if overwrite or not self._table_exists():
            table = self.db.create_table(
                self.config.table_name,
                data=records,
                mode="overwrite",
            )
        else:
            table = self.db.open_table(self.config.table_name)
            table.add(records)

        self._ensure_fts_index(table)
        return len(records)

    def vector_search(self, query: str, k: int) -> list[dict[str, Any]]:
        table = self._open_table()
        query_vector = self.embed_texts([query])[0]
        return table.search(query_vector).limit(k).to_list()

    def all_records(self) -> list[dict[str, Any]]:
        table = self._open_table()
        dataframe = table.to_pandas()
        return dataframe.to_dict(orient="records")

    def _open_table(self):
        if not self._table_exists():
            raise RuntimeError(
                f"Table `{self.config.table_name}` does not exist. Run ingest first."
            )
        return self.db.open_table(self.config.table_name)

    def _table_exists(self) -> bool:
        return self.config.table_name in self.db.table_names()

    def _ensure_fts_index(self, table) -> None:  # noqa: ANN001
        try:
            table.create_fts_index("text", replace=True)
        except TypeError:
            try:
                table.create_fts_index("text")
            except Exception:
                pass
        except Exception:
            pass

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self.embedder.encode(texts)
        if hasattr(vectors, "tolist"):
            return vectors.tolist()
        return [list(row) for row in vectors]
