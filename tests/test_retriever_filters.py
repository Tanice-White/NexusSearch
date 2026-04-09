from core.config import NexusConfig
from core.retriever import HybridRetriever
from core.schemas import RetrievalFilters


class StubIndexer:
    def vector_search(self, query: str, k: int):  # noqa: ANN001
        return [
            {
                "id": "old",
                "_distance": 0.1,
                "text": "legacy result",
                "source_path": "docs/2024.md",
                "header_path": "Legacy",
                "updated_year": 2024,
                "file_type": "document",
                "language": "",
                "chunk_kind": "section",
                "metadata_json": "{}",
            },
            {
                "id": "new",
                "_distance": 0.2,
                "text": "fresh result",
                "source_path": "docs/2025.md",
                "header_path": "Fresh",
                "updated_year": 2025,
                "file_type": "document",
                "language": "",
                "chunk_kind": "section",
                "metadata_json": "{}",
            },
        ]

    def all_records(self):
        return self.vector_search("", 10)


def test_retriever_applies_year_filter() -> None:
    retriever = HybridRetriever(NexusConfig(enable_rerank=False), StubIndexer())

    results = retriever.retrieve(
        query="result",
        top_k=5,
        candidate_k=5,
        use_rerank=False,
        filters=RetrievalFilters(year_from=2025),
    )

    assert len(results) == 1
    assert results[0].id == "new"
