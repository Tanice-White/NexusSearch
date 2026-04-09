from core.config import NexusConfig
from core.schemas import RetrievedChunk
from core.workflow import SelfRAGWorkflow


class StubRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(  # noqa: ANN001
        self,
        query: str,
        top_k=None,
        candidate_k=None,
        use_rerank=None,
        filters=None,
    ):
        self.queries.append(query)
        text = "hybrid retrieval with dense and sparse fusion"
        if "rewrite" in query:
            text = "self rag rewrite path recovered relevant hybrid retrieval evidence"
        return [
            RetrievedChunk(
                id="1",
                text=text,
                source_path="README.md",
                header_path="Methods",
                score=0.9,
            )
        ]


class StubGenerator:
    def grade_relevance(self, query: str, contexts):  # noqa: ANN001
        if "rewrite" in query:
            return {"relevant": True, "score": 0.8, "reason": "rewrite_success"}
        return {"relevant": False, "score": 0.1, "reason": "needs_rewrite"}

    def rewrite_query(self, query: str, contexts):  # noqa: ANN001
        return f"{query} rewrite"

    def answer(self, query: str, contexts):  # noqa: ANN001
        return f"answer for {query}"


def test_self_rag_workflow_rewrites_before_generating() -> None:
    config = NexusConfig(prefer_langgraph=False, self_rag_max_loops=2)
    workflow = SelfRAGWorkflow(config, StubRetriever(), StubGenerator())

    result = workflow.run("hybrid retrieval", generate=True)

    assert result.query.endswith("rewrite")
    assert any(step["step"] == "rewrite" for step in result.trace)
    assert result.answer == "answer for hybrid retrieval rewrite"
