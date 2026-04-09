import json
from pathlib import Path

from core.evaluation import RagasEvaluator
from core.schemas import RAGAnswer, RetrievedChunk


class StubService:
    def answer(self, query: str, **kwargs):  # noqa: ANN003, ANN201
        return RAGAnswer(
            query=query,
            answer="RRF combines dense and sparse rankings",
            contexts=[
                RetrievedChunk(
                    id="1",
                    text="RRF combines dense and sparse rankings to align score scales",
                    source_path="README.md",
                    header_path="RRF",
                    score=0.9,
                )
            ],
            trace=[],
        )


def test_evaluator_falls_back_to_heuristics(tmp_path: Path) -> None:
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "query": "How does the system fuse dense and sparse retrieval?",
                "ground_truth": "RRF combines dense and sparse rankings.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    evaluator = RagasEvaluator(StubService())
    report = evaluator.evaluate_file(str(dataset))

    assert "summary" in report
    assert "faithfulness" in report["summary"]
    assert "context_recall" in report["summary"]
    assert report["rows"][0]["faithfulness"] > 0
