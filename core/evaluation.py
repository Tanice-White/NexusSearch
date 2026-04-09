from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

from core.schemas import RetrievalFilters

if TYPE_CHECKING:
    from core.service import NexusRAGService

TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")


@dataclass(slots=True)
class EvaluationSample:
    query: str
    ground_truth: str
    workflow: str = "self-rag"
    top_k: int = 6
    candidate_k: int = 24
    rerank: bool = True
    generate: bool = True
    year_from: int | None = None
    year_to: int | None = None
    file_type: str | None = None
    language: str | None = None
    source_contains: str | None = None

    def filters(self) -> RetrievalFilters:
        return RetrievalFilters(
            year_from=self.year_from,
            year_to=self.year_to,
            file_type=self.file_type,
            language=self.language,
            source_contains=self.source_contains,
        )


class RagasEvaluator:
    def __init__(self, service: "NexusRAGService") -> None:
        self.service = service

    def evaluate_file(
        self,
        dataset_path: str,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        path = Path(dataset_path)
        samples = self._load_samples(path)
        rows = [self._run_sample(sample) for sample in samples]

        try:
            metrics = self._score_with_ragas(rows)
            backend = "ragas"
        except Exception as exc:
            metrics = self._score_with_heuristics(rows)
            backend = f"heuristic_fallback:{exc.__class__.__name__}"

        result = {
            "dataset_path": str(path),
            "backend": backend,
            "summary": metrics["summary"],
            "rows": metrics["rows"],
        }
        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def _load_samples(self, path: Path) -> list[EvaluationSample]:
        samples: list[EvaluationSample] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            samples.append(EvaluationSample(**payload))
        if not samples:
            raise ValueError(f"No evaluation samples found in {path}")
        return samples

    def _run_sample(self, sample: EvaluationSample) -> dict[str, Any]:
        answer = self.service.answer(
            query=sample.query,
            top_k=sample.top_k,
            candidate_k=sample.candidate_k,
            use_rerank=sample.rerank,
            generate=sample.generate,
            workflow=sample.workflow,
            filters=sample.filters(),
        )
        return {
            "query": sample.query,
            "ground_truth": sample.ground_truth,
            "answer": answer.answer,
            "contexts": [chunk.text for chunk in answer.contexts],
            "trace": answer.trace,
            "filters": asdict(sample.filters()),
        }

    def _score_with_ragas(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import context_recall, faithfulness

        dataset = Dataset.from_list(
            [
                {
                    "question": row["query"],
                    "answer": row["answer"],
                    "contexts": row["contexts"],
                    "ground_truth": row["ground_truth"],
                }
                for row in rows
            ]
        )
        result = evaluate(dataset=dataset, metrics=[faithfulness, context_recall])

        if hasattr(result, "to_pandas"):
            frame = result.to_pandas()
            metric_rows = frame.to_dict(orient="records")
            summary = {
                "faithfulness": float(frame["faithfulness"].mean()),
                "context_recall": float(frame["context_recall"].mean()),
            }
        else:
            metric_rows = []
            summary = {
                "faithfulness": float(getattr(result, "faithfulness", 0.0)),
                "context_recall": float(getattr(result, "context_recall", 0.0)),
            }

        merged_rows = []
        for idx, row in enumerate(rows):
            merged = dict(row)
            if idx < len(metric_rows):
                merged["faithfulness"] = float(metric_rows[idx].get("faithfulness", 0.0))
                merged["context_recall"] = float(metric_rows[idx].get("context_recall", 0.0))
            merged_rows.append(merged)

        return {"summary": summary, "rows": merged_rows}

    def _score_with_heuristics(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        scored_rows: list[dict[str, Any]] = []
        for row in rows:
            faithfulness_score = self._faithfulness(row["answer"], row["contexts"])
            context_recall_score = self._context_recall(row["ground_truth"], row["contexts"])
            scored = dict(row)
            scored["faithfulness"] = faithfulness_score
            scored["context_recall"] = context_recall_score
            scored_rows.append(scored)

        summary = {
            "faithfulness": mean(row["faithfulness"] for row in scored_rows),
            "context_recall": mean(row["context_recall"] for row in scored_rows),
        }
        return {"summary": summary, "rows": scored_rows}

    @staticmethod
    def _faithfulness(answer: str, contexts: list[str]) -> float:
        answer_terms = set(TOKEN_RE.findall(answer.lower()))
        if not answer_terms:
            return 0.0
        context_terms = set(TOKEN_RE.findall(" ".join(contexts).lower()))
        return len(answer_terms & context_terms) / len(answer_terms)

    @staticmethod
    def _context_recall(ground_truth: str, contexts: list[str]) -> float:
        truth_terms = set(TOKEN_RE.findall(ground_truth.lower()))
        if not truth_terms:
            return 0.0
        context_terms = set(TOKEN_RE.findall(" ".join(contexts).lower()))
        return len(truth_terms & context_terms) / len(truth_terms)
