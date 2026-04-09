from __future__ import annotations

import json
from collections.abc import Iterator

import requests

from core.config import NexusConfig
from core.schemas import RetrievedChunk


class OllamaGenerator:
    def __init__(self, config: NexusConfig) -> None:
        self.config = config

    def answer(self, query: str, contexts: list[RetrievedChunk]) -> str:
        context_block = self._build_context_block(contexts)
        prompt = (
            "You are an enterprise RAG assistant. "
            "Answer only using the provided context. "
            "If the answer is uncertain, explicitly say so.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context_block}\n\n"
            "Output requirement:\n"
            "1) concise answer\n"
            "2) cite supporting snippets with [index]\n"
        )
        return self.chat(prompt)

    def stream_answer(self, query: str, contexts: list[RetrievedChunk]) -> Iterator[str]:
        context_block = self._build_context_block(contexts)
        prompt = (
            "You are an enterprise RAG assistant. "
            "Answer only using the provided context. "
            "If the answer is uncertain, explicitly say so.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context_block}\n\n"
            "Output requirement:\n"
            "1) concise answer\n"
            "2) cite supporting snippets with [index]\n"
        )
        return self.stream_chat(prompt)

    def grade_relevance(self, query: str, contexts: list[RetrievedChunk]) -> dict[str, object]:
        if not contexts:
            return {"relevant": False, "reason": "no_context", "score": 0.0}

        prompt = (
            "You are a retrieval evaluator. "
            "Decide whether the retrieved context is sufficient to answer the question.\n"
            "Return strict JSON with keys: relevant(boolean), score(number 0-1), reason(string).\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{self._build_context_block(contexts[:3])}"
        )
        try:
            raw = self.chat(prompt)
            payload = json.loads(self._extract_json(raw))
            return {
                "relevant": bool(payload.get("relevant", False)),
                "score": float(payload.get("score", 0.0)),
                "reason": str(payload.get("reason", "llm_grade")),
            }
        except Exception:
            lexical_overlap = self._lexical_overlap(query, contexts)
            return {
                "relevant": lexical_overlap >= self.config.relevance_threshold,
                "score": lexical_overlap,
                "reason": "heuristic_overlap",
            }

    def rewrite_query(self, query: str, contexts: list[RetrievedChunk]) -> str:
        prompt = (
            "Rewrite the question for retrieval. "
            "Keep it concise, expand missing entities, preserve original intent, and avoid answering it.\n"
            "Return only the rewritten query.\n\n"
            f"Original question:\n{query}\n\n"
            f"Observed context weakness:\n{self._build_context_block(contexts[:2])}"
        )
        try:
            rewritten = self.chat(prompt).strip()
            return rewritten if rewritten else query
        except Exception:
            if not contexts:
                return query
            keywords = " ".join(
                filter(None, [contexts[0].header_path, contexts[0].source_path.split("\\")[-1]])
            )
            return f"{query} {keywords}".strip()

    def chat(self, prompt: str) -> str:
        url = f"{self.config.ollama_base_url.rstrip('/')}/api/chat"
        response = requests.post(
            url,
            json={
                "model": self.config.ollama_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=self.config.ollama_timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("message", {}).get("content", "").strip()

    def stream_chat(self, prompt: str) -> Iterator[str]:
        url = f"{self.config.ollama_base_url.rstrip('/')}/api/chat"
        with requests.post(
            url,
            json={
                "model": self.config.ollama_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            timeout=self.config.ollama_timeout_sec,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                payload = json.loads(line)
                content = payload.get("message", {}).get("content", "")
                if content:
                    yield content

    @staticmethod
    def _extract_json(raw: str) -> str:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in model output.")
        return raw[start : end + 1]

    @staticmethod
    def _build_context_block(contexts: list[RetrievedChunk]) -> str:
        return "\n\n".join(
            [
                (
                    f"[{idx}] source={chunk.source_path} | header={chunk.header_path}\n"
                    f"{chunk.text}"
                )
                for idx, chunk in enumerate(contexts, start=1)
            ]
        )

    @staticmethod
    def _lexical_overlap(query: str, contexts: list[RetrievedChunk]) -> float:
        query_terms = {term for term in query.lower().split() if term}
        if not query_terms:
            return 0.0
        context_terms: set[str] = set()
        for chunk in contexts[:3]:
            context_terms.update(chunk.text.lower().split())
        return len(query_terms & context_terms) / len(query_terms)
