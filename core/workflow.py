from __future__ import annotations

from typing import TypedDict

from core.config import NexusConfig
from core.llm import OllamaGenerator
from core.retriever import HybridRetriever
from core.schemas import RAGAnswer, RetrievedChunk


class SelfRAGState(TypedDict, total=False):
    original_query: str
    query: str
    loop_count: int
    contexts: list[RetrievedChunk]
    trace: list[dict[str, object]]
    grade: dict[str, object]
    answer: str


class SelfRAGWorkflow:
    def __init__(
        self,
        config: NexusConfig,
        retriever: HybridRetriever,
        generator: OllamaGenerator,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.generator = generator

    def run(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        use_rerank: bool | None = None,
        generate: bool = True,
    ) -> RAGAnswer:
        if self.config.prefer_langgraph:
            try:
                return self._run_with_langgraph(
                    query=query,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    use_rerank=use_rerank,
                    generate=generate,
                )
            except Exception:
                pass
        return self._run_local_loop(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            use_rerank=use_rerank,
            generate=generate,
        )

    def _run_local_loop(
        self,
        query: str,
        top_k: int | None,
        candidate_k: int | None,
        use_rerank: bool | None,
        generate: bool,
    ) -> RAGAnswer:
        current_query = query
        trace: list[dict[str, object]] = []
        contexts: list[RetrievedChunk] = []

        for loop_idx in range(self.config.self_rag_max_loops + 1):
            contexts = self.retriever.retrieve(
                query=current_query,
                top_k=top_k,
                candidate_k=candidate_k,
                use_rerank=use_rerank,
            )
            trace.append(
                {
                    "step": "retrieve",
                    "loop": loop_idx,
                    "query": current_query,
                    "hits": len(contexts),
                }
            )

            grade = self.generator.grade_relevance(current_query, contexts)
            trace.append(
                {
                    "step": "grade",
                    "loop": loop_idx,
                    "query": current_query,
                    "grade": grade,
                }
            )

            if bool(grade.get("relevant")) or not self.config.enable_query_rewrite:
                break

            if loop_idx >= self.config.self_rag_max_loops:
                break

            rewritten = self.generator.rewrite_query(current_query, contexts)
            trace.append(
                {
                    "step": "rewrite",
                    "loop": loop_idx,
                    "from_query": current_query,
                    "to_query": rewritten,
                }
            )
            current_query = rewritten

        if generate:
            try:
                answer = self.generator.answer(current_query, contexts)
            except Exception:
                answer = self._extractive_fallback(current_query, contexts)
        else:
            answer = self._extractive_fallback(current_query, contexts)

        trace.append({"step": "generate", "query": current_query, "used_llm": generate})
        return RAGAnswer(query=current_query, answer=answer, contexts=contexts, trace=trace)

    def _run_with_langgraph(
        self,
        query: str,
        top_k: int | None,
        candidate_k: int | None,
        use_rerank: bool | None,
        generate: bool,
    ) -> RAGAnswer:
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(SelfRAGState)

        def retrieve_node(state: SelfRAGState) -> SelfRAGState:
            contexts = self.retriever.retrieve(
                query=state["query"],
                top_k=top_k,
                candidate_k=candidate_k,
                use_rerank=use_rerank,
            )
            trace = list(state.get("trace", []))
            trace.append(
                {
                    "step": "retrieve",
                    "loop": state["loop_count"],
                    "query": state["query"],
                    "hits": len(contexts),
                }
            )
            return {**state, "contexts": contexts, "trace": trace}

        def grade_node(state: SelfRAGState) -> SelfRAGState:
            grade = self.generator.grade_relevance(state["query"], state.get("contexts", []))
            trace = list(state.get("trace", []))
            trace.append(
                {
                    "step": "grade",
                    "loop": state["loop_count"],
                    "query": state["query"],
                    "grade": grade,
                }
            )
            return {**state, "grade": grade, "trace": trace}

        def rewrite_node(state: SelfRAGState) -> SelfRAGState:
            rewritten = self.generator.rewrite_query(state["query"], state.get("contexts", []))
            trace = list(state.get("trace", []))
            trace.append(
                {
                    "step": "rewrite",
                    "loop": state["loop_count"],
                    "from_query": state["query"],
                    "to_query": rewritten,
                }
            )
            return {
                **state,
                "query": rewritten,
                "loop_count": state["loop_count"] + 1,
                "trace": trace,
            }

        def generate_node(state: SelfRAGState) -> SelfRAGState:
            contexts = state.get("contexts", [])
            if generate:
                try:
                    answer = self.generator.answer(state["query"], contexts)
                except Exception:
                    answer = self._extractive_fallback(state["query"], contexts)
            else:
                answer = self._extractive_fallback(state["query"], contexts)
            trace = list(state.get("trace", []))
            trace.append({"step": "generate", "query": state["query"], "used_llm": generate})
            return {**state, "answer": answer, "trace": trace}

        def route_after_grade(state: SelfRAGState) -> str:
            grade = state.get("grade", {})
            if bool(grade.get("relevant")):
                return "generate"
            if not self.config.enable_query_rewrite:
                return "generate"
            if state["loop_count"] >= self.config.self_rag_max_loops:
                return "generate"
            return "rewrite"

        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("grade", grade_node)
        workflow.add_node("rewrite", rewrite_node)
        workflow.add_node("generate", generate_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade")
        workflow.add_conditional_edges(
            "grade",
            route_after_grade,
            {"rewrite": "rewrite", "generate": "generate"},
        )
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("generate", END)

        app = workflow.compile()
        result = app.invoke(
            {
                "original_query": query,
                "query": query,
                "loop_count": 0,
                "trace": [],
                "contexts": [],
            }
        )
        return RAGAnswer(
            query=str(result.get("query", query)),
            answer=str(result.get("answer", "")),
            contexts=list(result.get("contexts", [])),
            trace=list(result.get("trace", [])),
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
