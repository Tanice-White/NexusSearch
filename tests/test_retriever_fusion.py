from core.retriever import reciprocal_rank_fusion


def test_rrf_prefers_documents_appearing_in_both_rankings() -> None:
    dense = [("A", 0.9), ("B", 0.8), ("C", 0.7)]
    sparse = [("C", 10.0), ("A", 9.0), ("D", 8.0)]

    fused = reciprocal_rank_fusion(
        ranked_lists=[dense, sparse],
        weights=[0.7, 0.3],
        k=60,
    )
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

    assert ranked[0][0] in {"A", "C"}
    assert "B" in fused
    assert "D" in fused
