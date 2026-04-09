import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import api.server as server


class StubStreamService:
    def stream_answer(self, **kwargs):  # noqa: ANN003, ANN201
        yield {"event": "retrieval", "data": {"query": kwargs["query"], "trace": []}}
        yield {"event": "token", "data": {"text": "hello"}}
        yield {
            "event": "final",
            "data": {"query": kwargs["query"], "answer": "hello", "trace": [], "contexts": []},
        }


def test_query_stream_returns_sse_events(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_service", lambda: StubStreamService())
    client = TestClient(server.app)

    response = client.post(
        "/query/stream",
        json={
            "query": "stream this",
            "workflow": "self-rag",
            "generate": True,
        },
    )

    body = response.text
    assert response.status_code == 200
    assert "event: start" in body
    assert "event: retrieval" in body
    assert "event: token" in body
    assert "event: final" in body
    assert "event: end" in body
