"""API tests. A fake agent is injected, so no model is loaded and no LLM is called."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from assistant.agent.graph import AgentResult
from assistant.api import create_app
from assistant.llm import LLMError
from assistant.tools.doc_search import Citation
from tests.conftest import make_settings

RESULT = AgentResult(
    trace_id="abc123",
    question="What was Kering's revenue in 2025?",
    route="documents",
    answer="Kering's revenue in 2025 was EUR 14,675 million. [kering_urd_2025 p.13]",
    citations=[Citation("kering_urd_2025", 13, "11", "Kering in 2025 EUR 14,675 million")],
    sql=None,
    columns=[],
    rows=[],
    trace=[{"node": "route", "latency_s": 1.5}, {"node": "doc_rag", "latency_s": 12.0}],
    total_latency_s=13.567,
)


class FakeAgent:
    """Stands in for the real agent: same `ask`, no models."""

    retriever = None

    def __init__(self, result: Any = RESULT, error: Exception | None = None) -> None:
        self.result, self.error, self.asked = result, error, []

    def ask(self, question: str) -> Any:
        self.asked.append(question)
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def client(tmp_path):
    settings = make_settings(llm_provider="mock", data_dir=tmp_path, index_dir=tmp_path)
    with TestClient(create_app(settings, FakeAgent())) as c:
        yield c


def test_ask_returns_the_answer_and_its_citations(client: TestClient) -> None:
    r = client.post("/ask", json={"question": "What was Kering's revenue in 2025?"})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "documents"
    assert body["answer"].startswith("Kering's revenue")
    assert body["citations"] == [
        {
            "source": "kering_urd_2025",
            "page": 13,
            "page_label": "11",
            "snippet": "Kering in 2025 EUR 14,675 million",
        }
    ]
    assert body["sql"] is None and body["row_count"] == 0
    assert body["steps"][0] == {"node": "route", "latency_s": 1.5}
    assert body["latency_s"] == 13.57 and body["trace_id"] == "abc123"


@pytest.mark.parametrize("question", ["", "  ", "   ", " hi ", "x" * 501])
def test_a_question_that_is_empty_or_too_long_is_refused(client: TestClient, question) -> None:
    assert client.post("/ask", json={"question": question}).status_code == 422


def test_unknown_fields_are_refused(client: TestClient) -> None:
    body = {"question": "What was Kering's revenue in 2025?", "mode": "dense"}
    assert client.post("/ask", json=body).status_code == 422


def test_health_is_ok_and_calls_no_llm(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["detail"] is None
    assert body["llm"] == "mock" and body["sales_db"] is False
    assert body["uptime_s"] >= 0


def test_llm_errors_become_503(tmp_path) -> None:
    agent = FakeAgent(error=LLMError("connection refused"))
    settings = make_settings(llm_provider="ollama", llm_model="qwen2.5:3b", data_dir=tmp_path)
    with TestClient(create_app(settings, agent)) as c:
        r = c.post("/ask", json={"question": "What was Kering's revenue in 2025?"})
    assert r.status_code == 503 and "connection refused" in r.json()["detail"]


def test_a_broken_setup_does_not_kill_the_app(tmp_path) -> None:
    """No index on disk: the app starts, /health explains, /ask answers 503."""
    settings = make_settings(llm_provider="mock", data_dir=tmp_path, index_dir=tmp_path / "missing")
    with TestClient(create_app(settings)) as c:
        health = c.get("/health").json()
        assert health["status"] == "not ready" and health["detail"]
        r = c.post("/ask", json={"question": "What was Kering's revenue in 2025?"})
    assert r.status_code == 503


def test_a_second_question_while_one_is_running_gets_429(tmp_path, monkeypatch) -> None:
    import assistant.api as api

    monkeypatch.setattr(api, "BUSY_WAIT_SECONDS", 0.05)
    started = threading.Event()

    class SlowAgent(FakeAgent):
        def ask(self, question: str) -> Any:
            started.set()
            time.sleep(0.3)
            return RESULT

    settings = make_settings(llm_provider="mock", data_dir=tmp_path)
    with TestClient(create_app(settings, SlowAgent())) as c:
        body = {"question": "What was Kering's revenue in 2025?"}
        codes: list[int] = []
        first = threading.Thread(target=lambda: codes.append(c.post("/ask", json=body).status_code))
        first.start()
        assert started.wait(2)
        codes.append(c.post("/ask", json=body).status_code)
        first.join()
    assert sorted(codes) == [200, 429]
