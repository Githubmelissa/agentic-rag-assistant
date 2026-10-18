import json
from pathlib import Path

import pytest

from assistant.agent.graph import Agent, AgentConfig, parse_route, parse_split
from assistant.agent.prompts import OUT_OF_SCOPE_ANSWER
from assistant.ingest.chunking import Chunk
from assistant.llm import LLMResponse, MockLLM
from assistant.retrieval.bm25 import BM25Index
from assistant.retrieval.dense import DenseIndex
from assistant.retrieval.retriever import Retriever
from tests.conftest import FakeEmbedder


class RoleLLM:
    """Fake LLM that answers according to which prompt it receives."""

    model = "role-fake"

    def __init__(self, route: str) -> None:
        self.route = route
        self.calls: list[str] = []

    def complete(self, system: str, user: str) -> LLMResponse:
        if "Classify a user question" in system:
            kind, text = "route", f"Category: {self.route}"
        elif "Split it into two" in system:
            kind, text = "split", "REPORTS: What was Kering's revenue?\nSALES: How many lines?"
        elif "SQLite SELECT query" in system:
            kind, text = "sql", "```sql\nSELECT COUNT(*) AS n FROM sales_clean\n```"
        elif "explain the result of a database query" in system:
            kind, text = "sql_answer", "There are 3 sales lines."
        else:
            kind, text = "doc", "Revenue was 14,675 million euros [kering_urd p.47]."
        self.calls.append(kind)
        return LLMResponse(text, self.model, 10, 5, 0.01)


class NoRerank:
    def score(self, query: str, texts: list[str]) -> list[float]:
        return [0.0] * len(texts)


CHUNKS = [
    Chunk("k:p47:c0", "kering_urd", 47, "45", "Group revenue was 14,675 million euros.", 6),
    Chunk("h:p10:c0", "hermes_urd", 10, "", "Hermes employs many craftspeople.", 4),
    Chunk("l:p5:c0", "lvmh_ar", 5, "", "LVMH opened stores in Asia.", 5),
]


@pytest.fixture
def make_agent(tmp_path: Path, fake_embedder: FakeEmbedder, sales_db: Path):
    def make(llm, log: bool = False) -> Agent:
        dense = DenseIndex.build(CHUNKS, tmp_path / "idx", fake_embedder)
        retriever = Retriever(CHUNKS, dense, BM25Index(CHUNKS), NoRerank())
        config = AgentConfig(retrieval_mode="dense", top_k=2)
        return Agent(retriever, llm, sales_db, config, log_dir=tmp_path / "logs" if log else None)

    return make


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("documents", "documents"),
        ("Category: sales_data", "sales_data"),
        ("Sales data", "sales_data"),
        ("out-of-scope.", "out_of_scope"),
        ("BOTH", "both"),
        ("I think: documents (not sales_data)", "documents"),  # first label wins
        ("no idea", None),
    ],
)
def test_parse_route(reply: str, expected: str | None) -> None:
    assert parse_route(reply) == expected


def nodes(result) -> list[str]:
    return [step["node"] for step in result.trace]


def test_documents_route(make_agent) -> None:
    llm = RoleLLM("documents")
    r = make_agent(llm).ask("What was Kering's revenue?")
    assert r.route == "documents"
    assert nodes(r) == ["route", "doc_rag", "synthesize"]
    assert "[kering_urd p.47]" in r.answer
    assert [(c.source, c.page) for c in r.citations] == [("kering_urd", 47)]
    assert r.sql is None
    assert llm.calls == ["route", "doc"]  # no extra LLM call to synthesize


def test_sales_route(make_agent) -> None:
    llm = RoleLLM("sales_data")
    r = make_agent(llm).ask("How many sales lines?")
    assert nodes(r) == ["route", "sql_agent", "synthesize"]
    assert r.sql is not None and r.sql.startswith("SELECT COUNT(*)")
    assert r.rows == [(3,)]
    assert r.answer == "There are 3 sales lines."


def test_both_route(make_agent) -> None:
    llm = RoleLLM("both")
    r = make_agent(llm).ask("Compare Kering revenue with the shop.")
    assert nodes(r) == ["route", "split", "doc_rag", "sql_agent", "synthesize"]
    assert llm.calls == ["route", "split", "doc", "sql", "sql_answer"]  # merge uses no LLM
    assert "From the company reports: Revenue was 14,675 million euros" in r.answer
    assert "From the gift-shop sales data: There are 3 sales lines." in r.answer
    split_step = r.trace[1]
    assert split_step["doc_question"] == "What was Kering's revenue?"
    assert split_step["sql_question"] == "How many lines?"
    assert r.citations and r.citations[0].page == 47
    assert r.rows == [(3,)]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("REPORTS: What was X?\nSALES: How many Y?", ("What was X?", "How many Y?")),
        ("reports : a?\n  sales: b?", ("a?", "b?")),
        ("Here you go:\nREPORTS: a?", ("a?", None)),
        ("nonsense", (None, None)),
    ],
)
def test_parse_split(reply: str, expected: tuple) -> None:
    assert parse_split(reply) == expected


def test_failed_split_falls_back_to_full_question(make_agent) -> None:
    class BadSplit(RoleLLM):
        def complete(self, system: str, user: str) -> LLMResponse:
            if "Split it into two" in system:
                return LLMResponse("sorry", self.model, 1, 1, 0.0)
            return super().complete(system, user)

    r = make_agent(BadSplit("both")).ask("Compare things")
    assert r.trace[1]["parsed_ok"] is False
    assert r.trace[1]["doc_question"] == "Compare things"


def test_out_of_scope_route_never_calls_tools(make_agent) -> None:
    llm = RoleLLM("out_of_scope")
    r = make_agent(llm).ask("What is the capital of Australia?")
    assert nodes(r) == ["route", "synthesize"]
    assert r.answer == OUT_OF_SCOPE_ANSWER
    assert llm.calls == ["route"]


def test_unparseable_route_falls_back_to_both(make_agent) -> None:
    r = make_agent(RoleLLM("???")).ask("Something")
    assert r.route == "both"
    assert r.trace[0]["parsed_ok"] is False


def test_trace_written_as_json(make_agent) -> None:
    r = make_agent(RoleLLM("sales_data"), log=True).ask("How many sales lines?")
    assert r.trace_path is not None and r.trace_path.exists()
    data = json.loads(r.trace_path.read_text(encoding="utf-8"))
    assert data["trace_id"] == r.trace_id
    assert data["route"] == "sales_data"
    assert [s["node"] for s in data["steps"]] == ["route", "sql_agent", "synthesize"]
    sql_step = data["steps"][1]
    assert sql_step["attempts"][0]["error"] is None
    assert sql_step["input_tokens"] == 20  # two LLM calls of 10 tokens each
    assert all("latency_s" in s for s in data["steps"])


def test_mock_llm_runs_the_whole_graph(make_agent) -> None:
    r = make_agent(MockLLM()).ask("What was Kering's revenue?")
    assert r.route == "documents"
    assert r.answer
