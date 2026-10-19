"""The router agent, built with LangGraph.

    START -> route --documents----> doc_rag ------------------------> synthesize -> END
                   --sales_data---> sql_agent ----------------------> synthesize
                   --both---------> split -> doc_rag -> sql_agent --> synthesize
                                          (either tool may be skipped)
                   --out_of_scope----------------------------------> synthesize

- `route` asks the LLM for one of four labels (documents, sales_data, both, out_of_scope).
- `split` (only for "both") rewrites the question as one sub-question per source, so each
  tool receives a question it can fully answer (query decomposition). A sub-question may
  be NONE (that source is not needed). A sub-question that is not a question, or that
  mentions a number or a company absent from the user's question, is dropped: the 3B model
  sometimes comments instead of splitting, or copies the example in its prompt. When the
  reports sub-question is dropped, `doc_rag` runs on the user's own question, because it
  can answer "not found" while the SQL tool always returns some number.
- `doc_rag` and `sql_agent` call the Stage 3 and Stage 4 tools.
- `synthesize` returns the tool's answer as is when only one tool ran, places the two
  answers side by side for "both" (a fixed template: an LLM merge once corrupted a correct
  number), and returns a fixed polite refusal for "out_of_scope". It never calls the LLM.
Every node appends a step (timing, inputs, outputs, token counts) to the trace, which is
written as JSON to logs/ at the end of each run.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from assistant.agent.prompts import (
    BOTH_TEMPLATE,
    OUT_OF_SCOPE_ANSWER,
    ROUTER_SYSTEM,
    ROUTER_USER,
    ROUTES,
    SPLIT_SYSTEM,
    SPLIT_USER,
)
from assistant.llm import LLM, LLMResponse
from assistant.retrieval.retriever import Retriever
from assistant.tools.doc_search import Citation, DocAnswer, answer_question
from assistant.tools.sql_tool import SQLAnswer, answer_sql_question

FALLBACK_ROUTE = "both"  # used when the router's reply contains no valid label
TRACE_ROWS = 5  # result rows copied into the trace


class AgentState(TypedDict, total=False):
    question: str
    route: str
    doc_question: str  # sub-question for the reports (set by `split`, else = question)
    sql_question: str  # sub-question for the sales data
    doc: DocAnswer
    sql: SQLAnswer
    answer: str
    citations: list[Citation]
    trace: list[dict[str, Any]]


@dataclass
class AgentConfig:
    retrieval_mode: str = "hybrid_rerank"
    top_k: int = 5
    sql_max_retries: int = 2
    sql_retry_on_empty: bool = False
    sql_max_rows: int = 100
    sql_timeout_s: float = 15.0


@dataclass
class AgentResult:
    trace_id: str
    question: str
    route: str
    answer: str
    citations: list[Citation]
    sql: str | None
    columns: list[str]
    rows: list[tuple[object, ...]]
    trace: list[dict[str, Any]] = field(repr=False)
    total_latency_s: float = 0.0
    trace_path: Path | None = None


def parse_route(text: str) -> str | None:
    """Return the first valid label in the LLM reply ("sales_data", "Sales data", ...)."""
    normalized = re.sub(r"[\s\-]+", "_", text.strip().lower())
    found = [(normalized.find(r), r) for r in ROUTES if r in normalized]
    return min(found)[1] if found else None


_SPLIT_LINE = re.compile(r"^\s*(REPORTS|SALES)\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_NUMBER = re.compile(r"\d+")
# Companies named in the three reports. Fixed strings, so this check has no false alarms.
COMPANY_NAMES = (
    "kering",
    "hermes",
    "lvmh",
    "gucci",
    "yves saint laurent",
    "saint laurent",
    "bottega veneta",
    "balenciaga",
    "alexander mcqueen",
    "boucheron",
    "pomellato",
    "qeelin",
    "ginori",
    "louis vuitton",
    "christian dior",
    "dior",
    "sephora",
    "tiffany",
    "bulgari",
    "loro piana",
    "fendi",
    "celine",
    "loewe",
    "moet",
    "hennessy",
    "veuve clicquot",
)


def parse_split(text: str) -> tuple[str | None, str | None]:
    """Read the "REPORTS: ..." and "SALES: ..." lines of the split reply ("NONE" -> None)."""
    found = {label.upper(): q.strip() for label, q in _SPLIT_LINE.findall(text)}

    def clean(value: str | None) -> str | None:
        if value is None or value.strip().rstrip(".").upper() == "NONE":
            return None
        return value

    return clean(found.get("REPORTS")), clean(found.get("SALES"))


def _fold(text: str) -> str:
    """Lower case without accents, so "Hermes" and "Hermès" compare equal."""
    stripped = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in stripped if not unicodedata.combining(c))


def subquestion_is_grounded(subquestion: str, question: str) -> bool:
    """True if the sub-question adds no number and no company absent from the question.

    The split model sometimes ignores the question and repeats the example in its prompt
    ("Who is the CEO of Kering?" -> "What was Kering's revenue in 2025?"). Such a
    sub-question mentions a year or a company the user never wrote, which is what this
    check rejects. It cannot judge wording, only invented facts.
    """
    sub, full = _fold(subquestion), _fold(question)
    if not set(_NUMBER.findall(sub)) <= set(_NUMBER.findall(full)):
        return False
    return all(name in full for name in COMPANY_NAMES if name in sub)


def _said_none(text: str, label: str) -> bool:
    """True if the model deliberately wrote "NONE" for that source (not just no line)."""
    return re.search(rf"^\s*{label}\s*:\s*NONE\b", text, re.IGNORECASE | re.MULTILINE) is not None


def drop_reason(subquestion: str, question: str) -> str:
    """Why this sub-question cannot be used, or "" if it can be."""
    if "?" not in subquestion:
        return "not a question"  # the model commented instead of splitting
    if not subquestion_is_grounded(subquestion, question):
        return "ungrounded"
    return ""


def _usage(responses: list[LLMResponse]) -> dict[str, Any]:
    def total(values: list[int | None]) -> int | None:
        return None if any(v is None for v in values) else sum(v for v in values if v)

    return {
        "llm_calls": len(responses),
        "input_tokens": total([r.input_tokens for r in responses]),
        "output_tokens": total([r.output_tokens for r in responses]),
        "llm_latency_s": round(sum(r.latency_s for r in responses), 3),
    }


def _step(state: AgentState, node: str, t0: float, **details: Any) -> list[dict[str, Any]]:
    step = {"node": node, "latency_s": round(time.perf_counter() - t0, 3), **details}
    return [*state.get("trace", []), step]


def build_graph(retriever: Retriever, llm: LLM, db_path: Path, config: AgentConfig) -> Any:
    """Compile the LangGraph graph. Dependencies are passed in, so tests can use fakes."""

    def route(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        response = llm.complete(ROUTER_SYSTEM, ROUTER_USER.format(question=state["question"]))
        parsed = parse_route(response.text)
        chosen = parsed or FALLBACK_ROUTE
        trace = _step(
            state,
            "route",
            t0,
            raw_reply=response.text[:200],
            route=chosen,
            parsed_ok=parsed is not None,
            **_usage([response]),
        )
        return {"route": chosen, "trace": trace}

    def split(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        question = state["question"]
        response = llm.complete(SPLIT_SYSTEM, SPLIT_USER.format(question=question))
        doc_q, sql_q = parse_split(response.text)
        parsed_ok = bool(doc_q or sql_q)
        dropped = {
            label: drop_reason(sub, question)
            for label, sub in (("reports", doc_q), ("sales", sql_q))
            if sub and drop_reason(sub, question)
        }
        doc_q = None if "reports" in dropped else doc_q
        sql_q = None if "sales" in dropped else sql_q
        # No usable reports sub-question: ask the reports tool the user's own question. It
        # answers "not found" when the reports do not cover it, whereas the SQL tool always
        # returns some number. Skipped only when the model deliberately said NONE.
        fell_back = doc_q is None and not _said_none(response.text, "REPORTS")
        if fell_back:
            doc_q = question
        trace = _step(
            state,
            "split",
            t0,
            doc_question=doc_q,
            sql_question=sql_q,
            parsed_ok=parsed_ok,
            dropped_subquestions=dropped,
            fell_back_to_full_question=fell_back,
            **_usage([response]),
        )
        return {"doc_question": doc_q, "sql_question": sql_q, "trace": trace}

    def doc_rag(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        doc = answer_question(
            state.get("doc_question") or state["question"],
            retriever,
            llm,
            mode=config.retrieval_mode,
            k=config.top_k,
        )
        trace = _step(
            state,
            "doc_rag",
            t0,
            retrieval_mode=doc.retrieval_mode,
            retrieved=[
                {"chunk_id": r.chunk.chunk_id, "page": r.chunk.page, "score": round(r.score, 4)}
                for r in doc.retrieved
            ],
            answer=doc.answer,
            found=doc.found,
            citations=[f"{c.source} p.{c.page}" for c in doc.citations],
            invalid_citations=doc.invalid_citations,
            **_usage([doc.llm]),
        )
        return {"doc": doc, "trace": trace}

    def sql_agent(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        sql = answer_sql_question(
            state.get("sql_question") or state["question"],
            llm,
            db_path,
            max_retries=config.sql_max_retries,
            retry_on_empty=config.sql_retry_on_empty,
            max_rows=config.sql_max_rows,
            timeout_s=config.sql_timeout_s,
        )
        trace = _step(
            state,
            "sql_agent",
            t0,
            attempts=[
                {"sql": a.sql, "error": a.error, "rejected_by_guard": a.rejected_by_guard}
                for a in sql.attempts
            ],
            succeeded=sql.succeeded,
            row_count=len(sql.result.rows) if sql.result else 0,
            first_rows=[list(map(str, r)) for r in sql.result.rows[:TRACE_ROWS]]
            if sql.result
            else [],
            answer=sql.answer,
            **_usage(sql.llm_calls),
        )
        return {"sql": sql, "trace": trace}

    def synthesize(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        doc, sql = state.get("doc"), state.get("sql")
        citations: list[Citation] = doc.citations if doc else []
        if state["route"] == "out_of_scope":
            answer = OUT_OF_SCOPE_ANSWER
        elif doc and sql:
            # Deterministic merge: no LLM rewriting, so verified numbers cannot be corrupted.
            answer = BOTH_TEMPLATE.format(doc_answer=doc.answer, sql_answer=sql.answer)
        else:
            answer = doc.answer if doc else sql.answer if sql else OUT_OF_SCOPE_ANSWER
        trace = _step(state, "synthesize", t0, answer=answer, **_usage([]))
        return {"answer": answer, "citations": citations, "trace": trace}

    def after_route(state: AgentState) -> str:
        return {"documents": "doc_rag", "both": "split", "sales_data": "sql_agent"}.get(
            state["route"], "synthesize"
        )

    def after_split(state: AgentState) -> str:
        return "doc_rag" if state.get("doc_question") else "sql_agent"

    def after_docs(state: AgentState) -> str:
        if state["route"] != "both" or not state.get("sql_question"):
            return "synthesize"
        return "sql_agent"

    graph = StateGraph(AgentState)
    graph.add_node("route", route)
    graph.add_node("split", split)
    graph.add_node("doc_rag", doc_rag)
    graph.add_node("sql_agent", sql_agent)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route", after_route, ["split", "doc_rag", "sql_agent", "synthesize"]
    )
    graph.add_conditional_edges("split", after_split, ["doc_rag", "sql_agent"])
    graph.add_conditional_edges("doc_rag", after_docs, ["sql_agent", "synthesize"])
    graph.add_edge("sql_agent", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def write_trace(result: AgentResult, log_dir: Path, extra: dict[str, Any]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = log_dir / f"{stamp}_{result.trace_id}.json"
    payload = {
        "trace_id": result.trace_id,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "question": result.question,
        "route": result.route,
        "answer": result.answer,
        "citations": [f"{c.source} p.{c.page}" for c in result.citations],
        "sql": result.sql,
        "total_latency_s": round(result.total_latency_s, 3),
        **extra,
        "steps": result.trace,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), "utf-8")
    return path


class Agent:
    """Convenience wrapper: run the graph on a question and write its trace."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        db_path: Path,
        config: AgentConfig | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.llm = llm
        self.graph = build_graph(retriever, llm, db_path, self.config)
        self.log_dir = log_dir

    def ask(self, question: str) -> AgentResult:
        t0 = time.perf_counter()
        state: AgentState = self.graph.invoke({"question": question, "trace": []})
        sql = state.get("sql")
        result = AgentResult(
            trace_id=uuid.uuid4().hex[:12],
            question=question,
            route=state["route"],
            answer=state["answer"],
            citations=state.get("citations", []),
            sql=sql.sql if sql else None,
            columns=sql.result.columns if sql and sql.result else [],
            rows=sql.result.rows if sql and sql.result else [],
            trace=state["trace"],
            total_latency_s=time.perf_counter() - t0,
        )
        if self.log_dir is not None:
            extra = {
                "llm_model": getattr(self.llm, "model", "unknown"),
                "retrieval_mode": self.config.retrieval_mode,
            }
            result.trace_path = write_trace(result, self.log_dir, extra)
        return result
