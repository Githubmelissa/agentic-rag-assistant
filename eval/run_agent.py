"""End-to-end evaluation of the full agent on document, mixed and out-of-scope questions.

Per question we record the route, the answer, the citations, the SQL, latency per node
and token counts (from the agent's trace), then score:
- documents: the gold headline number (or name) appears in the answer; at least one cited
  page is a gold page;
- mixed: the reports part (number/name match) and the SQL part (execution match);
- out-of-scope: the agent refused (out_of_scope route or "could not find").
Latency (median, p95) per route and tokens per query come from the same runs.
This is the slow step: expect about a minute per question on a laptop CPU.

    python -m eval.run_agent
"""

from __future__ import annotations

import argparse
from statistics import mean, median
from typing import Any

from assistant.agent.graph import Agent, AgentConfig
from assistant.retrieval.retriever import Retriever
from assistant.tools.sql_guard import run_safe_query
from eval import common, metrics, setup


def _tokens(trace: list[dict[str, Any]]) -> int | None:
    values = [s.get(k) for s in trace for k in ("input_tokens", "output_tokens") if k in s]
    return None if any(v is None for v in values) else int(sum(values))


def score_doc(q: dict[str, Any], answer: str, cited_pages: list[str]) -> dict[str, Any]:
    num = metrics.number_match(answer, q["gold_answer"])
    content = num if num is not None else metrics.name_match(answer, q["gold_answer"])
    gold_labels = {f"{q['gold_source']} p.{p}" for p in q["gold_pages"]}
    return {
        "content_match": content,
        "match_type": "number" if num is not None else "name",
        "cites_gold_page": any(c in gold_labels for c in cited_pages),
    }


def page_texts(retriever: Retriever, citations: list[Any]) -> list[str]:
    """Full text of every cited page (all its chunks), for the LLM judge."""
    out = []
    for c in citations:
        chunks = [x for x in retriever.chunks.values() if (x.source, x.page) == (c.source, c.page)]
        out.append(f"[{c.source} p.{c.page}] " + " ".join(x.text for x in chunks))
    return out


def evaluate_question(
    agent: Agent, retriever: Retriever, q: dict[str, Any], kind: str
) -> dict[str, Any]:
    r = agent.ask(q["question"])
    cited = [f"{c.source} p.{c.page}" for c in r.citations]
    doc_step = next((s for s in r.trace if s["node"] == "doc_rag"), None)
    row: dict[str, Any] = {
        "id": q["id"],
        "set": kind,
        "expected_route": q["route"],
        "route": r.route,
        "answer": r.answer,
        "citations": cited,
        "found": doc_step["found"] if doc_step else None,
        "doc_answer": doc_step["answer"] if doc_step else None,
        "cited_texts": page_texts(retriever, r.citations),
        "sql": r.sql,
        "first_rows": [list(map(str, x)) for x in r.rows[:5]],
        "latency_s": round(r.total_latency_s, 2),
        "node_latency_s": {s["node"]: s["latency_s"] for s in r.trace},
        "tokens": _tokens(r.trace),
        "trace_id": r.trace_id,
    }
    if kind == "doc":
        row.update(score_doc(q, r.answer, cited))
    elif kind == "mixed":
        row.update(score_doc(q, doc_step["answer"] if doc_step else "", cited))
        s = setup.settings()
        gold = run_safe_query(s.sales_db_path, q["gold_sql"], s.sql_row_limit, 60.0)
        row["sql_match"] = bool(
            r.sql is not None and metrics.results_match(r.rows, gold.rows, q.get("ordered", False))
        )
    else:
        row["refused"] = metrics.is_refusal(r.answer, r.route)
    return row


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(mean(vals), 3) if vals else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    doc = [r for r in rows if r["set"] == "doc"]
    mixed = [r for r in rows if r["set"] == "mixed"]
    oos = [r for r in rows if r["set"] == "oos"]
    by_route: dict[str, list[float]] = {}
    for r in rows:
        by_route.setdefault(r["route"], []).append(r["latency_s"])
    tokens = [r["tokens"] for r in rows if r["tokens"] is not None]
    return {
        "documents": {
            "n": len(doc),
            "answer_match": _rate(doc, "content_match"),
            "cites_gold_page": _rate(doc, "cites_gold_page"),
            "said_not_found": _rate([{"x": r["found"] is False} for r in doc], "x"),
        },
        "mixed": {
            "n": len(mixed),
            "routed_both": _rate([{"x": r["route"] == "both"} for r in mixed], "x"),
            "reports_part_match": _rate(mixed, "content_match"),
            "sql_part_match": _rate(mixed, "sql_match"),
        },
        "out_of_scope": {"n": len(oos), "refusal_rate": _rate(oos, "refused")},
        "latency_by_route": {k: metrics.latency_summary(v) for k, v in sorted(by_route.items())},
        "tokens_per_query": {
            "n": len(tokens),
            "median": median(tokens) if tokens else None,
            "mean": round(mean(tokens), 1) if tokens else None,
        },
    }


def run(mock: bool = False, include_unverified: bool = False, limit: int | None = None) -> dict:
    s = setup.settings()
    sets = {k: common.load_set(k, include_unverified)[:limit] for k in ("doc", "mixed", "oos")}
    if not any(sets.values()):
        raise SystemExit("No verified questions. Verify some first (eval.verify).")
    name = "agent_mock" if mock else "agent"
    cache = common.Cache(name)
    agent: Agent | None = None
    retriever: Retriever | None = None
    ids = []
    for kind, questions in sets.items():
        for q in questions:
            ids.append(q["id"])
            if q["id"] in cache:
                continue
            if agent is None:
                config = AgentConfig(
                    top_k=s.answer_top_k,
                    sql_max_retries=s.sql_max_retries,
                    sql_max_rows=s.sql_row_limit,
                    sql_timeout_s=s.sql_timeout_seconds,
                )
                retriever = setup.retriever()
                agent = Agent(retriever, setup.llm(mock), s.sales_db_path, config, s.log_dir)
            assert retriever is not None
            row = evaluate_question(agent, retriever, q, kind)
            cache.add(row)
            print(f"  {q['id']}: route={row['route']} latency={row['latency_s']}s")
    rows = cache.select(ids)
    payload = {
        "description": "End-to-end agent results on verified document, mixed and OOS questions.",
        "llm": setup.llm_name(mock),
        "include_unverified": include_unverified,
        "mock": mock,
        **aggregate(rows),
    }
    common.save_result(name, payload)
    print(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--include-unverified", action="store_true")
    parser.add_argument("--limit", type=int, help="max questions per set")
    a = parser.parse_args()
    run(a.mock, a.include_unverified, a.limit)


if __name__ == "__main__":
    main()
