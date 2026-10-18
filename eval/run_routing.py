"""Routing accuracy: does the router pick the labelled route for each question?

Runs only the `route` step (one short LLM call per question) on every verified question
of the four sets: documents, sales_data, both, out_of_scope.

    python -m eval.run_routing
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from statistics import mean
from typing import Any

from assistant.agent.graph import FALLBACK_ROUTE, parse_route
from assistant.agent.prompts import ROUTER_SYSTEM, ROUTER_USER, ROUTES
from assistant.llm import LLM
from eval import common, setup


def evaluate_question(q: dict[str, Any], llm: LLM) -> dict[str, Any]:
    t0 = time.perf_counter()
    r = llm.complete(ROUTER_SYSTEM, ROUTER_USER.format(question=q["question"]))
    parsed = parse_route(r.text)
    predicted = parsed or FALLBACK_ROUTE
    return {
        "id": q["id"],
        "expected": q["route"],
        "predicted": predicted,
        "parsed_ok": parsed is not None,
        "correct": predicted == q["route"],
        "raw_reply": r.text[:100],
        "latency_s": round(time.perf_counter() - t0, 2),
    }


def run(mock: bool = False, include_unverified: bool = False, limit: int | None = None) -> dict:
    questions = [
        q for name in common.SETS for q in common.load_set(name, include_unverified)[:limit]
    ]
    if not questions:
        raise SystemExit("No verified questions. Verify some first (eval.verify).")
    llm = setup.llm(mock)
    name = "routing_mock" if mock else "routing"
    cache = common.Cache(name)
    for q in questions:
        if q["id"] not in cache:
            cache.add(evaluate_question(q, llm))
    rows = cache.select([q["id"] for q in questions])
    per_route = {
        route: {
            "n": len(sub),
            "accuracy": round(mean(r["correct"] for r in sub), 3),
        }
        for route in ROUTES
        if (sub := [r for r in rows if r["expected"] == route])
    }
    confusion = Counter(f"{r['expected']} -> {r['predicted']}" for r in rows)
    payload = {
        "description": "Router accuracy against the labelled route of each verified question.",
        "llm": setup.llm_name(mock),
        "include_unverified": include_unverified,
        "mock": mock,
        "n": len(rows),
        "accuracy": round(mean(r["correct"] for r in rows), 3),
        "unparsed_replies": sum(not r["parsed_ok"] for r in rows),
        "per_route": per_route,
        "confusion": dict(sorted(confusion.items())),
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
