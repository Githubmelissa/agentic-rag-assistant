"""SQL evaluation: execution accuracy, with and without the retry loop.

The gold query and the predicted query are both executed; they match if they return the
same rows (see metrics.results_match). We also count guard rejections and retries.
Only the SQL step is evaluated here (no answer sentence), to save LLM time.

    python -m eval.run_sql
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from statistics import mean
from typing import Any

from assistant.llm import LLM
from assistant.tools.sql_guard import run_safe_query
from assistant.tools.sql_tool import answer_sql_question
from eval import common, metrics, setup

RETRY_SETTINGS = (2, 0)  # with retry loop (default) and without (ablation)


def evaluate_question(q: dict[str, Any], llm: LLM, max_retries: int) -> dict[str, Any]:
    s = setup.settings()
    gold = run_safe_query(s.sales_db_path, q["gold_sql"], s.sql_row_limit, 60.0)
    t0 = time.perf_counter()
    a = answer_sql_question(
        q["question"],
        llm,
        s.sales_db_path,
        max_retries=max_retries,
        max_rows=s.sql_row_limit,
        timeout_s=s.sql_timeout_seconds,
        generate_answer=False,
    )
    latency = time.perf_counter() - t0
    match = a.result is not None and metrics.results_match(
        a.result.rows, gold.rows, q.get("ordered", False)
    )
    return {
        "id": f"{q['id']}|r{max_retries}",
        "question_id": q["id"],
        "max_retries": max_retries,
        "match": match,
        "succeeded": a.succeeded,
        "attempts": len(a.attempts),
        "retries": a.retries,
        "guard_rejections": a.guard_rejections,
        "final_sql": a.sql,
        "errors": [x.error for x in a.attempts if x.error],
        "pred_first_rows": [list(map(str, r)) for r in (a.result.rows[:5] if a.result else [])],
        "gold_first_rows": [list(map(str, r)) for r in gold.rows[:5]],
        "latency_s": round(latency, 2),
        "input_tokens": sum(c.input_tokens or 0 for c in a.llm_calls),
        "output_tokens": sum(c.output_tokens or 0 for c in a.llm_calls),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_attempts = sum(r["attempts"] for r in rows)
    return {
        "n": len(rows),
        "execution_accuracy": round(mean(r["match"] for r in rows), 3),
        "query_ran": round(mean(r["succeeded"] for r in rows), 3),
        "guard_rejection_rate": round(
            sum(r["guard_rejections"] for r in rows) / max(total_attempts, 1), 3
        ),
        "questions_with_retry": sum(r["retries"] > 0 for r in rows),
        "mean_retries": round(mean(r["retries"] for r in rows), 2),
        **metrics.latency_summary([r["latency_s"] for r in rows]),
    }


def run(mock: bool = False, include_unverified: bool = False, limit: int | None = None) -> dict:
    questions = common.load_set("sql", include_unverified)[:limit]
    if not questions:
        raise SystemExit("No verified SQL questions. Verify some first (eval.verify).")
    llm = setup.llm(mock)
    cache = common.Cache("sql_mock" if mock else "sql")
    table = []
    for retries in RETRY_SETTINGS:
        rows = []
        for q in questions:
            key = f"{q['id']}|r{retries}"
            if key not in cache:
                try:
                    cache.add(evaluate_question(q, llm, retries))
                except sqlite3.Error as e:
                    raise SystemExit(f"Gold SQL of {q['id']} failed: {e}") from e
                print(f"  {key}: match={cache.rows[key]['match']}")
            rows.append(cache.rows[key])
        agg = {"max_retries": retries, **aggregate(rows)}
        table.append(agg)
        print(agg)
    payload = {
        "description": "Execution accuracy on verified SQL questions.",
        "llm": setup.llm_name(mock),
        "n_questions": len(questions),
        "include_unverified": include_unverified,
        "mock": mock,
        "rows": table,
    }
    common.save_result("sql_mock" if mock else "sql", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mock", action="store_true", help="use the fake LLM (dry run)")
    parser.add_argument("--include-unverified", action="store_true")
    parser.add_argument("--limit", type=int)
    a = parser.parse_args()
    run(a.mock, a.include_unverified, a.limit)


if __name__ == "__main__":
    main()
