"""Run the whole evaluation and regenerate every table in results/ and in the README.

    python -m eval.run_all                      # all steps
    python -m eval.run_all --steps retrieval sql
    python -m eval.run_all --mock --include-unverified --limit 2   # quick dry run

Each step caches its per-question results in results/raw/, so an interrupted run can
simply be started again. Delete results/raw/<step>.jsonl to recompute a step.
"""

from __future__ import annotations

import argparse
import time

from assistant.retrieval.retriever import MODES
from eval import judge, report, run_agent, run_retrieval, run_routing, run_sql

STEPS = ("retrieval", "sql", "routing", "agent", "judge", "report")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evaluation.")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[300, 500, 1000])
    parser.add_argument("--mock", action="store_true", help="fake LLM (dry run)")
    parser.add_argument("--include-unverified", action="store_true", help="dry run only")
    parser.add_argument("--limit", type=int, help="max questions per set (dry run)")
    a = parser.parse_args()
    t0 = time.perf_counter()
    for step in a.steps:
        print(f"\n=== {step} ===")
        if step == "retrieval":
            run_retrieval.run(a.chunk_sizes, list(MODES), a.include_unverified, a.limit)
        elif step == "sql":
            run_sql.run(a.mock, a.include_unverified, a.limit)
        elif step == "routing":
            run_routing.run(a.mock, a.include_unverified, a.limit)
        elif step == "agent":
            run_agent.run(a.mock, a.include_unverified, a.limit)
        elif step == "judge":
            judge.run(a.mock, a.include_unverified)
        elif step == "report" and not a.mock:
            report.main()
    print(f"\nDone in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
