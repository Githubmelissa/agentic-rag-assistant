"""Ask questions about the sales data, or run your own SQL through the safety guard.

Usage:
    python -m assistant.sql_ask "What was the total revenue in 2011?"
    python -m assistant.sql_ask                         # interactive
    python -m assistant.sql_ask --sql "SELECT country, COUNT(*) FROM sales GROUP BY country"
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import textwrap
import time

from assistant.config import Settings, get_settings
from assistant.llm import LLM, LLMError, make_llm
from assistant.tools.sql_guard import QueryResult, SQLGuardError, run_safe_query
from assistant.tools.sql_tool import answer_sql_question


def print_table(result: QueryResult, limit: int = 15) -> None:
    print(" | ".join(result.columns))
    for row in result.rows[:limit]:
        print(" | ".join(str(v) for v in row))
    if len(result.rows) > limit:
        print(f"... ({len(result.rows)} rows, first {limit} shown)")


def run_raw_sql(sql: str, s: Settings) -> None:
    try:
        result = run_safe_query(s.sales_db_path, sql, s.sql_row_limit, s.sql_timeout_seconds)
    except SQLGuardError as e:
        sys.exit(f"Rejected by the guard: {e}")
    except sqlite3.Error as e:
        sys.exit(f"SQLite error: {e}")
    print(f"Executed: {result.sql}\n")
    print_table(result)


def ask_once(question: str, llm: LLM, s: Settings) -> None:
    t0 = time.perf_counter()
    try:
        a = answer_sql_question(
            question,
            llm,
            s.sales_db_path,
            max_retries=s.sql_max_retries,
            retry_on_empty=s.sql_retry_on_empty,
            max_rows=s.sql_row_limit,
            timeout_s=s.sql_timeout_seconds,
        )
    except LLMError as e:
        print(f"\nLLM error: {e}", file=sys.stderr)
        return
    for i, attempt in enumerate(a.attempts, start=1):
        status = "OK" if attempt.error is None else f"FAILED: {attempt.error}"
        print(f"\nSQL attempt {i} ({status})")
        print(textwrap.indent(attempt.sql, "    "))
    if a.result is not None:
        print("\nRESULT")
        print_table(a.result)
    print("\nANSWER")
    print(textwrap.fill(a.answer, width=90))
    llm_s = sum(c.latency_s for c in a.llm_calls)
    print(
        f"\n[{len(a.llm_calls)} LLM calls, {a.retries} retries, {a.guard_rejections} guard "
        f"rejections | LLM {llm_s:.1f} s | total {time.perf_counter() - t0:.1f} s]"
    )


def main(argv: list[str] | None = None) -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(description="Ask about the sales data.")
    parser.add_argument("question", nargs="?", help="omit for interactive mode")
    parser.add_argument("--sql", help="run this SQL directly (through the guard), no LLM")
    args = parser.parse_args(argv)

    if args.sql:
        run_raw_sql(args.sql, s)
        return
    try:
        llm = make_llm(s)
    except LLMError as e:
        sys.exit(f"LLM configuration error: {e}")
    if args.question:
        ask_once(args.question, llm, s)
        return
    while True:
        try:
            question = input("\nQuestion (empty line to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        ask_once(question, llm, s)


if __name__ == "__main__":
    main()
