"""Ask the full assistant (router + tools) from the command line.

Usage:
    python -m assistant.agent "What was Hermès' revenue in 2025?"
    python -m assistant.agent            # interactive
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from assistant.agent.graph import Agent, AgentConfig, AgentResult
from assistant.config import get_settings
from assistant.llm import LLMError, make_llm
from assistant.retrieval.retriever import Retriever


def show(r: AgentResult) -> None:
    print(f"\nROUTE: {r.route}")
    print("\nANSWER")
    print(textwrap.fill(r.answer, width=90))
    if r.citations:
        print("\nCITATIONS")
        for c in r.citations:
            printed = f" (printed p.{c.page_label})" if c.page_label else ""
            print(f"  {c.source} PDF p.{c.page}{printed}")
    if r.sql:
        print("\nSQL")
        print(textwrap.indent(r.sql, "    "))
        print(f"  -> {len(r.rows)} rows; first: {r.rows[:3]}")
    steps = ", ".join(f"{s['node']} {s['latency_s']:.1f}s" for s in r.trace)
    print(f"\n[{steps} | total {r.total_latency_s:.1f} s]")
    if r.trace_path:
        print(f"[trace: {r.trace_path}]")


def main(argv: list[str] | None = None) -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(description="Ask the assistant.")
    parser.add_argument("question", nargs="?", help="omit for interactive mode")
    args = parser.parse_args(argv)
    try:
        llm = make_llm(s)
    except LLMError as e:
        sys.exit(f"LLM configuration error: {e}")
    config = AgentConfig(
        top_k=s.answer_top_k,
        sql_max_retries=s.sql_max_retries,
        sql_max_rows=s.sql_row_limit,
        sql_timeout_s=s.sql_timeout_seconds,
    )
    agent = Agent(Retriever.from_settings(s), llm, s.sales_db_path, config, log_dir=s.log_dir)

    def ask(question: str) -> None:
        try:
            show(agent.ask(question))
        except LLMError as e:
            print(f"\nLLM error: {e}", file=sys.stderr)

    if args.question:
        ask(args.question)
        return
    while True:
        try:
            question = input("\nQuestion (empty line to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        ask(question)


if __name__ == "__main__":
    main()
