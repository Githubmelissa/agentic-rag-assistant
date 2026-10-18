"""Ask questions about the reports from the command line.

Usage:
    python -m assistant.ask "What was Kering's revenue in 2025?"
    python -m assistant.ask                      # interactive: type questions, empty line to quit
    python -m assistant.ask "..." --mode hybrid -k 3
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time

from assistant.config import get_settings
from assistant.llm import LLM, LLMError, make_llm
from assistant.retrieval.retriever import MODES, Retriever
from assistant.tools.doc_search import DocAnswer, answer_question


def show(a: DocAnswer, total_s: float) -> None:
    print("\nANSWER")
    print(textwrap.fill(a.answer, width=90))
    if a.citations:
        print("\nCITATIONS")
        for c in a.citations:
            printed = f" (printed p.{c.page_label})" if c.page_label else ""
            print(f"  {c.source} PDF p.{c.page}{printed}")
            print(textwrap.indent(textwrap.shorten(c.snippet, 200), "      "))
    if a.invalid_citations:
        print(
            "\nWARNING: cited pages that were not in the context:", ", ".join(a.invalid_citations)
        )
    if a.uncited:
        print("\nWARNING: this answer cites no passage from the context; treat it as unsupported.")
    if not a.found:
        pages = ", ".join(f"{r.chunk.source} p.{r.chunk.page}" for r in a.retrieved)
        print(f"\n(retrieved but not used: {pages})")
    tokens = f"{a.llm.input_tokens} in / {a.llm.output_tokens} out"
    print(
        f"\n[{a.retrieval_mode} | {a.llm.model} | {tokens} tokens | "
        f"LLM {a.llm.latency_s:.1f} s | total {total_s:.1f} s]"
    )


def ask_once(question: str, retriever: Retriever, llm: LLM, mode: str, k: int) -> None:
    t0 = time.perf_counter()
    try:
        a = answer_question(question, retriever, llm, mode=mode, k=k)
    except LLMError as e:
        print(f"\nLLM error: {e}", file=sys.stderr)
        return
    show(a, time.perf_counter() - t0)


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ask a question about the reports.")
    parser.add_argument("question", nargs="?", help="omit for interactive mode")
    parser.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    parser.add_argument("-k", type=int, default=settings.answer_top_k)
    args = parser.parse_args(argv)

    try:
        llm = make_llm(settings)
    except LLMError as e:
        sys.exit(f"LLM configuration error: {e}")
    retriever = Retriever.from_settings(settings)
    print(f"Using LLM provider '{settings.llm_provider}' ({settings.llm_model or 'mock'}).")

    if args.question:
        ask_once(args.question, retriever, llm, args.mode, args.k)
        return
    while True:
        try:
            question = input("\nQuestion (empty line to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        ask_once(question, retriever, llm, args.mode, args.k)


if __name__ == "__main__":
    main()
