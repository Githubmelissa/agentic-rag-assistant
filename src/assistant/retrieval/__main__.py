"""Try a search from the command line.

Usage:
    python -m assistant.retrieval "What was Kering's revenue in 2025?"
    python -m assistant.retrieval "carbon emissions target" --mode bm25 -k 3
    python -m assistant.retrieval "What was Kering's revenue in 2025?" --compare
"""

from __future__ import annotations

import argparse
import textwrap
import time

from assistant.config import get_settings
from assistant.retrieval.retriever import MODES, RetrievedChunk, Retriever


def cite(c: RetrievedChunk) -> str:
    label = f", printed p.{c.chunk.page_label}" if c.chunk.page_label else ""
    return f"{c.chunk.source} PDF p.{c.chunk.page}{label}"


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Search the indexed reports.")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--compare", action="store_true", help="show the top-k pages of every mode")
    args = parser.parse_args(argv)

    retriever = Retriever.from_settings(settings, chunk_size=args.chunk_size)

    if args.compare:
        retriever.retrieve(args.query, "hybrid_rerank", 1)  # warm-up: load both models once
        for mode in MODES:
            t0 = time.perf_counter()
            results = retriever.retrieve(args.query, mode, args.k)
            ms = (time.perf_counter() - t0) * 1000
            pages = ", ".join(f"{r.chunk.source.split('_')[0]} p.{r.chunk.page}" for r in results)
            print(f"{mode:<14} {ms:7.0f} ms   {pages}")
        return

    for r in retriever.retrieve(args.query, args.mode, args.k):
        print(f"\n#{r.rank}  {cite(r)}  score={r.score:.3f}")
        print(textwrap.indent(textwrap.shorten(r.chunk.text, 400), "    "))


if __name__ == "__main__":
    main()
