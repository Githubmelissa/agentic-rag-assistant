"""Try a search from the command line.

Usage:
    python -m assistant.retrieval "What was Kering's revenue in 2025?"
    python -m assistant.retrieval "carbon emissions target" --mode bm25 -k 3
"""

from __future__ import annotations

import argparse
import textwrap

from assistant.config import get_settings
from assistant.ingest.store import load_chunks
from assistant.retrieval.bm25 import BM25Index
from assistant.retrieval.dense import DenseIndex
from assistant.retrieval.embeddings import SentenceTransformerEmbedder


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Search the indexed reports.")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["dense", "bm25"], default="dense")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    args = parser.parse_args(argv)

    index_dir = settings.index_dir_for(args.chunk_size)
    chunks = {c.chunk_id: c for c in load_chunks(index_dir)}
    if args.mode == "bm25":
        results = BM25Index(list(chunks.values())).search(args.query, args.k)
    else:
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
        results = DenseIndex.open(index_dir, embedder).search(args.query, args.k)

    for rank, (chunk_id, score) in enumerate(results, start=1):
        c = chunks[chunk_id]
        label = f", printed p.{c.page_label}" if c.page_label else ""
        print(f"\n#{rank}  {c.source}  PDF p.{c.page}{label}  score={score:.3f}")
        print(textwrap.indent(textwrap.shorten(c.text, 400), "    "))


if __name__ == "__main__":
    main()
