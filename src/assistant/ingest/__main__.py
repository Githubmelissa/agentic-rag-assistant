"""Build the indexes from the PDFs.

Usage (from the project root, with the virtual environment active):
    python -m assistant.ingest
    python -m assistant.ingest --input data/raw --chunk-size 500 --chunk-overlap 80
    python -m assistant.ingest --bm25-only      # fast: skip the embedding model
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from assistant.config import get_settings
from assistant.ingest.chunking import Chunk, TokenCounter, chunk_pages
from assistant.ingest.loaders import Page, load_pdfs
from assistant.ingest.store import save_chunks
from assistant.retrieval.bm25 import BM25Index
from assistant.retrieval.dense import DenseIndex
from assistant.retrieval.embeddings import (
    Embedder,
    SentenceTransformerEmbedder,
    make_token_counter,
)

MANIFEST_FILE = "manifest.json"


def summarize(pages: list[Page], chunks: list[Chunk]) -> list[dict[str, object]]:
    """Per-document statistics, printed and saved in the manifest."""
    pages_by_doc: dict[str, int] = defaultdict(int)
    tokens_by_doc: dict[str, list[int]] = defaultdict(list)
    for p in pages:
        pages_by_doc[p.source] += 1
    for c in chunks:
        tokens_by_doc[c.source].append(c.n_tokens)
    return [
        {
            "source": src,
            "pages_with_text": pages_by_doc[src],
            "chunks": len(tokens_by_doc[src]),
            "median_tokens": int(statistics.median(tokens_by_doc[src])),
            "max_tokens": max(tokens_by_doc[src]),
        }
        for src in sorted(pages_by_doc)
    ]


def build(
    input_dir: Path,
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    count_tokens: TokenCounter,
    embedder: Embedder | None,
    embedding_model: str,
) -> dict[str, object]:
    """Load PDFs, chunk them, save chunks, build both indexes. Returns the manifest."""
    t0 = time.perf_counter()
    pages = load_pdfs(input_dir)
    chunks = chunk_pages(pages, chunk_size, chunk_overlap, count_tokens)
    save_chunks(chunks, index_dir)
    BM25Index(chunks)  # built here only to fail early if something is wrong
    t_chunk = time.perf_counter() - t0

    t1 = time.perf_counter()
    if embedder is not None:
        DenseIndex.build(chunks, index_dir, embedder)
    t_embed = time.perf_counter() - t1

    manifest: dict[str, object] = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input_files": sorted(p.name for p in input_dir.glob("*.pdf")),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model if embedder is not None else None,
        "n_chunks": len(chunks),
        "documents": summarize(pages, chunks),
        "seconds_load_and_chunk": round(t_chunk, 1),
        "seconds_embed_and_index": round(t_embed, 1),
    }
    (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def print_summary(manifest: dict[str, object]) -> None:
    print(f"\n{'document':<28}{'pages':>7}{'chunks':>8}{'median tok':>12}{'max tok':>9}")
    for d in manifest["documents"]:  # type: ignore[union-attr]
        print(
            f"{d['source']:<28}{d['pages_with_text']:>7}{d['chunks']:>8}"
            f"{d['median_tokens']:>12}{d['max_tokens']:>9}"
        )
    print(f"\nTotal chunks: {manifest['n_chunks']}")
    print(f"Load + chunk: {manifest['seconds_load_and_chunk']} s")
    print(f"Embed + index: {manifest['seconds_embed_and_index']} s")


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the dense and BM25 indexes.")
    parser.add_argument("--input", type=Path, default=settings.raw_dir)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument(
        "--bm25-only", action="store_true", help="skip embeddings (fast, for testing)"
    )
    args = parser.parse_args(argv)

    index_dir = settings.index_dir_for(args.chunk_size)
    count_tokens = make_token_counter(settings.embedding_model)
    embedder = None if args.bm25_only else SentenceTransformerEmbedder(settings.embedding_model)
    if embedder is not None and args.chunk_size > embedder.max_seq_length:
        print(
            f"WARNING: chunk size {args.chunk_size} > model limit {embedder.max_seq_length} "
            "tokens. The embedding model will only see the start of each chunk "
            "(BM25 still sees all of it)."
        )

    print(f"Reading PDFs from {args.input}")
    print(f"Writing indexes to {index_dir}")
    manifest = build(
        args.input,
        index_dir,
        args.chunk_size,
        args.chunk_overlap,
        count_tokens,
        embedder,
        settings.embedding_model,
    )
    print_summary(manifest)


if __name__ == "__main__":
    main()
