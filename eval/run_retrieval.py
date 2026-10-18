"""Retrieval evaluation and ablations (no LLM needed).

For every verified document question, each retrieval mode returns a ranked list of
chunks; chunks are reduced to unique (report, page) pairs, and we look for the first page
listed in `gold_pages`. Metrics: recall@3, recall@5, recall@10 and MRR, plus the median
retrieval latency per mode. Run for every chunk size that has an index
(`python -m assistant.ingest --chunk-size 300` builds one).

    python -m eval.run_retrieval
    python -m eval.run_retrieval --chunk-sizes 300 500 1000
"""

from __future__ import annotations

import argparse
import time
from statistics import mean, median
from typing import Any

from assistant.retrieval.retriever import MODES, Retriever
from eval import common, metrics, setup

KS = (3, 5, 10)
CHUNKS_RETRIEVED = 30  # chunks per query before reducing to unique pages


def evaluate_question(
    retriever: Retriever, q: dict[str, Any], mode: str, chunk_size: int
) -> dict[str, Any]:
    t0 = time.perf_counter()
    results = retriever.retrieve(q["question"], mode=mode, k=CHUNKS_RETRIEVED)
    latency = time.perf_counter() - t0
    pages = metrics.unique_pages([(r.chunk.source, r.chunk.page) for r in results])[:10]
    gold = {(q["gold_source"], p) for p in q["gold_pages"]}
    rank = metrics.first_hit_rank(pages, gold)
    return {
        "id": f"{q['id']}|cs{chunk_size}|{mode}",
        "question_id": q["id"],
        "chunk_size": chunk_size,
        "mode": mode,
        "rank": rank,
        "top_pages": [f"{s} p.{p}" for s, p in pages],
        "latency_s": round(latency, 4),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [r["rank"] for r in rows]
    out: dict[str, Any] = {"n": len(rows)}
    for k in KS:
        out[f"recall@{k}"] = round(mean(metrics.recall_at_k(r, k) for r in ranks), 3)
    out["mrr"] = round(mean(metrics.reciprocal_rank(r) for r in ranks), 3)
    out["median_latency_s"] = round(median(r["latency_s"] for r in rows), 3)
    return out


def run(
    chunk_sizes: list[int],
    modes: list[str],
    include_unverified: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    questions = common.load_set("doc", include_unverified)[:limit]
    if not questions:
        raise SystemExit("No verified document questions. Verify some first (eval.verify).")
    cache = common.Cache("retrieval")
    table = []
    for cs in chunk_sizes:
        index_dir = setup.settings().index_dir_for(cs)
        if not (index_dir / "chunks.jsonl").exists():
            print(f"skip chunk size {cs}: no index (python -m assistant.ingest --chunk-size {cs})")
            continue
        retriever: Retriever | None = None
        for mode in modes:
            rows = []
            for q in questions:
                key = f"{q['id']}|cs{cs}|{mode}"
                if key not in cache:
                    retriever = retriever or setup.retriever(cs)
                    cache.add(evaluate_question(retriever, q, mode, cs))
                rows.append(cache.rows[key])
            agg = {"chunk_size": cs, "mode": mode, **aggregate(rows)}
            table.append(agg)
            print(agg)
    payload = {
        "description": "Page-level retrieval metrics on verified document questions.",
        "n_questions": len(questions),
        "include_unverified": include_unverified,
        "rows": table,
    }
    common.save_result("retrieval", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[300, 500, 1000])
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--include-unverified", action="store_true")
    parser.add_argument("--limit", type=int)
    a = parser.parse_args()
    run(a.chunk_sizes, a.modes, a.include_unverified, a.limit)


if __name__ == "__main__":
    main()
