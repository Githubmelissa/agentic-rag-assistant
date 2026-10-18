"""Reciprocal Rank Fusion (RRF): merge several ranked lists into one.

    score(d) = sum over lists L containing d of  1 / (k + rank_L(d)),   rank starts at 1

Why rank-based fusion: dense scores (cosine, ~0.7-0.8) and BM25 scores (0 to ~40) live on
different scales, so adding them directly would be meaningless. RRF only uses positions.
A chunk ranked well by *both* retrievers gets two contributions and rises to the top.

Why k = 60: the value from the original paper (Cormack, Clarke & Büttcher, SIGIR 2009),
found to work well across collections. A large k flattens the gap between rank 1 and rank
5 (1/61 vs 1/65), so agreement between lists matters more than being first in one list.
"""

from __future__ import annotations

DEFAULT_K = 60


def rrf_fuse(rankings: list[list[str]], k: int = DEFAULT_K) -> list[tuple[str, float]]:
    """Fuse ranked lists of IDs (best first). Returns (id, rrf_score), best first.

    Ties are broken by first appearance (list order, then rank), so output is deterministic.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            if item not in first_seen:
                first_seen[item] = order
                order += 1
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
