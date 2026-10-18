"""Keyword index: BM25 over the same chunks as the dense index.

BM25 scores a chunk higher when it contains the query's words, especially rare words
(names, figures, acronyms such as "EBITDA"), and normalises for chunk length. It is
rebuilt in memory from `chunks.jsonl` when loaded: this takes about a second, and avoids
pickle files (which are fragile across library versions and unsafe to load if untrusted).
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from assistant.ingest.chunking import Chunk

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")  # words and numbers such as 3.5 or 1,234

STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the this to
    was were what which who will with how did does do our their they we you""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words/numbers, drop very common English words."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build a BM25 index from zero chunks")
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return up to k chunks as (chunk_id, BM25 score), best first; zero scores dropped."""
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top if scores[i] > 0]
