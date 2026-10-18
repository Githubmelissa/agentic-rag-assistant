from pathlib import Path

import pytest

from assistant.ingest.chunking import Chunk, count_words
from assistant.retrieval.bm25 import BM25Index
from assistant.retrieval.dense import DenseIndex
from assistant.retrieval.rerank import rerank
from assistant.retrieval.retriever import MODES, Retriever
from tests.conftest import FakeEmbedder


class KeywordReranker:
    """Fake cross-encoder: score = how many query words appear in the text."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        words = set(query.lower().split())
        return [float(len(words & set(t.lower().split()))) for t in texts]


def chunk(cid: str, text: str, page: int = 1) -> Chunk:
    return Chunk(cid, "doc", page, "", text, count_words(text))


CHUNKS = [
    chunk("rev", "group revenue amounted to 14,675 million euros in 2025", 47),
    chunk("tax", "income tax breakdown for the year 2025", 457),
    chunk("eye", "kering eyewear revenue grew in 2025", 66),
    chunk("co2", "greenhouse gas emissions and renewable energy", 200),
]


@pytest.fixture
def retriever(tmp_path: Path, fake_embedder: FakeEmbedder) -> Retriever:
    dense = DenseIndex.build(CHUNKS, tmp_path, fake_embedder)
    return Retriever(CHUNKS, dense, BM25Index(CHUNKS), KeywordReranker(), candidates=4)


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_ranked_chunks(retriever: Retriever, mode: str) -> None:
    results = retriever.retrieve("group revenue 2025", mode=mode, k=2)
    assert 1 <= len(results) <= 2
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert all(isinstance(r.chunk, Chunk) for r in results)


def test_hybrid_rerank_puts_best_match_first(retriever: Retriever) -> None:
    results = retriever.retrieve("group revenue amounted", mode="hybrid_rerank", k=3)
    assert results[0].chunk.chunk_id == "rev"
    assert results[0].chunk.page == 47


def test_unknown_mode_rejected(retriever: Retriever) -> None:
    with pytest.raises(ValueError):
        retriever.retrieve("revenue", mode="magic")


def test_rerank_reorders_and_truncates() -> None:
    candidates = [("a", "nothing relevant"), ("b", "revenue in 2025"), ("c", "revenue")]
    result = rerank("revenue 2025", candidates, KeywordReranker(), top_k=2)
    assert [cid for cid, _ in result] == ["b", "c"]


def test_rerank_keeps_order_on_ties() -> None:
    candidates = [("a", "x"), ("b", "y")]
    assert [cid for cid, _ in rerank("z", candidates, KeywordReranker(), 2)] == ["a", "b"]
