"""One entry point for all retrieval modes: `retrieve(query, mode=...)`.

Modes (compared against each other in the Stage 6 ablation):
- dense:          embedding similarity only
- bm25:           keyword matching only
- hybrid:         RRF fusion of the dense and BM25 candidate lists
- hybrid_rerank:  hybrid, then the cross-encoder reorders the top-N fused candidates
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import get_args

from assistant.config import RetrievalMode, Settings, get_settings
from assistant.ingest.chunking import Chunk
from assistant.ingest.store import load_chunks
from assistant.retrieval.bm25 import BM25Index
from assistant.retrieval.dense import DenseIndex
from assistant.retrieval.embeddings import Embedder, SentenceTransformerEmbedder
from assistant.retrieval.hybrid import rrf_fuse
from assistant.retrieval.rerank import CrossEncoderReranker, Reranker, rerank

MODES: tuple[str, ...] = get_args(RetrievalMode)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float  # meaning depends on the mode: cosine, BM25, RRF, or cross-encoder score
    rank: int  # 1 = best


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        dense: DenseIndex,
        bm25: BM25Index,
        reranker: Reranker,
        candidates: int = 20,
        rerank_top_n: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.chunks = {c.chunk_id: c for c in chunks}
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker
        self.candidates = candidates  # how many results each retriever contributes
        self.rerank_top_n = rerank_top_n  # how many fused results the cross-encoder reads
        self.rrf_k = rrf_k

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        chunk_size: int | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> Retriever:
        """Load the indexes built by `python -m assistant.ingest`."""
        s = settings or get_settings()
        index_dir = s.index_dir_for(chunk_size or s.chunk_size)
        chunks = load_chunks(index_dir)
        embedder = embedder or SentenceTransformerEmbedder(s.embedding_model)
        return cls(
            chunks=chunks,
            dense=DenseIndex.open(index_dir, embedder),
            bm25=BM25Index(chunks),
            reranker=reranker or CrossEncoderReranker(s.reranker_model),
            candidates=s.retrieval_candidates,
            rerank_top_n=s.rerank_top_n,
            rrf_k=s.rrf_k,
        )

    def _ranked_ids(self, query: str, mode: str) -> list[tuple[str, float]]:
        if mode == "dense":
            return self.dense.search(query, self.candidates)
        if mode == "bm25":
            return self.bm25.search(query, self.candidates)
        dense_ids = [cid for cid, _ in self.dense.search(query, self.candidates)]
        bm25_ids = [cid for cid, _ in self.bm25.search(query, self.candidates)]
        fused = rrf_fuse([dense_ids, bm25_ids], k=self.rrf_k)
        if mode == "hybrid":
            return fused
        top = fused[: self.rerank_top_n]
        return rerank(
            query, [(cid, self.chunks[cid].text) for cid, _ in top], self.reranker, len(top)
        )

    def retrieve(self, query: str, mode: str = "hybrid_rerank", k: int = 5) -> list[RetrievedChunk]:
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
        ranked = self._ranked_ids(query, mode)[:k]
        return [
            RetrievedChunk(self.chunks[cid], score, rank)
            for rank, (cid, score) in enumerate(ranked, start=1)
        ]


@lru_cache(maxsize=1)
def default_retriever() -> Retriever:
    return Retriever.from_settings()


def retrieve(query: str, mode: str = "hybrid_rerank", k: int = 5) -> list[RetrievedChunk]:
    """Search the indexed reports with the default settings (models loaded once)."""
    return default_retriever().retrieve(query, mode=mode, k=k)
