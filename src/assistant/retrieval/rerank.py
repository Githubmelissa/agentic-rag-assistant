"""Cross-encoder reranking.

Dense retrieval embeds the query and each chunk *separately* (a "bi-encoder"), which is
fast but coarse. A cross-encoder reads the query and one chunk *together* and outputs a
relevance score, so it can check that the chunk actually answers the question. It is too
slow to run on 2,000 chunks per query, so we only apply it to the top-N fused candidates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (small, CPU-friendly)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict([(query, t) for t in texts], show_progress_bar=False)
        return [float(s) for s in scores]


def rerank(
    query: str, candidates: list[tuple[str, str]], reranker: Reranker, top_k: int
) -> list[tuple[str, float]]:
    """Reorder (chunk_id, text) candidates by cross-encoder score; return the best top_k."""
    scores = reranker.score(query, [text for _, text in candidates])
    ranked = sorted(
        zip((cid for cid, _ in candidates), scores, strict=True),
        key=lambda pair: -pair[1],  # sorted() is stable: equal scores keep fused order
    )
    return ranked[:top_k]
