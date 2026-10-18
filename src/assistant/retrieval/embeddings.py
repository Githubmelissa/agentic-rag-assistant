"""Embedding model and tokenizer wrappers.

Both are loaded lazily (only when first used), because loading PyTorch models takes a few
seconds and is not needed by BM25-only runs or by tests. Tests use small fakes that follow
the same `Embedder` protocol, so they never download a model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from assistant.ingest.chunking import TokenCounter

# BGE v1.5 models work best when *queries* (not documents) carry this instruction.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Local embedding model (default: BAAI/bge-small-en-v1.5, 384 dimensions, CPU-friendly)."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    @property
    def max_seq_length(self) -> int:
        return int(self.model.max_seq_length)

    def _query_prefix(self) -> str:
        return BGE_QUERY_INSTRUCTION if "bge" in self.model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # unit vectors: cosine similarity = dot product
            show_progress_bar=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(self._query_prefix() + text, normalize_embeddings=True)
        return vector.tolist()


def make_token_counter(model_name: str) -> TokenCounter:
    """Count tokens with the embedding model's own tokenizer (what the model really sees)."""
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # silence "sequence longer than max length" warnings
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count
