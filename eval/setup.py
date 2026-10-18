"""Build the objects the evaluation needs (real models, or fakes for a dry run)."""

from __future__ import annotations

from functools import cache

from assistant.config import Settings, get_settings
from assistant.llm import LLM, MockLLM, make_llm
from assistant.retrieval.embeddings import SentenceTransformerEmbedder
from assistant.retrieval.rerank import CrossEncoderReranker
from assistant.retrieval.retriever import Retriever


def settings() -> Settings:
    return get_settings()


def llm(mock: bool) -> LLM:
    return MockLLM() if mock else make_llm(settings())


@cache
def _embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(settings().embedding_model)


@cache
def _reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(settings().reranker_model)


def retriever(chunk_size: int | None = None) -> Retriever:
    """Models are shared between chunk sizes, so they are loaded only once."""
    return Retriever.from_settings(
        settings(), chunk_size=chunk_size, embedder=_embedder(), reranker=_reranker()
    )


def llm_name(mock: bool) -> str:
    s = settings()
    return "mock" if mock else f"{s.llm_provider}:{s.llm_model}"
