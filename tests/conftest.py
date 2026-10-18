"""Shared test helpers: a tiny PDF generator and a fake embedder (no model download)."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pymupdf
import pytest

from assistant.retrieval.bm25 import tokenize


class FakeEmbedder:
    """Bag-of-words hashed into 64 dimensions: texts sharing words get similar vectors."""

    dim = 64

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for token in tokenize(text):
            v[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def make_pdf(path: Path, pages: list[str], footer: str | None = None) -> Path:
    """Write a PDF with one text block per page (plus an optional repeated footer)."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 700), text, fontsize=10)
        if footer:
            page.insert_text((50, 800), footer, fontsize=8)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
