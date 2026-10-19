"""Shared test helpers: a tiny PDF generator and a fake embedder (no model download)."""

from __future__ import annotations

import hashlib
import math
import sqlite3
from contextlib import closing
from pathlib import Path

import pymupdf
import pytest

from assistant.config import Settings
from assistant.retrieval.bm25 import tokenize
from assistant.sales_data import SCHEMA_SQL


def make_settings(**overrides: object) -> Settings:
    """Settings for tests.

    `_env_file=None` is what makes them reproducible: without it, pydantic-settings reads
    the developer's own `.env`, so the same test passes on a fresh clone and fails on the
    machine where the project is actually configured (see failure log 19).
    """
    return Settings(_env_file=None, **overrides)


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


SAMPLE_ROWS = [
    # invoice, stock_code, description, qty, date, price, customer, country, cancelled, product
    ("1001", "85123A", "WHITE HEART", 6, "2010-03-01 10:00:00", 2.55, 1, "United Kingdom", 0, 1),
    ("1001", "POST", "POSTAGE", 1, "2010-03-01 10:00:00", 18.0, 1, "United Kingdom", 0, 0),
    ("1002", "22423", "CAKESTAND", 2, "2011-05-02 11:00:00", 12.75, 2, "France", 0, 1),
    ("1003", "22423", "CAKESTAND", 1, "2011-06-03 12:00:00", 12.75, None, "EIRE", 0, 1),
    ("C1004", "22423", "CAKESTAND", -1, "2011-06-04 12:00:00", 12.75, 2, "France", 1, 1),
]


@pytest.fixture
def sales_db(tmp_path: Path) -> Path:
    """A tiny database with the real schema (built by the same code as the real one)."""
    path = tmp_path / "sales.db"
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executescript(SCHEMA_SQL)
        conn.executemany(
            "INSERT INTO sales (invoice, stock_code, description, quantity, invoice_date, "
            "unit_price, line_total, customer_id, country, is_cancelled, is_product) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [r[:6] + (round(r[3] * r[5], 2),) + r[6:] for r in SAMPLE_ROWS],
        )
    return path
