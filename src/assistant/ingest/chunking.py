"""Split pages into overlapping chunks without cutting sentences.

Algorithm (a simple "recursive" splitter):
1. A page is split into blocks (paragraphs), each block into sentences.
2. A sentence longer than the chunk size is split into word windows (rare: tables, lists).
3. Sentences are packed greedily into chunks of at most `chunk_size` tokens.
4. Each new chunk starts with the last sentences of the previous one, up to
   `chunk_overlap` tokens, so a fact spanning a boundary appears whole in one chunk.

Chunks never cross page boundaries, so every chunk has exactly one page to cite.
Token counting is injected (`count_tokens`) so tests can use a simple word counter while
real runs use the embedding model's own tokenizer.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass

from assistant.ingest.loaders import Page

TokenCounter = Callable[[str], int]

# Sentence end: . ! ? followed by whitespace and an upper-case letter, digit, quote or bracket.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“‘(\[])")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<source>:p<page>:c<index>", stable across runs
    source: str
    page: int
    page_label: str
    text: str
    n_tokens: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def count_words(text: str) -> int:
    """Simple token counter (whitespace words). Used in tests and as a fallback."""
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def _split_long_sentence(sentence: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Split one over-long sentence into word windows of at most `max_tokens` tokens."""
    words = sentence.split()
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and count(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _units(page: Page, chunk_size: int, count: TokenCounter) -> list[tuple[str, int]]:
    """Return (text, n_tokens) units: sentences, or word windows for over-long sentences."""
    units: list[tuple[str, int]] = []
    for block in page.blocks:
        for sentence in split_sentences(block):
            n = count(sentence)
            if n <= chunk_size:
                units.append((sentence, n))
            else:
                units.extend(
                    (piece, count(piece))
                    for piece in _split_long_sentence(sentence, chunk_size, count)
                )
    return units


def chunk_page(
    page: Page, chunk_size: int, chunk_overlap: int, count: TokenCounter = count_words
) -> list[Chunk]:
    """Split one page into chunks of at most `chunk_size` tokens."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    groups: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    current_tokens = 0
    for unit in _units(page, chunk_size, count):
        if current and current_tokens + unit[1] > chunk_size:
            groups.append(current)
            # Carry trailing units of the finished chunk as overlap.
            overlap: list[tuple[str, int]] = []
            overlap_tokens = 0
            for prev in reversed(current):
                if overlap_tokens + prev[1] > chunk_overlap:
                    break
                overlap.insert(0, prev)
                overlap_tokens += prev[1]
            # Keep the overlap only if the new unit still fits with it.
            if overlap_tokens + unit[1] > chunk_size:
                overlap, overlap_tokens = [], 0
            current, current_tokens = overlap, overlap_tokens
        current.append(unit)
        current_tokens += unit[1]
    if current:
        groups.append(current)

    chunks = []
    for i, group in enumerate(groups):
        text = " ".join(u[0] for u in group)
        chunks.append(
            Chunk(
                chunk_id=f"{page.source}:p{page.page}:c{i}",
                source=page.source,
                page=page.page,
                page_label=page.page_label,
                text=text,
                n_tokens=count(text),
            )
        )
    return chunks


def chunk_pages(
    pages: list[Page], chunk_size: int, chunk_overlap: int, count: TokenCounter = count_words
) -> list[Chunk]:
    return [c for p in pages for c in chunk_page(p, chunk_size, chunk_overlap, count)]
