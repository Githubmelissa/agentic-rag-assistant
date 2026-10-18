"""Load PDF reports page by page, keeping `{source, page}` metadata.

Text is extracted per *block* (PyMuPDF groups lines into paragraph-like blocks), so the
chunker can respect paragraph boundaries. Headers and footers that repeat on most pages
(e.g. "KERING — 2025 Universal Registration Document") are removed, because they add
noise to both keyword and embedding search without carrying information.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# A block is "repeated boilerplate" if its digit-free text appears on this share of pages.
REPEATED_BLOCK_THRESHOLD = 0.3
# Pages with less text than this (after cleaning) are skipped: covers, photo pages.
MIN_PAGE_CHARS = 30


@dataclass
class Page:
    """One PDF page. `page` is the 1-based PDF page number shown by PDF viewers."""

    source: str  # file name without extension, e.g. "kering_urd_2025"
    page: int
    page_label: str  # printed page number if the PDF defines one, else ""
    blocks: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(self.blocks)


def clean_block(text: str) -> str:
    """Join a block's lines into one paragraph and normalise whitespace."""
    text = text.replace("­", "")  # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # word broken across lines: "sustain-\nable"
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _boilerplate_key(block: str) -> str:
    """Key used to detect repeated headers/footers: lowercase, digits removed."""
    return re.sub(r"\d+", "", block.lower()).strip()


def remove_repeated_blocks(
    pages: list[Page], threshold: float = REPEATED_BLOCK_THRESHOLD
) -> list[Page]:
    """Drop blocks that repeat on at least `threshold` of the document's pages."""
    if len(pages) < 5:  # too few pages to tell boilerplate from content
        return pages
    counts = Counter(key for p in pages for key in {_boilerplate_key(b) for b in p.blocks})
    min_count = threshold * len(pages)
    repeated = {key for key, n in counts.items() if n >= min_count}
    for p in pages:
        p.blocks = [b for b in p.blocks if _boilerplate_key(b) not in repeated]
    return pages


def load_pdf(path: Path) -> list[Page]:
    """Return the non-empty pages of a PDF, cleaned, with page metadata."""
    pages: list[Page] = []
    with pymupdf.open(path) as doc:
        for index, pdf_page in enumerate(doc):
            blocks = [
                clean_block(b[4])
                for b in pdf_page.get_text("blocks", sort=True)
                if b[6] == 0  # 0 = text block, 1 = image block
            ]
            pages.append(
                Page(
                    source=path.stem,
                    page=index + 1,
                    page_label=pdf_page.get_label() or "",
                    blocks=[b for b in blocks if b],
                )
            )
    pages = remove_repeated_blocks(pages)
    return [p for p in pages if len(p.text) >= MIN_PAGE_CHARS]


def load_pdfs(input_dir: Path) -> list[Page]:
    """Load every PDF in `input_dir` (sorted by name, so results are reproducible)."""
    paths = sorted(input_dir.glob("*.pdf"))
    if not paths:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")
    return [page for path in paths for page in load_pdf(path)]
