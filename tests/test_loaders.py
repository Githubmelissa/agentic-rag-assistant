from pathlib import Path

import pytest

from assistant.ingest.loaders import Page, clean_block, load_pdf, load_pdfs, remove_repeated_blocks
from tests.conftest import make_pdf


def test_clean_block_joins_lines_and_hyphenation() -> None:
    assert clean_block("Revenue grew\nby 5%.\nThis is sustain-\nable.") == (
        "Revenue grew by 5%. This is sustainable."
    )


def test_load_pdf_keeps_source_and_page_numbers(tmp_path: Path) -> None:
    pdf = make_pdf(
        tmp_path / "acme_report.pdf",
        ["The first page describes the group strategy.", "The second page lists the brands."],
    )
    pages = load_pdf(pdf)
    assert [(p.source, p.page) for p in pages] == [("acme_report", 1), ("acme_report", 2)]
    assert "lists the brands" in pages[1].text


def test_empty_pages_are_skipped(tmp_path: Path) -> None:
    pdf = make_pdf(
        tmp_path / "r.pdf",
        ["Some real content on this first page.", "", "More real content on the third page."],
    )
    assert [p.page for p in load_pdf(pdf)] == [1, 3]


def test_repeated_footer_is_removed(tmp_path: Path) -> None:
    topics = ["leather goods", "jewelry", "eyewear", "fragrances", "watches", "logistics"]
    texts = [f"This page explains the {t} business in detail." for t in topics]
    pdf = make_pdf(tmp_path / "r.pdf", texts, footer="ACME 2025 Annual Report")
    pages = load_pdf(pdf)
    assert all("ACME 2025 Annual Report" not in p.text for p in pages)
    assert "fragrances business" in pages[3].text


def test_remove_repeated_blocks_ignores_page_numbers() -> None:
    pages = [Page("s", i, "", [f"Unique content {chr(65 + i)}", f"Report {i}"]) for i in range(6)]
    cleaned = remove_repeated_blocks(pages)
    assert all(p.blocks == [f"Unique content {chr(65 + p.page)}"] for p in cleaned)


def test_load_pdfs_raises_without_pdfs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_pdfs(tmp_path)
