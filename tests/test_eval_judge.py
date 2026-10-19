from pathlib import Path
from typing import Any

from eval import common, judge


def test_highlight_marks_the_gold_number() -> None:
    text = "Revenue 14,675 million, up from 16,874."
    assert judge.highlight(text, "€14,675 million") == (
        "Revenue >>>14,675<<< million, up from 16,874."
    )


def test_highlight_falls_back_to_the_first_word_for_names() -> None:
    assert ">>>Luca<<<" in judge.highlight("Luca de Meo, CEO", "Luca de Meo")


def write_sheet(tmp_path: Path, answer: str, pages: list[str]) -> str:
    rows: dict[str, Any] = {"doc_01": {"answer": answer, "cited_texts": pages}}
    questions: dict[str, Any] = {
        "doc_01": {"question": "What was the revenue?", "gold_answer": "€14,675 million"}
    }
    judge.write_review_sheet(["doc_01"], rows, questions)
    return (tmp_path / "judge_review.md").read_text(encoding="utf-8")


def test_review_sheet_shows_question_answer_and_cited_pages(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(common, "RAW_DIR", tmp_path)
    text = write_sheet(tmp_path, "Revenue was €14,675 million.", ["[src p.13] Revenue 14,675 m"])
    assert "## 1. doc_01" in text
    assert "What was the revenue?" in text
    assert "faithful=" not in text  # the judge's own scores stay out: grading is blind
    assert ">>>14,675<<<" in text  # the gold number is marked in the cited page


def test_review_sheet_says_when_nothing_was_cited(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(common, "RAW_DIR", tmp_path)
    assert "**Cited pages:** none" in write_sheet(tmp_path, "Revenue was high.", [])


CSV_HEADER = "id,judge_correct,judge_faithful,human_correct,human_faithful\n"


def test_reads_a_csv_saved_by_a_windows_spreadsheet(tmp_path: Path) -> None:
    """WPS and Excel save CSV in the local code page: '€' becomes the single byte 0x80."""
    path = tmp_path / "check.csv"
    text = CSV_HEADER + "€ doc_01,1,1,1,0\n"
    path.write_bytes(text.encode("cp1252"))  # "€" -> the single byte 0x80
    assert judge.read_csv_any_encoding(path)[0]["human_correct"] == "1"


def test_reads_a_utf8_csv_with_a_byte_order_mark(tmp_path: Path) -> None:
    path = tmp_path / "check.csv"
    path.write_text(CSV_HEADER + "€14675 million,1,1,0,1\n", encoding="utf-8-sig")
    rows = judge.read_csv_any_encoding(path)
    assert rows[0]["id"] == "€14675 million" and rows[0]["human_faithful"] == "1"
