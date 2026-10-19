import math
from pathlib import Path

import pytest

from eval import metrics
from eval.judge import parse_judgement
from eval.report import END, START, update_readme


def test_unique_pages_keeps_first_occurrence() -> None:
    pages = [("k", 1), ("k", 1), ("h", 2), ("k", 1), ("k", 3)]
    assert metrics.unique_pages(pages) == [("k", 1), ("h", 2), ("k", 3)]


def test_rank_recall_and_mrr() -> None:
    ranked = [("k", 5), ("k", 13), ("k", 47)]
    rank = metrics.first_hit_rank(ranked, {("k", 47), ("k", 13)})
    assert rank == 2
    assert metrics.recall_at_k(rank, 1) == 0.0
    assert metrics.recall_at_k(rank, 3) == 1.0
    assert metrics.reciprocal_rank(rank) == 0.5
    assert metrics.first_hit_rank(ranked, {("h", 1)}) is None
    assert metrics.reciprocal_rank(None) == 0.0


def test_results_match_exact_and_tolerant() -> None:
    assert metrics.results_match([(9471580.96,)], [(9471580.96,)])
    assert metrics.results_match([(3,)], [(3.0,)])  # int vs float
    assert metrics.results_match([(519.756,)], [(519.76,)])  # rounding to cents
    assert not metrics.results_match([(8532,)], [(3510,)])


def test_results_match_numbers_stored_as_text() -> None:
    assert metrics.results_match([(12,)], [("12",)])  # CAST(strftime(...)) vs strftime(...)
    assert metrics.results_match([("2011-11",)], [("2011-11",)])
    assert not metrics.results_match([("13",)], [("12",)])


def test_results_match_allows_extra_columns() -> None:
    pred = [("2011-11", 1_161_817.38)]  # month and revenue
    gold = [("2011-11",)]  # month only
    assert metrics.results_match(pred, gold)


def test_results_match_order_only_when_asked() -> None:
    a = [("2009", 1.0), ("2010", 2.0)]
    b = [("2010", 2.0), ("2009", 1.0)]
    assert metrics.results_match(a, b, ordered=False)
    assert not metrics.results_match(a, b, ordered=True)


def test_results_match_row_count_must_agree() -> None:
    assert not metrics.results_match([(1,), (2,)], [(1,)])
    assert metrics.results_match([], [])


def test_numbers_in_text() -> None:
    assert metrics.numbers_in("€14,675 million, -13%, 3.00 per share, 11.1%") == [
        "14675",
        "13",
        "3",
        "11.1",
    ]


def test_number_match_uses_headline_number() -> None:
    gold = "€14,675 million (-13% as reported)"
    assert metrics.number_match("Revenue was €14,675 million [k p.13].", gold)
    assert not metrics.number_match("Revenue was €16,874 million.", gold)
    assert metrics.number_match("Luca de Meo", "Luca de Meo") is None  # no number


@pytest.mark.parametrize(
    ("answer", "gold", "expected"),
    [
        ("Revenue was €16 billion [h p.8].", "€16,002 million", True),
        ("Net profit was €4.5 billion.", "€4,524 million", True),
        ("Operating income: €6.6bn.", "€6,569 million", True),
        ("Revenue was €17 billion.", "€16,002 million", False),  # 6% off
        ("Free cash flow was over €11 billion.", "€11,333 million", False),  # 3% off
        ("The margin was 11 billion.", "11.1%", False),  # small numbers: exact only
    ],
)
def test_number_match_accepts_billions_within_rounding(
    answer: str, gold: str, expected: bool
) -> None:
    assert metrics.number_match(answer, gold) is expected


def test_name_match() -> None:
    assert metrics.name_match("The CEO is Luca de Meo [k p.7].", "Luca de Meo")
    assert not metrics.name_match("The CEO is François-Henri Pinault.", "Luca de Meo")


def test_refusal_detection() -> None:
    assert metrics.is_refusal("anything", "out_of_scope")
    assert metrics.is_refusal("I could not find this in the documents.", "documents")
    assert not metrics.is_refusal("Revenue was 5.", "documents")


def test_percentile_nearest_rank() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert metrics.percentile(values, 50) == 3.0
    assert metrics.percentile(values, 95) == 100.0
    assert math.isnan(metrics.percentile([], 95))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"correct": 1, "faithful": 0}', {"correct": 1, "faithful": 0}),
        ('Sure! {"correct": 0, "faithful": 1} done', {"correct": 0, "faithful": 1}),
        ('{"correct": 1}', None),
        ("no json", None),
    ],
)
def test_parse_judgement(text: str, expected: dict | None) -> None:
    assert parse_judgement(text) == expected


def test_update_readme_replaces_only_the_results_block(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{START}\nold\n{END}\n\n## After\n", encoding="utf-8")
    update_readme("new table", readme)
    text = readme.read_text(encoding="utf-8")
    assert "new table" in text and "old" not in text
    assert text.startswith("# Title") and text.endswith("## After\n")


def test_verify_search_pattern_does_not_match_inside_numbers() -> None:
    import re

    from eval.verify import search_pattern

    p = search_pattern("€3.00 per share")
    assert re.search(p, "dividend €3.00 | €1.00")
    assert not re.search(p, "revenue 1,631 and 13% and 3,675")
    assert re.search(search_pattern("€18.00 per share"), "dividend per share of €18. After")
    p = search_pattern("92%")
    assert re.search(p, "made up 92% of sales")
    assert not re.search(p, "revenue 5,992 million")
    p = search_pattern("€14,675 million")
    assert re.search(p, "Revenue 14,675 16,874") and re.search(p, "total 14675")
