"""Metric functions. Pure functions, unit-tested in tests/test_eval_metrics.py."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from statistics import median

FLOAT_TOLERANCE = 0.01  # values equal within 1 cent / 0.01 unit


# ---------- Retrieval (page level) ----------


def unique_pages(pages: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
    """Keep the first occurrence of each (source, page): several chunks can share a page."""
    seen: set[tuple[str, int]] = set()
    out = []
    for p in pages:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def first_hit_rank(ranked: Sequence[tuple[str, int]], gold: set[tuple[str, int]]) -> int | None:
    """1-based rank of the first gold page in the ranked list, or None."""
    for rank, page in enumerate(ranked, start=1):
        if page in gold:
            return rank
    return None


def recall_at_k(rank: int | None, k: int) -> float:
    """1.0 if a gold page appears in the top k (any gold page counts: the fact is there)."""
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


# ---------- SQL execution accuracy ----------


def _norm(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return round(float(value), 2)  # 3 and 3.0 compare equal
    if isinstance(value, str):
        return value.strip()
    return value


def _same(a: object, b: object) -> bool:
    if isinstance(a, int | float) and isinstance(b, int | float):
        return math.isclose(float(a), float(b), abs_tol=FLOAT_TOLERANCE)
    return _norm(a) == _norm(b)


def _column(rows: Sequence[Sequence[object]], j: int) -> list[object]:
    return [_norm(r[j]) for r in rows]


def _columns_equal(pred: list[object], gold: list[object], ordered: bool) -> bool:
    if len(pred) != len(gold):
        return False
    if ordered:
        return all(_same(p, g) for p, g in zip(pred, gold, strict=True))
    # unordered: compare as multisets (floats already rounded to 2 decimals)
    return Counter(map(repr, pred)) == Counter(map(repr, gold))


def results_match(
    pred_rows: Sequence[Sequence[object]],
    gold_rows: Sequence[Sequence[object]],
    ordered: bool = False,
) -> bool:
    """Execution match: same number of rows, and every gold column appears (with the same
    values) among the predicted columns. Extra predicted columns are allowed, e.g. a query
    returning (month, revenue) when the gold query returns only (month). Row order matters
    only when the question asks for an ordering (`ordered`)."""
    if len(pred_rows) != len(gold_rows):
        return False
    if not gold_rows:
        return True
    n_pred = len(pred_rows[0])
    used: set[int] = set()
    for j in range(len(gold_rows[0])):
        gold_col = _column(gold_rows, j)
        match = next(
            (
                i
                for i in range(n_pred)
                if i not in used and _columns_equal(_column(pred_rows, i), gold_col, ordered)
            ),
            None,
        )
        if match is None:
            return False
        used.add(match)
    return True


# ---------- Answers ----------

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[str]:
    """Numbers in a text, normalized: '14,675' -> '14675', '3.00' -> '3', '11.1%' -> '11.1'."""
    out = []
    for m in _NUMBER.findall(text):
        n = m.replace(",", "")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.append(n)
    return out


def key_number(gold_answer: str) -> str | None:
    """The first number of the gold answer (the headline figure), or None for text answers."""
    nums = numbers_in(gold_answer)
    return nums[0] if nums else None


def number_match(answer: str, gold_answer: str) -> bool | None:
    """True if the gold answer's headline number appears in the answer; None if the gold
    answer has no number (e.g. a person's name), so the check does not apply."""
    key = key_number(gold_answer)
    if key is None:
        return None
    return key in numbers_in(answer)


def name_match(answer: str, gold_answer: str) -> bool:
    """For text answers (names): all words of the gold answer appear in the answer."""
    words = re.findall(r"\w+", gold_answer.lower())
    return all(w in answer.lower() for w in words)


def is_refusal(answer: str, route: str) -> bool:
    return route == "out_of_scope" or "could not find" in answer.lower()


# ---------- Latency ----------


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100). No interpolation: always an observed value."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "median_s": round(median(values), 2) if values else float("nan"),
        "p95_s": round(percentile(values, 95), 2) if values else float("nan"),
    }
