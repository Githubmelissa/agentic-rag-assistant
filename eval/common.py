"""Shared helpers: dataset loading, result caching, and output paths.

Results are cached per question in `results/raw/*.jsonl`, so a long run interrupted on a
laptop can be resumed: already-computed questions are skipped.

Only items marked `"verified": true` are evaluated (rule 4 of the project brief). The
`--include-unverified` flag exists for dry runs; results produced that way are marked
and `report.py` refuses to put them in the README.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"

SETS = {
    "doc": "doc_questions.jsonl",
    "sql": "sql_questions.jsonl",
    "mixed": "mixed_questions.jsonl",
    "oos": "oos_questions.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_set(name: str, include_unverified: bool = False) -> list[dict[str, Any]]:
    rows = read_jsonl(EVAL_DIR / SETS[name])
    return rows if include_unverified else [r for r in rows if r.get("verified") is True]


def set_sizes() -> dict[str, dict[str, int]]:
    sizes = {}
    for name in SETS:
        rows = read_jsonl(EVAL_DIR / SETS[name])
        sizes[name] = {"total": len(rows), "verified": sum(r.get("verified") is True for r in rows)}
    return sizes


class Cache:
    """Append-only JSONL cache keyed by question id."""

    def __init__(self, name: str) -> None:
        self.path = RAW_DIR / f"{name}.jsonl"
        self.rows = {r["id"]: r for r in read_jsonl(self.path)}

    def __contains__(self, key: str) -> bool:
        return key in self.rows

    def add(self, row: dict[str, Any]) -> None:
        self.rows[row["id"]] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def select(self, ids: list[str]) -> list[dict[str, Any]]:
        return [self.rows[i] for i in ids if i in self.rows]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.rows.values())


def save_result(name: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_result(name: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
