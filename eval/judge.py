"""LLM-as-judge for document answers, plus the human check of 20 judgements.

For each document question answered (not "not found"), the judge sees the question, the
gold answer, the system's answer and the full text of the pages it cited, and gives two
binary scores with a fixed rubric:
- correct:  the answer states the same fact as the gold answer (numbers must match);
- faithful: every fact in the answer is supported by the cited pages.

Known weaknesses (reported in the README): the judge is the same small local model as
the system; it can be lenient, follow the answer's wording, or misread numbers. That is
why 20 judgements are checked by hand: `python -m eval.judge` writes
results/judge_human_check.csv; fill the two `human_*` columns with 1 or 0, then run
`python -m eval.report`, which computes the agreement rate.

    python -m eval.judge
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from statistics import mean
from typing import Any

from assistant.llm import LLM
from eval import common, setup

JUDGE_SYSTEM = """You grade an answer produced by a question-answering system. Reply with \\
JSON only: {"correct": 0 or 1, "faithful": 0 or 1}.

Rubric:
- correct = 1 if the answer states the same fact as the reference answer. Numbers must \\
match (rounding such as "16 billion" for "16,002 million" is fine; a different number \\
is wrong). Extra correct details are fine. Otherwise 0.
- faithful = 1 if every fact in the answer is stated in the cited pages. If the cited \\
pages do not contain a fact of the answer, or there are no cited pages, faithful = 0."""

JUDGE_USER = """Question: {question}
Reference answer: {gold}
System answer: {answer}

Cited pages:
{pages}

JSON:"""

PAGE_CHARS = 2500  # per cited page, to keep the prompt within the context window
HUMAN_SAMPLE = 20
SEED = 42
_JSON = re.compile(r"\{[^{}]*\}")


def parse_judgement(text: str) -> dict[str, int] | None:
    match = _JSON.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return {k: int(bool(int(data[k]))) for k in ("correct", "faithful")}
    except (KeyError, ValueError, TypeError):
        return None


def judge_row(row: dict[str, Any], q: dict[str, Any], llm: LLM) -> dict[str, Any]:
    pages = "\n\n".join(t[:PAGE_CHARS] for t in row["cited_texts"]) or "(none)"
    r = llm.complete(
        JUDGE_SYSTEM,
        JUDGE_USER.format(
            question=q["question"], gold=q["gold_answer"], answer=row["answer"], pages=pages
        ),
    )
    verdict = parse_judgement(r.text)
    return {
        "id": row["id"],
        "judge_correct": verdict["correct"] if verdict else None,
        "judge_faithful": verdict["faithful"] if verdict else None,
        "raw": r.text[:200],
    }


def run(mock: bool = False, include_unverified: bool = False) -> dict:
    questions = {q["id"]: q for q in common.load_set("doc", include_unverified)}
    agent_rows = [
        r for r in common.Cache("agent_mock" if mock else "agent") if r["id"] in questions
    ]
    answered = [r for r in agent_rows if r.get("found")]
    if not answered:
        raise SystemExit("No answered document questions yet: run eval.run_agent first.")
    llm = setup.llm(mock)
    name = "judge_mock" if mock else "judge"
    cache = common.Cache(name)
    for row in answered:
        if row["id"] not in cache:
            cache.add(judge_row(row, questions[row["id"]], llm))
    judged = cache.select([r["id"] for r in answered])
    valid = [j for j in judged if j["judge_correct"] is not None]
    payload = {
        "description": "LLM-as-judge on answered document questions (same local model).",
        "llm": setup.llm_name(mock),
        "include_unverified": include_unverified,
        "mock": mock,
        "n_answered": len(answered),
        "n_judged": len(valid),
        "unparsed": len(judged) - len(valid),
        "correct_rate": round(mean(j["judge_correct"] for j in valid), 3) if valid else None,
        "faithful_rate": round(mean(j["judge_faithful"] for j in valid), 3) if valid else None,
    }
    common.save_result(name, payload)
    if not mock:
        write_human_sheet(valid, {r["id"]: r for r in answered}, questions)
    print(payload)
    return payload


def write_human_sheet(
    judged: list[dict[str, Any]], rows: dict[str, Any], questions: dict[str, Any]
) -> None:
    path = common.RESULTS_DIR / "judge_human_check.csv"
    if path.exists():
        print(f"{path} already exists: not overwritten (it may contain your labels).")
        return
    sample = random.Random(SEED).sample(judged, min(HUMAN_SAMPLE, len(judged)))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["id", "question", "gold_answer", "system_answer", "cited_pages",
             "judge_correct", "judge_faithful", "human_correct", "human_faithful"]
        )  # fmt: skip
        for j in sample:
            q, r = questions[j["id"]], rows[j["id"]]
            w.writerow(
                [j["id"], q["question"], q["gold_answer"], r["answer"], "; ".join(r["citations"]),
                 j["judge_correct"], j["judge_faithful"], "", ""]
            )  # fmt: skip
    print(f"Wrote {path}: fill human_correct / human_faithful with 1 or 0.")


def human_agreement() -> dict[str, Any] | None:
    """Agreement between the judge and the human labels in judge_human_check.csv."""
    path = common.RESULTS_DIR / "judge_human_check.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["human_correct"] in ("0", "1")]
    if not rows:
        return None
    pairs = [(r[f"judge_{k}"], r[f"human_{k}"]) for r in rows for k in ("correct", "faithful")]
    return {
        "n_checked": len(rows),
        "agreement_correct": round(mean(r["judge_correct"] == r["human_correct"] for r in rows), 3),
        "agreement_faithful": round(
            mean(r["judge_faithful"] == r["human_faithful"] for r in rows), 3
        ),
        "agreement_overall": round(mean(a == b for a, b in pairs), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--include-unverified", action="store_true")
    a = parser.parse_args()
    run(a.mock, a.include_unverified)


if __name__ == "__main__":
    main()
