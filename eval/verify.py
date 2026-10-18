"""Verify the drafted evaluation questions, one by one (rule 4 of the brief).

    python -m eval.verify doc       # 40 document questions
    python -m eval.verify sql       # 20 SQL questions
    python -m eval.verify mixed     # 10 questions needing both sources
    python -m eval.verify oos       # 10 out-of-scope questions

For a document question, the tool prints the text of each gold page around the answer,
so you can check it (and open the PDF at that page when in doubt). For a SQL question,
it runs the gold SQL and prints the result. Then type:
    y = correct, mark verified        e = edit the gold answer / SQL
    p = edit the gold pages           x = mark as NOT verified (wrong or unclear)
    s = skip for now                  q = save and quit
Your choices are saved after every question, so you can stop and continue any time.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import textwrap
from functools import cache

import pymupdf

from assistant.config import get_settings
from assistant.tools.sql_guard import SQLGuardError, run_safe_query
from eval import common, metrics

SNIPPET = 300


@cache
def page_text(source: str, page: int) -> str:
    path = get_settings().raw_dir / f"{source}.pdf"
    with pymupdf.open(path) as doc:
        return re.sub(r"\s+", " ", doc[page - 1].get_text())


def search_pattern(gold_answer: str) -> str:
    """Regex for the answer's headline number ('14675' also matches '14,675'), or its first
    word for a name."""
    key = metrics.key_number(gold_answer)
    if key and "." in key:
        return re.escape(key)
    if key:
        return (
            ",?".join(re.escape(d) for d in key.split(".")[0]) if len(key) > 3 else re.escape(key)
        )
    return re.escape(re.findall(r"\w+", gold_answer)[0])


def show_pages(item: dict) -> None:
    pattern = search_pattern(item["gold_answer"])
    for page in item["gold_pages"]:
        text = page_text(item["gold_source"], page)
        m = re.search(pattern, text, re.I)
        start = max(0, m.start() - SNIPPET) if m else 0
        snippet = text[start : start + 2 * SNIPPET]
        if m:
            snippet = re.sub(pattern, lambda x: f">>>{x.group(0)}<<<", snippet, flags=re.I)
        print(f"\n  --- {item['gold_source']} PDF p.{page} ---")
        print(textwrap.indent(textwrap.fill(snippet, 100), "  "))


def show_sql(sql: str) -> None:
    print("\n  Gold SQL:\n" + textwrap.indent(sql, "    "))
    try:
        r = run_safe_query(get_settings().sales_db_path, sql, 20, 60.0)
    except (SQLGuardError, sqlite3.Error) as e:
        print(f"  ERROR: {e}")
        return
    print("  Result: " + " | ".join(r.columns))
    for row in r.rows[:10]:
        print("    " + " | ".join(str(v) for v in row))


def ask(prompt: str, choices: str) -> str:
    while True:
        answer = input(prompt).strip().lower()
        if answer in choices:
            return answer


def review(item: dict, kind: str) -> str:
    print("\n" + "=" * 100)
    print(f"{item['id']}  [{kind}]  {item['question']}")
    if kind in ("doc", "mixed"):
        print(f"  Gold answer: {item['gold_answer']}   (source: {item['gold_source']})")
        print(f"  Gold pages:  {item['gold_pages']}")
        show_pages(item)
    if kind in ("sql", "mixed"):
        show_sql(item["gold_sql"])
    if kind == "oos":
        print("  Expected: the assistant refuses (not answerable from the reports or the data).")
    return ask("\n  [y]es verified / [e]dit / [p]ages / [x] not ok / [s]kip / [q]uit: ", "yepxsq")


def edit(item: dict, kind: str, what: str) -> None:
    if what == "p":
        raw = input(f"  New gold pages (comma separated) [{item['gold_pages']}]: ").strip()
        if raw:
            item["gold_pages"] = [int(x) for x in re.findall(r"\d+", raw)]
    elif kind == "sql" or (kind == "mixed" and input("  Edit [a]nswer or [s]ql? ") == "s"):
        sql = input("  New gold SQL (one line): ").strip()
        if sql:
            item["gold_sql"] = sql
    else:
        answer = input("  New gold answer: ").strip()
        if answer:
            item["gold_answer"] = answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("set", choices=list(common.SETS))
    parser.add_argument("--all", action="store_true", help="also review verified items")
    a = parser.parse_args()
    path = common.EVAL_DIR / common.SETS[a.set]
    items = common.read_jsonl(path)
    todo = [i for i in items if a.all or not i.get("verified")]
    print(f"{len(todo)} item(s) to review in {path.name}.")
    for item in todo:
        while True:
            choice = review(item, a.set)
            if choice in "ep":
                edit(item, a.set, choice)
                continue  # show again with the change
            break
        if choice == "q":
            break
        if choice == "y":
            item["verified"] = True
        elif choice == "x":
            item["verified"] = False
            item["note"] = input("  Why (short note)? ").strip()
        common.write_jsonl(path, items)  # save after every item
    common.write_jsonl(path, items)
    done = sum(i.get("verified") is True for i in items)
    print(f"\nSaved. {done}/{len(items)} verified in {path.name}.")


if __name__ == "__main__":
    main()
