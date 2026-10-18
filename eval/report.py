"""Turn results/*.json into Markdown tables, one chart, and the README's results section.

Every number in the README comes from these files (rule 1 of the brief). Results computed
with --include-unverified or with the mock LLM are never written into the README.

    python -m eval.report
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from eval import common
from eval.judge import human_agreement

README = common.ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
CHART = common.RESULTS_DIR / "recall_at_5_by_mode.png"


def pct(x: float | None) -> str:
    return "TBD" if x is None else f"{x * 100:.0f}%"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def qualifier(r: dict[str, Any]) -> str:
    """'verified' for publishable results; a loud warning for dry runs."""
    return "UNVERIFIED, dry run only" if r.get("include_unverified") else "verified"


def retrieval_section(r: dict[str, Any] | None) -> str:
    if not r:
        return "### Retrieval\n\nTBD (run `python -m eval.run_retrieval`)."
    rows = [
        [x["chunk_size"], x["mode"], pct(x["recall@3"]), pct(x["recall@5"]),
         pct(x["recall@10"]), f"{x['mrr']:.3f}", f"{x['median_latency_s']:.2f} s"]
        for x in r["rows"]
    ]  # fmt: skip
    return (
        f"### Retrieval ({r['n_questions']} {qualifier(r)} document questions, page level)\n\n"
        + table(
            ["chunk size", "mode", "recall@3", "recall@5", "recall@10", "MRR", "median latency"],
            rows,
        )  # fmt: skip
        + f"\n\n![Recall@5 by retrieval mode](results/{CHART.name})"
    )


def sql_section(r: dict[str, Any] | None) -> str:
    if not r:
        return "### Text-to-SQL\n\nTBD (run `python -m eval.run_sql`)."
    rows = [
        [x["max_retries"], pct(x["execution_accuracy"]), pct(x["query_ran"]),
         pct(x["guard_rejection_rate"]), x["questions_with_retry"], f"{x['median_s']} s"]
        for x in r["rows"]
    ]  # fmt: skip
    return (
        f"### Text-to-SQL ({r['n_questions']} {qualifier(r)} questions, {r['llm']})\n\n"
        + table(["max retries", "execution accuracy", "query ran", "guard rejections "
                 "(share of attempts)", "questions with a retry", "median latency"], rows)
    )  # fmt: skip


def routing_section(r: dict[str, Any] | None) -> str:
    if not r:
        return "### Routing\n\nTBD (run `python -m eval.run_routing`)."
    rows = [[k, v["n"], pct(v["accuracy"])] for k, v in r["per_route"].items()]
    return (
        f"### Routing ({r['n']} {qualifier(r)} questions, {r['llm']})\n\n"
        f"Overall accuracy: **{pct(r['accuracy'])}** "
        f"(unparseable router replies: {r['unparsed_replies']}).\n\n"
        + table(["expected route", "n", "accuracy"], rows)
    )


def agent_section(a: dict[str, Any] | None, j: dict[str, Any] | None) -> str:
    if not a:
        return "### End-to-end answers\n\nTBD (run `python -m eval.run_agent`)."
    d, m, o = a["documents"], a["mixed"], a["out_of_scope"]
    rows = [
        ["Document questions: gold number/name in the answer", d["n"], pct(d["answer_match"])],
        ["Document questions: cites a gold page", d["n"], pct(d["cites_gold_page"])],
        ['Document questions: answered "not found"', d["n"], pct(d["said_not_found"])],
        ["Mixed questions: routed to both", m["n"], pct(m["routed_both"])],
        ["Mixed questions: reports part correct", m["n"], pct(m["reports_part_match"])],
        ["Mixed questions: SQL part correct", m["n"], pct(m["sql_part_match"])],
        ["Out-of-scope questions: refused", o["n"], pct(o["refusal_rate"])],
    ]
    if j:
        rows += [
            ["LLM judge: correct (answered questions)", j["n_judged"], pct(j["correct_rate"])],
            ["LLM judge: faithful to cited pages", j["n_judged"], pct(j["faithful_rate"])],
        ]
    h = human_agreement()
    judge_note = (
        f"Hand check of {h['n_checked']} judgements: agreement {pct(h['agreement_overall'])} "
        f"(correctness {pct(h['agreement_correct'])}, faithfulness "
        f"{pct(h['agreement_faithful'])})."
        if h
        else "Hand check of the LLM judge: TBD (fill results/judge_human_check.csv)."
    )
    lat = [[k, v["n"], f"{v['median_s']} s", f"{v['p95_s']} s"]
           for k, v in a["latency_by_route"].items()]  # fmt: skip
    t = a["tokens_per_query"]
    return (
        f"### End-to-end answers ({qualifier(a)} questions, {a['llm']})\n\n"
        + table(["metric", "n", "value"], rows)
        + "\n\nThe LLM judge is the same small local model as the system; it is imperfect "
        "(it can be lenient or misread numbers). "
        + judge_note
        + "\n\n#### Latency per route (laptop CPU, local LLM)\n\n"
        + table(["route", "n", "median", "p95"], lat)
        + f"\n\nTokens per query (input + output, all LLM calls): median {t['median']}, "
        f"mean {t['mean']} (n = {t['n']})."
    )


def make_chart(r: dict[str, Any] | None) -> None:
    if not r:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    default_cs = min(r["rows"], key=lambda x: abs(x["chunk_size"] - 500))["chunk_size"]
    rows = [x for x in r["rows"] if x["chunk_size"] == default_cs]
    labels = [x["mode"] for x in rows]
    values = [x["recall@5"] * 100 for x in rows]
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    bars = ax.bar(labels, values, color="#2a78d6", width=0.55)
    for bar, v in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center",
                fontsize=9, color="#0b0b0b")  # fmt: skip
    ax.set_ylim(0, 105)
    ax.set_ylabel("recall@5 (%)", color="#52514e")
    ax.set_title(
        f"Recall@5 by retrieval mode (chunk size {default_cs}, n = {r['n_questions']})"
        + (" - DRY RUN, unverified" if r.get("include_unverified") else ""),
        fontsize=10, color="#0b0b0b", loc="left",
    )  # fmt: skip
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c9c8c3")
    ax.spines["bottom"].set_color("#c9c8c3")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.yaxis.grid(True, color="#ebeae6", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(CHART)
    plt.close(fig)


def build() -> tuple[str, bool]:
    results = {n: common.load_result(n) for n in ("retrieval", "sql", "routing", "agent", "judge")}
    unverified = [n for n, r in results.items() if r and r.get("include_unverified")]
    sizes = common.set_sizes()
    size_rows = [[k, v["total"], v["verified"]] for k, v in sizes.items()]
    body = "\n\n".join(
        [
            "Only questions verified by hand are used; unverified items are excluded.\n\n"
            + table(["set", "drafted", "verified (used)"], size_rows),
            retrieval_section(results["retrieval"]),
            sql_section(results["sql"]),
            routing_section(results["routing"]),
            agent_section(results["agent"], results["judge"]),
        ]
    )
    make_chart(results["retrieval"])
    return body, not unverified


def update_readme(body: str, path: Path = README) -> None:
    text = path.read_text(encoding="utf-8")
    block = f"{START}\n{body}\n{END}"
    if START in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    else:
        text += f"\n## Evaluation results\n\n{block}\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    body, publishable = build()
    out = common.RESULTS_DIR / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    if publishable:
        update_readme(body)
        print(f"Updated the results section of {README}")
    else:
        print("Some results used unverified questions: README NOT updated.")


if __name__ == "__main__":
    main()
