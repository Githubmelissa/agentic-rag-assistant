"""Text-to-SQL tool: question -> SQL (LLM) -> guard -> read-only execution -> answer (LLM).

If the query is rejected by the guard or fails in SQLite, the error message is sent back
to the LLM, which gets up to `max_retries` more attempts (default 2). The retry loop is
evaluated with and without retries in the Stage 6 ablation.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from assistant.agent.prompts import (
    SQL_ANSWER_SYSTEM,
    SQL_ANSWER_USER,
    SQL_RETRY_USER,
    SQL_SYSTEM,
    SQL_USER,
)
from assistant.llm import LLM, LLMResponse
from assistant.tools.sql_guard import QueryResult, SQLGuardError, run_safe_query

SCHEMA_PATH = Path(__file__).with_name("schema.md")
ROWS_IN_PROMPT = 20  # rows shown to the LLM when it writes the answer
_CODE_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@cache
def load_schema() -> str:
    return SCHEMA_PATH.read_text(encoding="utf-8")


def extract_sql(text: str) -> str:
    """Take the SQL out of the model's reply (code block if present, else the whole text)."""
    match = _CODE_BLOCK.search(text)
    return (match.group(1) if match else text).strip().rstrip(";").strip()


@dataclass
class Attempt:
    sql: str
    error: str | None  # None if the query ran
    rejected_by_guard: bool = False


@dataclass
class SQLAnswer:
    question: str
    answer: str
    sql: str | None  # the SQL that ran successfully, if any
    result: QueryResult | None
    attempts: list[Attempt]
    llm_calls: list[LLMResponse] = field(repr=False)

    @property
    def succeeded(self) -> bool:
        return self.result is not None

    @property
    def retries(self) -> int:
        return max(0, len(self.attempts) - 1)

    @property
    def guard_rejections(self) -> int:
        return sum(a.rejected_by_guard for a in self.attempts)


def format_rows(result: QueryResult, limit: int = ROWS_IN_PROMPT) -> str:
    return "\n".join(" | ".join(str(v) for v in row) for row in result.rows[:limit])


def answer_sql_question(
    question: str,
    llm: LLM,
    db_path: Path,
    max_retries: int = 2,
    max_rows: int = 100,
    timeout_s: float = 5.0,
) -> SQLAnswer:
    system = SQL_SYSTEM.format(schema=load_schema())
    user = SQL_USER.format(question=question)
    attempts: list[Attempt] = []
    calls: list[LLMResponse] = []
    result: QueryResult | None = None

    for _ in range(max_retries + 1):
        response = llm.complete(system, user)
        calls.append(response)
        sql = extract_sql(response.text)
        try:
            result = run_safe_query(db_path, sql, max_rows=max_rows, timeout_s=timeout_s)
            attempts.append(Attempt(result.sql, None))
            break
        except SQLGuardError as e:
            attempts.append(Attempt(sql, str(e), rejected_by_guard=True))
        except sqlite3.Error as e:
            attempts.append(Attempt(sql, f"SQLite error: {e}"))
        user = SQL_RETRY_USER.format(question=question, sql=sql, error=attempts[-1].error)

    if result is None:
        return SQLAnswer(
            question,
            "I could not answer this from the sales data: the query failed. "
            f"Last error: {attempts[-1].error}",
            None,
            None,
            attempts,
            calls,
        )

    truncated = f", first {ROWS_IN_PROMPT} shown" if len(result.rows) > ROWS_IN_PROMPT else ""
    response = llm.complete(
        SQL_ANSWER_SYSTEM,
        SQL_ANSWER_USER.format(
            question=question,
            sql=result.sql,
            columns=", ".join(result.columns),
            n_rows=len(result.rows),
            truncated=truncated,
            rows=format_rows(result) or "(no rows)",
        ),
    )
    calls.append(response)
    return SQLAnswer(question, response.text, result.sql, result, attempts, calls)
