"""Safety guard for LLM-generated SQL. Four independent layers ("defence in depth"):

1. Parse with sqlglot and allow exactly ONE statement that is a SELECT (or a UNION of
   SELECTs). INSERT, UPDATE, DELETE, DROP, CREATE, PRAGMA, ATTACH, VACUUM, ... and multiple
   statements ("SELECT 1; DROP TABLE sales") are rejected. Anything sqlglot cannot parse
   as a known statement is rejected too.
2. Allow-list of tables (`sales`, `sales_clean`, plus CTE names defined in the query) and a
   deny-list of SQLite functions that touch files or extensions.
3. Enforce a LIMIT: added when missing, lowered when larger than the maximum.
4. Execute on a READ-ONLY connection (`mode=ro`, plus `PRAGMA query_only`), with a time
   limit enforced by a progress handler that aborts long queries.

Layer 4 alone would already block writes; layers 1-3 give clear error messages that are
fed back to the LLM so it can fix its query.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

ALLOWED_TABLES = frozenset({"sales", "sales_clean"})
DENIED_FUNCTIONS = frozenset(
    {
        "load_extension",
        "readfile",
        "writefile",
        "edit",
        "fts3_tokenizer",
        "sqlite_compileoption_get",
    }
)
_FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command,
    exp.Pragma, exp.Attach, exp.Detach, exp.Merge, exp.Transaction, exp.Commit, exp.Rollback,
)  # fmt: skip


class SQLGuardError(ValueError):
    """The query was rejected before execution. The message explains why."""


@dataclass(frozen=True)
class QueryResult:
    sql: str  # the SQL actually executed (after validation and LIMIT enforcement)
    columns: list[str]
    rows: list[tuple[object, ...]]
    seconds: float


def validate(sql: str, max_rows: int) -> str:
    """Return a safe version of `sql` (with an enforced LIMIT) or raise SQLGuardError."""
    sql = sql.strip()
    if not sql:
        raise SQLGuardError("Empty query.")
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except ParseError as e:
        raise SQLGuardError(f"Could not parse the SQL: {str(e).splitlines()[0]}") from e
    if len(statements) != 1:
        raise SQLGuardError(f"Exactly one statement is allowed, got {len(statements)}.")
    tree = statements[0]

    if not isinstance(tree, exp.Select | exp.Union):
        raise SQLGuardError(f"Only SELECT queries are allowed, got {tree.key.upper()}.")
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SQLGuardError(f"Forbidden operation inside the query: {node.key.upper()}.")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if table.args.get("db") or table.args.get("catalog"):
            raise SQLGuardError(f"Schema-qualified tables are not allowed: {table.sql()}.")
        if name not in ALLOWED_TABLES and name not in cte_names:
            raise SQLGuardError(
                f"Unknown table '{table.name}'. Allowed: {', '.join(sorted(ALLOWED_TABLES))}."
            )
    for func in tree.find_all(exp.Anonymous):
        if func.name.lower() in DENIED_FUNCTIONS:
            raise SQLGuardError(f"Function '{func.name}' is not allowed.")

    limit = tree.args.get("limit")
    current = None
    if limit is not None and isinstance(limit.expression, exp.Literal):
        try:
            current = int(limit.expression.this)
        except ValueError:
            current = None
    if limit is not None and current is None:
        raise SQLGuardError("LIMIT must be a plain number.")
    if current is None or current > max_rows:
        tree = tree.limit(max_rows)
    return tree.sql(dialect="sqlite")


def connect_read_only(db_path: Path) -> sqlite3.Connection:
    """Open the database read-only. Fails if the file does not exist (never creates it)."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} not found. Build it first: python scripts/build_sales_db.py"
        )
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def execute(conn: sqlite3.Connection, safe_sql: str, timeout_s: float) -> QueryResult:
    """Run validated SQL with a time limit. Raises sqlite3.Error on failure or timeout."""
    deadline = time.perf_counter() + timeout_s
    # Called every 10,000 SQLite VM steps; returning True aborts the query.
    conn.set_progress_handler(lambda: time.perf_counter() > deadline, 10_000)
    t0 = time.perf_counter()
    try:
        cursor = conn.execute(safe_sql)
        rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e):
            raise sqlite3.OperationalError(
                f"Query took longer than {timeout_s:g} s and was stopped."
            ) from e
        raise
    finally:
        conn.set_progress_handler(None, 0)
    columns = [d[0] for d in cursor.description or []]
    return QueryResult(safe_sql, columns, rows, time.perf_counter() - t0)


def run_safe_query(
    db_path: Path, sql: str, max_rows: int = 100, timeout_s: float = 5.0
) -> QueryResult:
    """Validate, then execute on a read-only connection."""
    safe_sql = validate(sql, max_rows)
    with closing(connect_read_only(db_path)) as conn:  # always closed (Windows file locks)
        return execute(conn, safe_sql, timeout_s)
