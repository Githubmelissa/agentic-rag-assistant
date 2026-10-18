import sqlite3
from pathlib import Path

import pytest

from assistant.tools.sql_guard import (
    SQLGuardError,
    connect_read_only,
    execute,
    run_safe_query,
    validate,
)

ATTACKS = [
    "DROP TABLE sales",
    "DELETE FROM sales",
    "UPDATE sales SET unit_price = 0",
    "INSERT INTO sales (invoice) VALUES ('x')",
    "REPLACE INTO sales VALUES (1)",
    "SELECT 1; DROP TABLE sales",  # stacked statements
    "SELECT * FROM sales; -- harmless?\nDELETE FROM sales",
    "WITH x AS (SELECT 1) DELETE FROM sales",  # write hidden behind a CTE
    "PRAGMA table_info(sales)",
    "ATTACH DATABASE 'evil.db' AS evil",
    "CREATE TABLE copy AS SELECT * FROM sales",
    "VACUUM",
    "SELECT load_extension('evil.dll')",
    "SELECT * FROM sqlite_master",  # table outside the allow-list
    "SELECT * FROM main.sales",  # schema-qualified
    "SELECT * FROM sales LIMIT (SELECT 1000000)",
    "",
    "this is not sql",
]


@pytest.mark.parametrize("sql", ATTACKS)
def test_attacks_are_rejected(sql: str) -> None:
    with pytest.raises(SQLGuardError):
        validate(sql, max_rows=100)


def test_limit_added_when_missing() -> None:
    assert validate("SELECT country FROM sales_clean", 100).endswith("LIMIT 100")


def test_limit_capped_when_too_large() -> None:
    assert validate("SELECT country FROM sales LIMIT 5000", 100).endswith("LIMIT 100")


def test_small_limit_kept() -> None:
    assert validate("SELECT country FROM sales LIMIT 3", 100).endswith("LIMIT 3")


def test_cte_and_union_allowed() -> None:
    validate("WITH t AS (SELECT country FROM sales_clean) SELECT country FROM t", 10)
    validate("SELECT country FROM sales UNION SELECT country FROM sales_clean", 10)


def test_trailing_semicolon_allowed() -> None:
    assert validate("SELECT 1 FROM sales;", 10) == "SELECT 1 FROM sales LIMIT 10"


def test_run_safe_query(sales_db: Path) -> None:
    result = run_safe_query(sales_db, "SELECT COUNT(*) AS n FROM sales_clean")
    assert result.columns == ["n"]
    assert result.rows == [(3,)]  # cancellation and postage excluded by the view


def test_connection_is_read_only_even_without_guard(sales_db: Path) -> None:
    conn = connect_read_only(sales_db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM sales")
    conn.close()


def test_missing_database_is_not_created(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        connect_read_only(tmp_path / "nope.db")
    assert not (tmp_path / "nope.db").exists()


def test_timeout_stops_runaway_query(sales_db: Path) -> None:
    infinite = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    )
    conn = connect_read_only(sales_db)
    with pytest.raises(sqlite3.OperationalError, match="longer than"):
        execute(conn, infinite, timeout_s=0.2)
    conn.close()
