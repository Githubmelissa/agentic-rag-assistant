import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from assistant.sales_data import clean, is_product_code, write_db


def sheet(rows: list[tuple]) -> pd.DataFrame:
    cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price"]
    cols += ["Customer ID", "Country"]
    df = pd.DataFrame(rows, columns=cols)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


ROW_A = ("100", "22423", " CAKESTAND ", 2, "2010-12-01 09:00", 12.75, 12345.0, "France")
ROW_B = ("101", "POST", "POSTAGE", 1, "2010-12-02 09:00", 18.0, None, "France")
ROW_C = ("C102", "22423", "CAKESTAND", -1, "2010-12-03 09:00", 12.75, 12345.0, "France")


def test_product_codes() -> None:
    assert is_product_code("85123A") and is_product_code("DCGS0058")
    assert not is_product_code("POST") and not is_product_code("gift_0001_20")
    assert not is_product_code("BANK CHARGES")


def test_clean_rules() -> None:
    sheet1 = sheet([ROW_A, ROW_A, ROW_B])  # ROW_A duplicated
    sheet2 = sheet([ROW_A, ROW_C])  # invoice 100 overlaps with sheet 1
    df, stats = clean([sheet1, sheet2])
    assert stats["overlap_rows_removed"] == 1
    assert stats["duplicate_rows_removed"] == 1
    assert stats["final_rows"] == 3
    assert stats["sales_clean_rows"] == 1  # only the cake stand sale
    row = df.set_index("invoice").loc["100"]
    assert row["description"] == "CAKESTAND"  # trimmed
    assert row["line_total"] == 25.5
    assert row["invoice_date"] == "2010-12-01 09:00:00"
    assert df.set_index("invoice").loc["C102", "is_cancelled"] == 1
    assert df.set_index("invoice").loc["101", "is_product"] == 0
    assert pd.isna(df.set_index("invoice").loc["101", "customer_id"])


def test_write_db_creates_table_and_view(tmp_path: Path) -> None:
    df, _ = clean([sheet([ROW_A, ROW_B, ROW_C])])
    db = tmp_path / "sales.db"
    write_db(df, db)
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sales").fetchone() == (3,)
        assert conn.execute("SELECT COUNT(*) FROM sales_clean").fetchone() == (1,)
        assert conn.execute("SELECT customer_id FROM sales WHERE invoice = '101'").fetchone() == (
            None,
        )
