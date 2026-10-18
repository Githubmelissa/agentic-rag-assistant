"""Clean the UCI Online Retail II dataset and load it into SQLite.

Every cleaning decision is explained in docs/data_notes.md. Summary:
1. The two Excel sheets overlap (December 2010): invoices present in both are kept once.
2. Exact duplicate rows are dropped.
3. Cancellations (invoice starting with "C") are kept and flagged `is_cancelled = 1`.
4. Non-product lines (postage, fees, manual adjustments, vouchers, bad-debt adjustments,
   tests) are kept and flagged `is_product = 0`.
5. Missing customer IDs stay NULL (about 23% of lines; they are guest/unknown customers).
6. Descriptions are trimmed; `line_total = quantity * unit_price` is precomputed.
7. A view `sales_clean` keeps only real product sales: not cancelled, product lines,
   quantity > 0 and unit_price > 0. Most business questions should use it.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

# Stock codes that are not products (checked by listing every non-numeric code).
NON_PRODUCT_CODES = frozenset(
    {
        "POST", "DOT", "M", "C2", "C3", "D", "S", "BANK CHARGES", "ADJUST", "ADJUST2",
        "AMAZONFEE", "B", "CRUK", "TEST001", "TEST002", "GIFT",
    }
)  # fmt: skip
NON_PRODUCT_PREFIXES = ("GIFT_",)  # gift vouchers: gift_0001_10, gift_0001_20, ...

RAW_COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}

SCHEMA_SQL = """
CREATE TABLE sales (
    invoice       TEXT    NOT NULL,
    stock_code    TEXT    NOT NULL,
    description   TEXT,
    quantity      INTEGER NOT NULL,
    invoice_date  TEXT    NOT NULL,   -- 'YYYY-MM-DD HH:MM:SS'
    unit_price    REAL    NOT NULL,   -- pounds sterling (GBP)
    line_total    REAL    NOT NULL,   -- quantity * unit_price
    customer_id   INTEGER,            -- NULL for unknown customers
    country       TEXT    NOT NULL,
    is_cancelled  INTEGER NOT NULL,   -- 1 if the invoice starts with 'C'
    is_product    INTEGER NOT NULL    -- 0 for postage, fees, adjustments, vouchers
);
CREATE INDEX idx_sales_date ON sales(invoice_date);
CREATE INDEX idx_sales_country ON sales(country);
CREATE INDEX idx_sales_customer ON sales(customer_id);
-- No index on stock_code on purpose: SQLite used it for GROUP BY stock_code and read the
-- million rows in index order with random lookups (2.6 s vs 1.0 s for a plain scan).
CREATE VIEW sales_clean AS
    SELECT * FROM sales
    WHERE is_cancelled = 0 AND is_product = 1 AND quantity > 0 AND unit_price > 0;
"""


def is_product_code(code: str) -> bool:
    upper = code.strip().upper()
    return upper not in NON_PRODUCT_CODES and not upper.startswith(NON_PRODUCT_PREFIXES)


def read_raw(path: Path) -> list[pd.DataFrame]:
    """Read every sheet of the Excel file (one per year), as strings where IDs matter."""
    sheets = pd.read_excel(path, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    return list(sheets.values())


def clean(sheets: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the cleaning rules. Returns the clean table and counts for the report."""
    stats: dict[str, int] = {"raw_rows": sum(len(s) for s in sheets)}

    # 1. Remove invoices already seen in an earlier sheet (the sheets overlap).
    kept, seen = [], set()
    for sheet in sheets:
        overlap = sheet["Invoice"].isin(seen)
        kept.append(sheet.loc[~overlap])
        seen.update(sheet["Invoice"])
    df = pd.concat(kept, ignore_index=True)
    stats["overlap_rows_removed"] = stats["raw_rows"] - len(df)

    # 2. Exact duplicate rows.
    before = len(df)
    df = df.drop_duplicates(ignore_index=True)
    stats["duplicate_rows_removed"] = before - len(df)

    df = df.rename(columns=RAW_COLUMNS)[list(RAW_COLUMNS.values())]
    df["invoice"] = df["invoice"].str.strip()
    df["stock_code"] = df["stock_code"].str.strip()
    df["description"] = df["description"].astype("string").str.strip().replace("", pd.NA)
    df["country"] = df["country"].str.strip()
    df["customer_id"] = df["customer_id"].astype("Int64")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["line_total"] = (df["quantity"] * df["unit_price"]).round(2)
    df["is_cancelled"] = df["invoice"].str.upper().str.startswith("C").astype(int)
    df["is_product"] = df["stock_code"].map(is_product_code).astype(int)

    stats["final_rows"] = len(df)
    stats["cancelled_rows"] = int(df["is_cancelled"].sum())
    stats["non_product_rows"] = int((df["is_product"] == 0).sum())
    stats["missing_customer_rows"] = int(df["customer_id"].isna().sum())
    clean_mask = (
        (df["is_cancelled"] == 0)
        & (df["is_product"] == 1)
        & (df["quantity"] > 0)
        & (df["unit_price"] > 0)
    )
    stats["sales_clean_rows"] = int(clean_mask.sum())
    return df, stats


def write_db(df: pd.DataFrame, db_path: Path) -> None:
    """Write a fresh database (any existing file is replaced)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    # `with sqlite3.connect()` commits but does NOT close the file; Windows cannot rename an
    # open file, so the connection is closed explicitly before the swap below.
    with closing(sqlite3.connect(tmp)) as conn, conn:
        conn.executescript(SCHEMA_SQL)
        columns = [
            "invoice", "stock_code", "description", "quantity", "invoice_date",
            "unit_price", "line_total", "customer_id", "country", "is_cancelled", "is_product",
        ]  # fmt: skip
        rows = (
            tuple(None if pd.isna(v) else v for v in row)
            for row in df[columns].astype(object).itertuples(index=False, name=None)
        )
        conn.executemany(
            f"INSERT INTO sales ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
            rows,
        )
        conn.execute("ANALYZE")  # table statistics help SQLite choose good query plans
    tmp.replace(db_path)  # atomic swap: a half-written database never replaces a good one
