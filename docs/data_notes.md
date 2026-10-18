# Sales data: cleaning decisions

Source: **UCI Online Retail II** (https://archive.ics.uci.edu/dataset/502/online+retail+ii),
CC BY 4.0. One Excel file, two sheets ("Year 2009-2010", "Year 2010-2011"): transactions of
a UK-based online gift shop from 2009-12-01 to 2011-12-09. Built by
`python scripts/build_sales_db.py` into `data/sales.db` (not committed). The script prints
the row counts below and saves them in `data/sales_db_stats.json`.

## Row counts (from the build run on 2026-10-18)

| step | rows |
|---|---:|
| raw rows, both sheets | 1,067,371 |
| removed: overlap between sheets | 22,523 |
| removed: exact duplicate rows | 11,812 |
| **rows in `sales`** | **1,033,036** |
| of which cancellation lines | 19,104 |
| of which non-product lines | 5,801 |
| of which missing customer ID | 235,151 |
| **rows in `sales_clean`** | **1,003,357** |

## Decisions

1. **Overlap between sheets.** Both sheets contain 2010-12-01 to 2010-12-09: 1,088 invoices
   (22,523 rows) appear in both, identical. An invoice already seen in an earlier sheet is
   dropped from later sheets.
2. **Exact duplicate rows are dropped** (11,812). They could in theory be the same item
   scanned twice on one invoice, but identical rows down to the minute are far more likely
   to be export duplicates; this is the common treatment for this dataset. Effect: totals
   are slightly lower than a raw sum.
3. **Cancellations are kept and flagged** (`is_cancelled = 1`, invoice starting with "C",
   negative quantity), so questions about returns can be answered. They are excluded from
   `sales_clean`. Note: the original order of a cancelled sale stays in `sales_clean`
   (e.g. one order of 80,995 units cancelled the same day), so `sales_clean` measures
   **gross** sales; net sales = gross + cancellation lines.
4. **Non-product lines are kept and flagged** (`is_product = 0`): postage (POST, DOT, C2,
   C3), manual entries (M), discounts (D), samples (S), bank charges, Amazon fees, CRUK
   commission, manual adjustments (ADJUST, ADJUST2), bad-debt adjustments (B, invoices
   starting with "A"), gift vouchers (gift_*), test products (TEST001/2). The list was made
   by inspecting all 63 non-numeric stock codes; codes like DCGS0058 are real products.
5. **Missing customer IDs stay NULL** (23% of lines): sales to unidentified customers are
   still sales. Customer-level questions must exclude NULLs (stated in the SQL prompt).
6. **Zero or negative prices and non-positive quantities** outside cancellations (stock
   write-offs, damaged goods, price-0 lines) stay in `sales` but not in `sales_clean`.
7. **Text**: descriptions trimmed (213,000 had stray spaces); empty becomes NULL.
   Dates stored as ISO text `YYYY-MM-DD HH:MM:SS`, so string comparison sorts correctly.
8. **`line_total = quantity * unit_price`**, rounded to pence, precomputed so the LLM does
   not have to.

## Known limitations

- Currency is assumed to be GBP (UK retailer); the dataset does not state it per line.
- 2009 contains only December, and 2011 stops on December 9: yearly totals are not
  comparable without that caveat.
