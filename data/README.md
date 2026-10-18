# Data

Nothing in `data/raw/` is committed. Download the sources below and place them there.

## Company reports (RAG corpus)

| File name in `data/raw/` | Company / report | Source URL | Downloaded on | Terms of use checked |
|---|---|---|---|---|
| `kering_urd_2025.pdf` | Kering, 2025 Universal Registration Document (516 pages) | TBD | 2026-10-18 | TBD |
| `hermes_urd_2025.pdf` | Hermès International, 2025 Universal Registration Document (535 pages) | TBD | 2026-10-18 | TBD |
| `lvmh_annual_report_2025.pdf` | LVMH, 2025 Annual Report (184 pages) | TBD | 2026-10-18 | TBD |

Downloaded from each company's investor-relations website. The PDFs are not redistributed
in this repository.

## Sales data

UCI Online Retail II (CC BY 4.0): https://archive.ics.uci.edu/dataset/502/online+retail+ii
Download the zip, extract `online_retail_II.xlsx` into `data/raw/` (downloaded on
2026-10-18, 45.6 MB), then run `python scripts/build_sales_db.py`. Cleaning decisions are
documented in `docs/data_notes.md`.

Citation: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D

## Evaluation sets

`data/eval/` holds small JSONL files that are committed (added in Stage 6).
