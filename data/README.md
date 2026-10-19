# Data

Nothing in `data/raw/` is committed. Download the sources below and place them there.

## Company reports (RAG corpus)

Public investor-relations publications, downloaded by hand from each company's own
website. **The PDFs are not redistributed in this repository**: download them yourself and
put them in `data/raw/` under exactly these file names, because the file name is the
`source` used in every citation.

| File name in `data/raw/` | Company / report | Pages | Downloaded from | Downloaded on |
|---|---|---|---|---|
| `kering_urd_2025.pdf` | Kering, 2025 Universal Registration Document (filed with the AMF on 21 April 2026, ref. D.26-0266) | 516 | FILL_KERING | 2026-10-18 |
| `hermes_urd_2025.pdf` | Hermès International, 2025 Universal Registration Document | 535 | FILL_HERMES | 2026-10-18 |
| `lvmh_annual_report_2025.pdf` | LVMH, 2025 Annual Report | 184 | FILL_LVMH | 2026-10-18 |

Each company publishes these documents freely for shareholders and the public. This
project reads them locally and quotes short passages with their page numbers; it does not
republish them.

## Sales data

UCI Online Retail II (CC BY 4.0): https://archive.ics.uci.edu/dataset/502/online+retail+ii
Download the zip, extract `online_retail_II.xlsx` into `data/raw/` (downloaded on
2026-10-18, 45.6 MB), then run `python scripts/build_sales_db.py`. Cleaning decisions are
documented in `docs/data_notes.md`.

Citation: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D

## Evaluation sets

`data/eval/` holds small JSONL files that are committed (added in Stage 6).
