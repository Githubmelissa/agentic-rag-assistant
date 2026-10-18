"""Build data/sales.db from the UCI Online Retail II Excel file.

Usage (from the project root):
    python scripts/build_sales_db.py
    python scripts/build_sales_db.py --input data/raw/online_retail_II.xlsx
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from assistant.config import get_settings
from assistant.sales_data import clean, read_raw, write_db


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=settings.raw_dir / "online_retail_II.xlsx")
    parser.add_argument("--output", type=Path, default=settings.sales_db_path)
    args = parser.parse_args()

    t0 = time.perf_counter()
    print(f"Reading {args.input} (this takes about a minute)...")
    df, stats = clean(read_raw(args.input))
    print(f"Writing {args.output}...")
    write_db(df, args.output)
    stats_path = args.output.with_name("sales_db_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    for key, value in stats.items():
        print(f"  {key:<24}{value:>10,}")
    print(f"Done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
