"""
Load EIA hourly electricity demand into Postgres.

Endpoint: https://api.eia.gov/v2/electricity/rto/region-data/data/
Filtered to type=D (demand). Free, but requires an API key:
    https://www.eia.gov/opendata/register.php

Usage:
    uv run python etl/01_load_eia.py
    uv run python etl/01_load_eia.py --region CISO --start 2020-01-01 --end 2024-12-31
    uv run python etl/01_load_eia.py --region ERCO CISO PJM NYIS
"""

import argparse
import os
import sys

import httpx
import psycopg
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
PAGE_SIZE = 5000


def fetch_page(api_key: str, region: str, start: str, end: str, offset: int) -> list[dict]:
    params = [
        ("api_key", api_key),
        ("frequency", "hourly"),
        ("data[]", "value"),
        ("facets[respondent][]", region),
        ("facets[type][]", "D"),
        ("start", start),
        ("end", end),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("offset", str(offset)),
        ("length", str(PAGE_SIZE)),
    ]
    r = httpx.get(API_URL, params=params, timeout=60.0)
    r.raise_for_status()
    return r.json().get("response", {}).get("data", [])


def _normalize_period(period: str) -> str:
    # EIA hourly returns "YYYY-MM-DDTHH" — pad to a full TIMESTAMPTZ string.
    return f"{period}:00:00+00:00" if len(period) == 13 else period


def load_region(conn: psycopg.Connection, api_key: str, region: str, start: str, end: str) -> int:
    total = 0
    offset = 0
    while True:
        print(f"  [{region}] fetch offset={offset}...", flush=True)
        rows = fetch_page(api_key, region, start, end, offset)
        if not rows:
            break

        values = [
            (r["respondent"], _normalize_period(r["period"]), r["value"], r.get("value-units"))
            for r in rows
            if r.get("value") is not None
        ]
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO eia.demand (region, period, value, value_units)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (region, period) DO UPDATE SET value = EXCLUDED.value
                """,
                values,
            )
        conn.commit()
        total += len(values)
        print(f"  [{region}] +{len(values)} rows (running total: {total})", flush=True)

        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region",
        nargs="+",
        default=["ERCO"],
        help="One or more EIA balancing authority codes (ERCO CISO PJM NYIS ...)",
    )
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()

    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        sys.exit(
            "EIA_API_KEY missing. Register (free, instant) at "
            "https://www.eia.gov/opendata/register.php and set it in .env"
        )

    db_url = os.environ["DATABASE_URL"]
    with psycopg.connect(db_url) as conn:
        grand_total = 0
        for region in args.region:
            grand_total += load_region(conn, api_key, region, args.start, args.end)
        print(f"\nDone. Inserted/updated {grand_total} rows across {len(args.region)} region(s).")


if __name__ == "__main__":
    main()
