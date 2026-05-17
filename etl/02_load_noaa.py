"""
Load NOAA GHCN-Daily weather observations for selected stations into Postgres.

Source:
    https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{STATION_ID}.csv

No API key needed. Each station CSV is wide-format: one row per date with
columns TMAX, TMIN, PRCP, AWND, ... (values in tenths — e.g. TMAX=271 means 27.1 deg C).

Default stations are picked to be near major US load zones:
    USW00012960  Houston Hobby        TX  -> ERCO  (Texas)
    USW00023174  Los Angeles Intl     CA  -> CISO  (California)
    USW00013739  Philadelphia Intl    PA  -> PJM
    USW00094728  New York JFK         NY  -> NYIS

Usage:
    uv run python etl/02_load_noaa.py
"""

import csv
import io
import os
import sys

import httpx
import psycopg
from dotenv import load_dotenv

load_dotenv()

BASE_URL = (
    "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/"
    "access/{station_id}.csv"
)

# (station_id, name, state, latitude, longitude, elevation_m, nearest_eia_region)
STATIONS = [
    ("USW00012960", "Houston Hobby",     "TX", 29.6453,  -95.2825,  14.0, "ERCO"),
    ("USW00023174", "Los Angeles Intl",  "CA", 33.9381, -118.3889,  29.6, "CISO"),
    ("USW00013739", "Philadelphia Intl", "PA", 39.8683,  -75.2367,   3.0, "PJM"),
    ("USW00094728", "New York JFK",      "NY", 40.6386,  -73.7622,   3.4, "NYIS"),
]

ELEMENTS = ("TMAX", "TMIN", "PRCP", "AWND")  # all reported in tenths


def parse_tenths(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw) / 10.0
    except ValueError:
        return None


def parse_station_csv(content: str, since_year: int) -> list[tuple]:
    """Return list of (date, tmax, tmin, prcp, awnd) tuples."""
    reader = csv.DictReader(io.StringIO(content))
    rows = []
    for row in reader:
        date = row.get("DATE")
        if not date or int(date[:4]) < since_year:
            continue
        rows.append((
            date,
            parse_tenths(row.get("TMAX")),
            parse_tenths(row.get("TMIN")),
            parse_tenths(row.get("PRCP")),
            parse_tenths(row.get("AWND")),
        ))
    return rows


def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL missing (check your .env)")

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO noaa.stations
                  (station_id, name, state, latitude, longitude, elevation_m, nearest_eia_region)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (station_id) DO UPDATE SET
                  name = EXCLUDED.name,
                  nearest_eia_region = EXCLUDED.nearest_eia_region
                """,
                STATIONS,
            )
        conn.commit()

        total = 0
        for station_id, name, *_ in STATIONS:
            url = BASE_URL.format(station_id=station_id)
            print(f"Downloading {station_id} ({name})...", flush=True)
            r = httpx.get(url, timeout=180.0)
            r.raise_for_status()
            rows = parse_station_csv(r.text, since_year=2020)
            insert_rows = [(station_id, *row) for row in rows]

            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO noaa.daily_weather
                      (station_id, obs_date, tmax_c, tmin_c, prcp_mm, awnd_ms)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (station_id, obs_date) DO UPDATE SET
                      tmax_c = EXCLUDED.tmax_c,
                      tmin_c = EXCLUDED.tmin_c,
                      prcp_mm = EXCLUDED.prcp_mm,
                      awnd_ms = EXCLUDED.awnd_ms
                    """,
                    insert_rows,
                )
            conn.commit()
            print(f"  +{len(insert_rows)} days", flush=True)
            total += len(insert_rows)

        print(f"\nDone. Inserted/updated {total} weather rows across {len(STATIONS)} stations.")


if __name__ == "__main__":
    main()
