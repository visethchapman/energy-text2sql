"""
Seed schema cards into docs.schema_cards. These are the human-readable
descriptions the planning agent will retrieve via vector search later (Day 5).
For Day 1 we just persist them; embeddings come once the agent is online.

Usage:
    uv run python etl/03_schema_docs.py
"""

import json
import os
import sys

import psycopg
from dotenv import load_dotenv

load_dotenv()

SCHEMA_CARDS: list[dict] = [
    {
        "schema_name": "eia",
        "table_name": "demand",
        "description": (
            "Hourly electricity demand in megawatt-hours, reported by US balancing "
            "authorities (BAs) to the EIA. One row per BA region per hour (UTC)."
        ),
        "columns_summary": (
            "region (BA code: ERCO=Texas, CISO=California, PJM=Mid-Atlantic, NYIS=New York); "
            "period (hour, UTC); "
            "value (demand in MWh); "
            "value_units."
        ),
        "sample_values": (
            "region in ('ERCO','CISO','PJM','NYIS'); "
            "period 2020-01-01 onward; "
            "value typically 10,000-80,000 MWh per hour."
        ),
    },
    {
        "schema_name": "noaa",
        "table_name": "daily_weather",
        "description": (
            "Daily weather observations from NOAA GHCN-Daily for stations near major "
            "US load zones. One row per station per day."
        ),
        "columns_summary": (
            "station_id (NOAA GHCN id); obs_date; "
            "tmax_c, tmin_c (daily max/min temperature in deg C); "
            "prcp_mm (precipitation, mm); "
            "awnd_ms (avg wind speed, m/s)."
        ),
        "sample_values": (
            "stations: USW00012960 (Houston), USW00023174 (Los Angeles), "
            "USW00013739 (Philadelphia), USW00094728 (NYC JFK)."
        ),
    },
    {
        "schema_name": "noaa",
        "table_name": "stations",
        "description": (
            "Weather station metadata. The nearest_eia_region column maps each "
            "station to a balancing authority for join queries."
        ),
        "columns_summary": (
            "station_id, name, state, latitude, longitude, elevation_m, nearest_eia_region."
        ),
        "sample_values": "nearest_eia_region in ('ERCO','CISO','PJM','NYIS').",
    },
]


def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL missing (check your .env)")

    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        for card in SCHEMA_CARDS:
            cur.execute(
                """
                INSERT INTO docs.schema_cards
                  (schema_name, table_name, description, columns_summary, sample_values)
                VALUES (%(schema_name)s, %(table_name)s, %(description)s,
                        %(columns_summary)s, %(sample_values)s)
                ON CONFLICT (schema_name, table_name) DO UPDATE SET
                  description     = EXCLUDED.description,
                  columns_summary = EXCLUDED.columns_summary,
                  sample_values   = EXCLUDED.sample_values
                """,
                card,
            )
        conn.commit()

    os.makedirs("data", exist_ok=True)
    out = "data/schema_cards.json"
    with open(out, "w") as f:
        json.dump(SCHEMA_CARDS, f, indent=2)
    print(f"Seeded {len(SCHEMA_CARDS)} schema cards into docs.schema_cards and wrote {out}.")


if __name__ == "__main__":
    main()
