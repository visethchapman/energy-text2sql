"""Seed a cloud Postgres (Neon) with the ERCOT demand + Houston weather tables.

Copies eia.demand, noaa.stations, noaa.daily_weather from the local Postgres
(DATABASE_URL) into the cloud DB (NEON_DATABASE_URL) using streaming CSV COPY,
which is version-agnostic (local PG16 -> Neon PG18) and needs no pg_dump.

Only the tables the hosted demo needs — no pgvector / docs schema (the deployed
demo runs baseline + multi-agent, not vector RAG).

Usage:
    uv run python scripts/seed_neon.py
"""
from __future__ import annotations

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

SRC = os.environ["DATABASE_URL"]
DST = os.environ["NEON_DATABASE_URL"]

DDL = """
CREATE SCHEMA IF NOT EXISTS eia;
CREATE SCHEMA IF NOT EXISTS noaa;

CREATE TABLE IF NOT EXISTS eia.demand (
    region       TEXT             NOT NULL,
    period       TIMESTAMPTZ      NOT NULL,
    value        DOUBLE PRECISION,
    value_units  TEXT,
    PRIMARY KEY (region, period)
);
CREATE INDEX IF NOT EXISTS idx_demand_period ON eia.demand(period);

CREATE TABLE IF NOT EXISTS noaa.stations (
    station_id           TEXT PRIMARY KEY,
    name                 TEXT,
    state                TEXT,
    latitude             DOUBLE PRECISION,
    longitude            DOUBLE PRECISION,
    elevation_m          DOUBLE PRECISION,
    nearest_eia_region   TEXT
);

CREATE TABLE IF NOT EXISTS noaa.daily_weather (
    station_id  TEXT             NOT NULL REFERENCES noaa.stations(station_id),
    obs_date    DATE             NOT NULL,
    tmax_c      DOUBLE PRECISION,
    tmin_c      DOUBLE PRECISION,
    prcp_mm     DOUBLE PRECISION,
    awnd_ms     DOUBLE PRECISION,
    PRIMARY KEY (station_id, obs_date)
);

-- Schema cards drive the agent's prompt (load_schema reads these text columns).
-- The embedding VECTOR column is intentionally omitted: the hosted demo runs
-- baseline + multi-agent, not vector RAG, so pgvector isn't needed here.
CREATE SCHEMA IF NOT EXISTS docs;
CREATE TABLE IF NOT EXISTS docs.schema_cards (
    id               SERIAL PRIMARY KEY,
    schema_name      TEXT,
    table_name       TEXT,
    description      TEXT,
    columns_summary  TEXT,
    sample_values    TEXT,
    UNIQUE (schema_name, table_name)
);
"""

# (table, columns) — columns=None copies all; a list scopes the copy (skips
# the source-only embedding column on docs.schema_cards).
# Load order respects the FK (stations before daily_weather).
TABLES = [
    ("eia.demand", None),
    ("noaa.stations", None),
    ("noaa.daily_weather", None),
    ("docs.schema_cards",
     ["schema_name", "table_name", "description", "columns_summary", "sample_values"]),
]


def copy_table(src: psycopg.Connection, dst: psycopg.Connection,
               table: str, cols: list[str] | None) -> int:
    collist = f"({', '.join(cols)})" if cols else ""
    src_sql = (f"COPY (SELECT {', '.join(cols)} FROM {table}) TO STDOUT (FORMAT CSV)"
               if cols else f"COPY {table} TO STDOUT (FORMAT CSV)")
    with dst.cursor() as cur:
        cur.execute(f"TRUNCATE {table} CASCADE")
    with dst.cursor().copy(f"COPY {table} {collist} FROM STDIN (FORMAT CSV)") as cp_in:
        with src.cursor().copy(src_sql) as cp_out:
            for chunk in cp_out:
                cp_in.write(chunk)
    dst.commit()
    with dst.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        return cur.fetchone()[0]


def main() -> None:
    with psycopg.connect(SRC) as src, psycopg.connect(DST) as dst:
        with dst.cursor() as cur:
            cur.execute(DDL)
        dst.commit()
        print("schema ready on Neon")
        for t, cols in TABLES:
            n = copy_table(src, dst, t, cols)
            print(f"  {t:22} {n:>8,} rows")
    print("\nDone. Neon is seeded.")


if __name__ == "__main__":
    main()
