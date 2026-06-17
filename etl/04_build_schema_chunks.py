"""
v2: build column-level schema chunks + embeddings.

One row per column across the user-facing tables. Each chunk gets
embedded with BAAI/bge-large-en-v1.5 (1024-dim, matches the
docs.schema_chunks.embedding column) and stored in Postgres for
vector-routed retrieval at agent runtime.

Run once after db/02_schema_chunks.sql has created the table:

    docker exec -i energy-postgres psql -U energy -d energy \
        < db/02_schema_chunks.sql
    uv run python etl/04_build_schema_chunks.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

MODEL_NAME = "BAAI/bge-large-en-v1.5"


# One entry per column. Descriptions are intentionally explicit about the
# things the synthesis prompt needs to know (timezone, units, semantic
# meaning) — the retriever pulls these directly into the agent's context.
CHUNKS: list[dict] = [
    # ---------- eia.demand ----------
    {
        "schema_name": "eia", "table_name": "demand", "column_name": "region",
        "data_type": "TEXT",
        "description": "US balancing-authority (BA) code identifying the regional grid operator.",
        "sample_values": "ERCO (Texas), CISO (California), PJM (Mid-Atlantic), NYIS (New York). Only ERCO is currently loaded.",
    },
    {
        "schema_name": "eia", "table_name": "demand", "column_name": "period",
        "data_type": "TIMESTAMPTZ",
        "description": "Hour-aligned timestamp of the demand observation in UTC. Each row represents one hour of demand.",
        "sample_values": "2020-01-01 00:00:00+00 through 2024-12-31 23:00:00+00.",
    },
    {
        "schema_name": "eia", "table_name": "demand", "column_name": "value",
        "data_type": "DOUBLE PRECISION",
        "description": "Electricity demand in megawatt-hours (MWh) for the given region during the given hour.",
        "sample_values": "Typical 30,000-80,000 MWh; ERCOT 2024 peak was 85,544 MWh on 2024-08-20.",
    },
    {
        "schema_name": "eia", "table_name": "demand", "column_name": "value_units",
        "data_type": "TEXT",
        "description": "Unit string for the value column (informational; always 'megawatthours' for loaded data).",
        "sample_values": "megawatthours",
    },
    # ---------- noaa.stations ----------
    {
        "schema_name": "noaa", "table_name": "stations", "column_name": "station_id",
        "data_type": "TEXT",
        "description": "NOAA GHCN-Daily station identifier. Primary key.",
        "sample_values": "USW00012960 (Houston Hobby), USW00023174 (Los Angeles Intl), USW00013739 (Philadelphia Intl), USW00094728 (New York JFK).",
    },
    {
        "schema_name": "noaa", "table_name": "stations", "column_name": "name",
        "data_type": "TEXT",
        "description": "Human-readable station name, usually an airport or city.",
        "sample_values": "Houston Hobby, Los Angeles Intl, Philadelphia Intl, New York JFK.",
    },
    {
        "schema_name": "noaa", "table_name": "stations", "column_name": "state",
        "data_type": "TEXT",
        "description": "US two-letter state code where the station is located.",
        "sample_values": "TX, CA, PA, NY.",
    },
    {
        "schema_name": "noaa", "table_name": "stations", "column_name": "latitude",
        "data_type": "DOUBLE PRECISION",
        "description": "Station latitude in decimal degrees.",
        "sample_values": "29.6453 (Houston), 33.9381 (LA), 39.8683 (Philly), 40.6386 (NYC).",
    },
    {
        "schema_name": "noaa", "table_name": "stations", "column_name": "longitude",
        "data_type": "DOUBLE PRECISION",
        "description": "Station longitude in decimal degrees.",
        "sample_values": "-95.2825 (Houston), -118.3889 (LA), -75.2367 (Philly), -73.7622 (NYC).",
    },
    {
        "schema_name": "noaa", "table_name": "stations", "column_name": "elevation_m",
        "data_type": "DOUBLE PRECISION",
        "description": "Station elevation above sea level in meters.",
        "sample_values": "3.0 to 29.6.",
    },
    {
        "schema_name": "noaa", "table_name": "stations", "column_name": "nearest_eia_region",
        "data_type": "TEXT",
        "description": "EIA balancing-authority region whose load this station's weather best represents. Used as the join key between weather and demand.",
        "sample_values": "ERCO (Houston), CISO (LA), PJM (Philly), NYIS (NYC).",
    },
    # ---------- noaa.daily_weather ----------
    {
        "schema_name": "noaa", "table_name": "daily_weather", "column_name": "station_id",
        "data_type": "TEXT",
        "description": "Foreign key to noaa.stations.station_id identifying which station produced this observation.",
        "sample_values": "USW00012960, USW00023174, USW00013739, USW00094728.",
    },
    {
        "schema_name": "noaa", "table_name": "daily_weather", "column_name": "obs_date",
        "data_type": "DATE",
        "description": "Calendar date of the observation in the station's LOCAL timezone (not UTC). To join with eia.demand.period, the UTC timestamp must be converted with AT TIME ZONE first (e.g. 'America/Chicago' for Houston/ERCO).",
        "sample_values": "2020-01-01 through 2026-05-13.",
    },
    {
        "schema_name": "noaa", "table_name": "daily_weather", "column_name": "tmax_c",
        "data_type": "DOUBLE PRECISION",
        "description": "Daily maximum temperature in degrees Celsius.",
        "sample_values": "Houston 2024 peak: 38.9C on Aug 20. Typical winter range: 5-20C.",
    },
    {
        "schema_name": "noaa", "table_name": "daily_weather", "column_name": "tmin_c",
        "data_type": "DOUBLE PRECISION",
        "description": "Daily minimum temperature in degrees Celsius.",
        "sample_values": "Houston freeze low: -10.5C on 2021-02-16. Typical summer range: 20-28C.",
    },
    {
        "schema_name": "noaa", "table_name": "daily_weather", "column_name": "prcp_mm",
        "data_type": "DOUBLE PRECISION",
        "description": "Daily total precipitation in millimeters.",
        "sample_values": "0 on dry days; up to ~200 mm on heavy storms.",
    },
    {
        "schema_name": "noaa", "table_name": "daily_weather", "column_name": "awnd_ms",
        "data_type": "DOUBLE PRECISION",
        "description": "Daily average wind speed in meters per second.",
        "sample_values": "Typical 1-10 m/s.",
    },
]


def _chunk_text(c: dict) -> str:
    """Text representation that gets embedded — fully qualified column + description + samples."""
    return (
        f"{c['schema_name']}.{c['table_name']}.{c['column_name']} ({c['data_type']}): "
        f"{c['description']} Examples: {c['sample_values']}"
    )


def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL missing (check your .env)")

    print(f"Loading {MODEL_NAME} (first run downloads ~1.3 GB)...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  → embedding dimension: {model.get_sentence_embedding_dimension()}")

    texts = [_chunk_text(c) for c in CHUNKS]
    print(f"\nEmbedding {len(texts)} chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    with psycopg.connect(db_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for c, emb in zip(CHUNKS, embeddings):
                cur.execute(
                    """
                    INSERT INTO docs.schema_chunks
                        (schema_name, table_name, column_name, data_type,
                         description, sample_values, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (schema_name, table_name, column_name) DO UPDATE SET
                        data_type     = EXCLUDED.data_type,
                        description   = EXCLUDED.description,
                        sample_values = EXCLUDED.sample_values,
                        embedding     = EXCLUDED.embedding
                    """,
                    (c["schema_name"], c["table_name"], c["column_name"],
                     c["data_type"], c["description"], c["sample_values"],
                     emb),
                )
        conn.commit()

    print(f"\nStored {len(CHUNKS)} chunks in docs.schema_chunks.")


if __name__ == "__main__":
    main()
