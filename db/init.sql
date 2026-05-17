-- Day 1 schema bootstrap.
-- Runs once when the postgres container is first created.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS eia;
CREATE SCHEMA IF NOT EXISTS noaa;
CREATE SCHEMA IF NOT EXISTS docs;

-- EIA: hourly electricity demand by US balancing authority (BA)
CREATE TABLE IF NOT EXISTS eia.demand (
    region       TEXT             NOT NULL,    -- BA code: ERCO, CISO, PJM, NYIS
    period       TIMESTAMPTZ      NOT NULL,    -- hour in UTC
    value        DOUBLE PRECISION,             -- demand in MWh
    value_units  TEXT,
    PRIMARY KEY (region, period)
);
CREATE INDEX IF NOT EXISTS idx_demand_period ON eia.demand(period);

-- NOAA GHCN-Daily: daily weather observations
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
    tmax_c      DOUBLE PRECISION,            -- daily max temp, deg C
    tmin_c      DOUBLE PRECISION,            -- daily min temp, deg C
    prcp_mm     DOUBLE PRECISION,            -- precipitation, mm
    awnd_ms     DOUBLE PRECISION,            -- average wind speed, m/s
    PRIMARY KEY (station_id, obs_date)
);

-- Schema cards for routing — embeddings filled in later (Day 5)
CREATE TABLE IF NOT EXISTS docs.schema_cards (
    id               SERIAL PRIMARY KEY,
    schema_name      TEXT NOT NULL,
    table_name       TEXT NOT NULL,
    description      TEXT,
    columns_summary  TEXT,
    sample_values    TEXT,
    embedding        VECTOR(1024),
    UNIQUE (schema_name, table_name)
);
