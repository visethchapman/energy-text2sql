# Energy Text-to-SQL Agent

A multi-agent natural-language-to-SQL system for the US electricity grid. Ask
plain-English questions like *"How did the February 2021 Texas freeze affect
ERCOT demand?"* and get an answer backed by real EIA hourly demand data joined
with NOAA weather.

Inspired by Databricks'
[`dbx-unifiedchat`](https://github.com/databricks-solutions/dbx-unifiedchat)
reference architecture, re-implemented on a fully open stack (Postgres +
pgvector + LangGraph + Claude API) so anyone can clone and run it locally
without a Databricks workspace.

## Status

**Week 1, Day 1** — data + schema bootstrap. Agent comes next.

## Stack

| Layer | Tech |
|---|---|
| Database | Postgres 16 + pgvector (Docker) |
| Data | EIA Open Data v2 (electricity), NOAA GHCN-Daily (weather) |
| Runtime | Python 3.11 + `uv` |
| Orchestration | LangGraph *(Day 4+)* |
| LLM | Anthropic Claude Sonnet 4.5 + Haiku 4.5 *(Day 2+)* |
| Tracing | MLflow *(Day 6+)* |

## Quick start

### Prerequisites

- Docker Desktop (or OrbStack — lighter on Mac)
- Python 3.11+
- `uv` — install via `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 1. EIA API key

Register (free, instant) at <https://www.eia.gov/opendata/register.php>.

### 2. Configure

```bash
cp .env.example .env
# edit .env: paste your EIA_API_KEY
```

### 3. Start Postgres

```bash
docker compose up -d
```

### 4. Install Python deps

```bash
uv sync
```

### 5. Load data

```bash
uv run python etl/01_load_eia.py --region ERCO          # ~5 min, ~40k rows
uv run python etl/02_load_noaa.py                        # ~30 sec
uv run python etl/03_schema_docs.py                      # instant
```

For all four major regions:

```bash
uv run python etl/01_load_eia.py --region ERCO CISO PJM NYIS
```

### 6. Verify

```bash
docker exec -it energy-postgres psql -U energy -d energy -c \
  "SELECT region, COUNT(*) AS rows, MIN(period) AS first, MAX(period) AS last
   FROM eia.demand GROUP BY region;"
```

You should see one row per loaded region with ~40,000+ hours of data.

## Repository layout

```
.
├── docker-compose.yml      # Postgres 16 + pgvector
├── pyproject.toml          # Python deps (uv)
├── .env.example            # Config template
├── db/init.sql             # Schemas + tables (auto-runs on first container start)
├── etl/
│   ├── 01_load_eia.py      # EIA hourly demand -> eia.demand
│   ├── 02_load_noaa.py     # NOAA daily weather -> noaa.daily_weather
│   └── 03_schema_docs.py   # Schema cards -> docs.schema_cards
├── agent/                  # LangGraph multi-agent (Day 4+)
└── eval/                   # Eval harness + scorer (Day 3+)
```

## Roadmap

| Day | Deliverable | Status |
|---|---|---|
| 1 | Postgres + ETL — EIA + NOAA loaded | ✅ |
| 2 | Baseline single-call agent (schema in prompt) | |
| 3 | First eval set (20 hand-curated questions) + result-equivalence scorer | |
| 4 | LangGraph multi-agent: planning → synthesis → execution → summarize | |
| 5 | Vector-search routing over schema cards (pgvector) | |
| 6 | SQL retry loop + MLflow tracing | |
| 7 | FastAPI server + minimal React UI | |
| 8+ | Polish, BIRD baseline number, recorded demo video | |

## Demo questions (target)

- *"How did the February 2021 Texas freeze affect ERCOT electricity demand?"*
- *"What is the correlation between daily peak temperature and electricity demand in California?"*
- *"Compare average electricity demand in summer vs winter across all four regions."*
- *"Find days where NYC temperature exceeded 35 deg C and show the next-day demand spike."*

## License

MIT.
