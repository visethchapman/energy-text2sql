# Eval harness

The job of this directory: produce one number that tracks whether the agent
is getting better or worse over time. Without it, every Day 4+ "improvement"
is a vibe.

## Files

| File | Role |
|---|---|
| `dataset.jsonl` | 12 hand-curated questions + gold SQL |
| `scorer.py` | Compare two result sets (sort-insensitive, float-tolerant) |
| `run.py` | Run an agent against the dataset, print + save a scoreboard |
| `runs/` | Per-run JSON snapshots (gitignored except a baseline reference) |

## Running

```bash
uv run python eval/run.py                # full dataset
uv run python eval/run.py --limit 3      # smoke test
uv run python eval/run.py --ids q09 q10  # specific questions
uv run python eval/run.py --save -v      # save + show failure reasons
```

## Dataset format (`dataset.jsonl`)

One JSON object per line:

```json
{
  "id": "q01",
  "question": "What is the lowest hourly ERCOT demand value in the dataset?",
  "gold_sql": "SELECT MIN(value) FROM eia.demand WHERE region = 'ERCO'",
  "tags": ["agg", "single_table"],
  "notes": "optional: subtle reasoning or known semantic considerations"
}
```

The `gold_sql` is *not* the canonical correct query — there can be many. It's
*one* query the maintainer verified produces the right rows. The eval
compares the agent's result-rows against the gold's result-rows.

## Known semantic gotchas

### Timezone alignment in cross-domain joins (q10, q12)

`eia.demand.period` is `TIMESTAMPTZ` stored in **UTC**.
`noaa.daily_weather.obs_date` is a **local-station date** (Houston station =
America/Chicago, etc.).

Naive `period::date` casts UTC hours to UTC dates, which **does not align**
with NOAA local dates. For Houston in winter (UTC−6), the calendar day "Feb
13" in Chicago covers UTC hours 2021-02-13 06:00 through 2021-02-14 05:59.
Grouping by `period::date` puts the wrong six hours of demand into "Feb 13."

Correct pattern, which the v1 baseline agent produced before we had it in
the gold:

```sql
GROUP BY (period AT TIME ZONE 'America/Chicago')::date
```

The first eval run surfaced this — the agent's answer was *more* correct than
the gold. Lesson: an eval's value is bidirectional. It tests the agent and
it tests your ground truth.

### Float tolerance

`scorer.py` compares floats with `rel_tol=1e-3, abs_tol=1e-3`. Some
correlation / aggregation queries may legitimately drift more than that
depending on aggregation order. If you see "value mismatch" by a tiny
fraction, that's the bound to revisit, not the agent.

### Row ordering

The current scorer sorts both result sets before comparing, so queries that
return the same rows in a different order are treated as equivalent. This is
wrong for queries that explicitly ask for ordering ("top 10 by X") — those
should preserve order. Tracked for a v2 scorer.

## Reference runs

| Date | Agent | Score | Total cost | Avg latency | Notes |
|---|---|---|---|---|---|
| 2026-05-21 | baseline | 10/12 → 12/12 after gold fix | $0.054 | 4.6s | Single LLM call, schema in prompt. Surfaced UTC-vs-local-date bug in ground truth. |
| 2026-05-22 | multi (LangGraph) | 7/12 → 9/12 → 12/12 over 3 iterations | $0.103 | 9.6s | Plan → synthesize → execute (with retry) → summarize. First run over-applied timezone conversion (regression to 58%); tightened prompts + aligned gold to Chicago-local semantics. |
| 2026-06-18 | multi + RAG (k=8) | 12/12 | $0.110 | 9.1s | Vector-routed schema retrieval via pgvector + bge-large embeddings. **+9% input tokens** at this 17-chunk scale because formatted column descriptions are more verbose than a raw schema dump. The pattern is designed for warehouse-scale schemas (1000s of columns) — at 17 chunks it doesn't save tokens, only demonstrates the production approach. |

The eval is a no-regression gate for SQL correctness. Multi-agent's prose
answers, retry recovery, and silent-zero detection are not measured here —
those live in the demo video and qualitative review.
