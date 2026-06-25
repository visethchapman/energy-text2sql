# TODO

Tracked work and known limitations. Items here are intentional skips,
not bugs. v1 + v2 (vector-routed RAG) shipped; remaining items below.

## Architectural gaps still to fix

### Timezone metadata is hardcoded to Houston
- `agent/multi.py` SYNTHESIS_SYSTEM contains the literal string
  `'America/Chicago'` because all loaded demand is ERCO.
- `eval/dataset.jsonl` gold SQL for q09/q10/q12 hardcodes the same.
- If CISO/PJM/NYIS data is loaded, the agent would have to guess the
  region's timezone from context.
- **Proper fix:** add a `timezone` column to `noaa.stations` (or a new
  `eia.regions` table), populate from a region→TZ map in
  `etl/02_load_noaa.py`, and update the SYNTHESIS prompt to instruct
  the model to look up the TZ from schema rather than guess.

### Scorer only measures SQL row-equivalence
- Multi-agent's prose summaries (`final_answer`) are not scored.
- Summary hallucinations (e.g., Haiku citing "62,018 MWh" when the row
  was "59,722 MWh" — see q12 commentary) are invisible to the eval.
- **Proper fix:** add an LLM-as-judge scoring mode that checks whether
  `final_answer` is faithful to `result_rows`. Or a regex-based check for
  specific expected numbers per question. Slower and costlier per run.

### Scorer is column-count strict
- A correct query that returns one extra helpful column (e.g.,
  `value_units` on a "peak demand" question) is marked wrong even if
  the requested values are present.
- The Day 4 fix was a synthesis-prompt instruction *"return ONLY
  requested columns"*. Works for now; will break for ambiguous
  questions. A scorer that compares the subset of requested columns
  would be better.

## Dataset growth ideas

- 5-8 questions designed to exercise the multi-agent's strengths:
  window functions, UNION ALL, 3-CTE chains, intentional column-name
  ambiguity, date arithmetic. Baseline should score lower on these.
- 2-3 "silent zero" questions targeted at CISO/PJM/NYIS regions once
  those are loaded. Tests whether the summarize node correctly
  acknowledges "no data" vs hallucinating an answer.

## Terminology

The repo calls the LangGraph workflow "multi-agent" because that's the
2024-2026 industry term (a graph of role-specialized LLM nodes sharing
state). It is *not* multi-agent in the academic sense — no autonomy,
no parallelism, no agent-to-agent messaging. README will clarify on
launch.

## Future enhancements (deliberately out of current scope)

- DSPy optimizer pass on the synthesis prompt (needs eval set ≥ 25).
- Adaptive RAG retrieval (rerank top-K by per-query relevance, or use a
  cross-encoder reranker on the retrieved chunks).

### Already shipped (kept for reference)

- ✅ Vector-search routing over `docs.schema_chunks` — shipped Day 7 as multi+RAG mode.
- ✅ FastAPI server + Pico.css UI — shipped Day 5.
- ✅ Recorded demo video — shipped Day 6.
