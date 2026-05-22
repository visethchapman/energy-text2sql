"""
Result-equivalence scoring for text-to-SQL eval.

The whole reason eval is hard: two different SQL queries can be both "correct"
yet produce different column names, column orders, row orders, or numeric
precision. We compare on RESULT VALUES, not SQL string.

v1 strategy (intentionally crude):
  - normalize each row into a canonical tuple (numbers rounded, dates as ISO)
  - sort all rows by their canonical form
  - compare row-by-row

Known limitations to iterate on later:
  - Doesn't enforce row order even when the question says "top 10 ordered by".
  - Treats columns as positional, not by name.
  - Float tolerance is a fixed 1e-3 relative; some aggregates need looser.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from decimal import Decimal


def _canon(v: object) -> object:
    """Canonicalize a single cell for comparison."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int,)):
        return float(v)
    if isinstance(v, Decimal):
        return round(float(v), 4)
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    return str(v)


def _canon_row(row: tuple) -> tuple:
    return tuple(_canon(v) for v in row)


def _values_close(a: object, b: object) -> bool:
    """Tolerant equality for cells."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-3)
    return a == b


@dataclass
class ScoreResult:
    correct: bool
    reason: str = ""
    gold_rows: int = 0
    agent_rows: int = 0
    gold_cols: int = 0
    agent_cols: int = 0


def score(gold_rows: list[tuple], agent_rows: list[tuple] | None) -> ScoreResult:
    """Compare agent result against gold result. Returns ScoreResult."""
    if agent_rows is None:
        return ScoreResult(False, "agent produced no result", gold_rows=len(gold_rows))

    g_cols = len(gold_rows[0]) if gold_rows else 0
    a_cols = len(agent_rows[0]) if agent_rows else 0

    if len(gold_rows) != len(agent_rows):
        return ScoreResult(
            False, f"row count mismatch: gold={len(gold_rows)} vs agent={len(agent_rows)}",
            gold_rows=len(gold_rows), agent_rows=len(agent_rows),
            gold_cols=g_cols, agent_cols=a_cols,
        )

    if g_cols != a_cols:
        return ScoreResult(
            False, f"column count mismatch: gold={g_cols} vs agent={a_cols}",
            gold_rows=len(gold_rows), agent_rows=len(agent_rows),
            gold_cols=g_cols, agent_cols=a_cols,
        )

    g_sorted = sorted(_canon_row(r) for r in gold_rows)
    a_sorted = sorted(_canon_row(r) for r in agent_rows)

    for i, (gr, ar) in enumerate(zip(g_sorted, a_sorted)):
        for j, (gv, av) in enumerate(zip(gr, ar)):
            if not _values_close(gv, av):
                return ScoreResult(
                    False,
                    f"value mismatch at sorted-row {i} col {j}: gold={gv!r} vs agent={av!r}",
                    gold_rows=len(gold_rows), agent_rows=len(agent_rows),
                    gold_cols=g_cols, agent_cols=a_cols,
                )

    return ScoreResult(
        True, "ok",
        gold_rows=len(gold_rows), agent_rows=len(agent_rows),
        gold_cols=g_cols, agent_cols=a_cols,
    )
