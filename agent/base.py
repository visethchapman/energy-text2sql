"""Shared interface every text-to-SQL agent in this project implements.

Lets eval/run.py treat baseline and multi-agent identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import psycopg


@dataclass
class AgentResult:
    """Standard envelope returned for a single question."""
    sql: str | None = None
    result_rows: list[tuple] | None = None
    result_columns: list[str] | None = None
    final_answer: str | None = None      # multi-agent prose summary; None for baseline
    error: str | None = None
    category: str = ""                    # "" if OK, else "no_sql" | "unsafe_sql" | "sql_error"
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_sec: float = 0.0
    extra: dict = field(default_factory=dict)  # per-agent extras (plan text, retry_count, ...)


@runtime_checkable
class Agent(Protocol):
    name: str

    def answer(self, question: str, conn: psycopg.Connection) -> AgentResult: ...
