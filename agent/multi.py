"""
Day 4: LangGraph multi-agent text-to-SQL.

Graph:

    START
      └── plan        (Sonnet)  — short bullet plan: tables, joins, pitfalls
            └── synthesize  (Sonnet)  — generate SQL
                  │
                  ├── on bad/missing SQL → summarize (explain why)
                  └── execute
                        │
                        ├── on Postgres error AND retries left → synthesize (with feedback)
                        └── on success / retries exhausted → summarize (Haiku) → END

State carries token/cost accumulators via Annotated[..., operator.add] reducers,
so each node returns only its delta and totals accumulate automatically.
"""

from __future__ import annotations

import operator
import time
from textwrap import dedent
from typing import Annotated, TypedDict

import anthropic
import psycopg
from langgraph.graph import StateGraph, START, END

from agent.base import AgentResult
from agent.baseline import (
    PRICE_IN as SONNET_IN,
    PRICE_OUT as SONNET_OUT,
    execute_sql, extract_sql, is_safe_sql, load_schema,
)

SONNET = "claude-sonnet-4-5"
HAIKU = "claude-haiku-4-5"
HAIKU_IN, HAIKU_OUT = 1.0, 5.0
MAX_RETRIES = 2


# ---------- prompts ----------
PLAN_SYSTEM = dedent("""\
    You are a Postgres data analyst. Given a question and a schema, write a
    SHORT plan (3-5 lines) covering:
      - Which tables/columns matter.
      - Any joins required and the join keys.
      - Subtleties only IF they apply (e.g. timezone alignment ONLY when joining
        eia.demand with noaa.daily_weather; NULL handling; date boundaries).

    Be concise. Do NOT write SQL. Stay under 5 lines.
""")

SYNTHESIS_SYSTEM = dedent("""\
    You are a Postgres SQL expert. Given a question, schema, and execution plan,
    return ONE SQL query that answers the question.

    Rules:
    - SELECT or WITH ... SELECT only. No DDL/DML.
    - Timezone handling:
        * eia.demand.period is TIMESTAMPTZ in UTC.
        * noaa.daily_weather.obs_date is a local-station DATE.
        * ONLY when joining demand with weather: convert period to local time,
          e.g. (period AT TIME ZONE 'America/Chicago')::date for Houston/ERCO.
        * For ANY query that does NOT join with noaa.daily_weather (including
          year/month/day grouping on demand alone, peak/min queries, etc.),
          use period AS-IS in UTC. Do not convert.
    - Available demand regions: ERCO (Texas) only is currently loaded.
    - Return ONLY the columns the question explicitly asks for. Do not include
      helpful extras like units, ids, or descriptive columns unless asked.
    - Wrap your SQL in a ```sql ... ``` fence.
""")

SYNTHESIS_RETRY_SUFFIX = dedent("""\

    --- PREVIOUS ATTEMPT FAILED ---
    Your previous SQL:
    ```sql
    {prev_sql}
    ```

    Error from Postgres:
    {error}

    Return a corrected query. Address the specific error above.
""")

SUMMARIZE_SYSTEM = dedent("""\
    You are a data analyst. Write a brief plain-English answer (2-4 sentences)
    to the user's question based on the SQL query result.

    Be direct. Cite specific numbers/dates from the result. Call out anomalies
    (zero rows, unusually high/low values, missing data) explicitly.

    If the result is empty or the query errored, say so clearly — do not invent
    numbers.
""")


# ---------- state ----------
class State(TypedDict, total=False):
    question: str
    schema: str
    plan: str | None
    sql: str | None
    prev_sql: str | None
    result_rows: list | None
    result_columns: list[str] | None
    error: str | None
    category: str
    final_answer: str | None
    retry_count: int
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    cost: Annotated[float, operator.add]


def _call(client: anthropic.Anthropic, model: str, system: str, user: str, max_tokens: int = 1024):
    resp = client.messages.create(
        model=model, max_tokens=max_tokens,
        system=system, messages=[{"role": "user", "content": user}],
    )
    text = "\n".join(b.text for b in resp.content if b.type == "text")
    if model.startswith("claude-sonnet"):
        pin, pout = SONNET_IN, SONNET_OUT
    else:
        pin, pout = HAIKU_IN, HAIKU_OUT
    cost = (resp.usage.input_tokens * pin + resp.usage.output_tokens * pout) / 1_000_000
    return text, resp.usage.input_tokens, resp.usage.output_tokens, cost


# ---------- graph factory ----------
def _build_graph(client: anthropic.Anthropic, conn: psycopg.Connection):
    """Compile a fresh graph wired to this (client, conn) pair."""

    def plan_node(state: State) -> dict:
        user = f"SCHEMA:\n{state['schema']}\n\nQUESTION: {state['question']}"
        text, ti, to, c = _call(client, SONNET, PLAN_SYSTEM, user, max_tokens=400)
        return {"plan": text.strip(), "input_tokens": ti, "output_tokens": to, "cost": c}

    def synthesize_node(state: State) -> dict:
        user = (
            f"SCHEMA:\n{state['schema']}\n\n"
            f"PLAN:\n{state['plan']}\n\n"
            f"QUESTION: {state['question']}"
        )
        if state.get("retry_count", 0) > 0 and state.get("error"):
            user += SYNTHESIS_RETRY_SUFFIX.format(
                prev_sql=state.get("prev_sql", state.get("sql") or "(none)"),
                error=state["error"],
            )
        text, ti, to, c = _call(client, SONNET, SYNTHESIS_SYSTEM, user, max_tokens=1024)
        sql = extract_sql(text)
        update: dict = {"input_tokens": ti, "output_tokens": to, "cost": c}
        if sql is None:
            update.update(sql=None, error="no SQL fence in synthesis output", category="no_sql")
            return update
        safe, reason = is_safe_sql(sql)
        if not safe:
            update.update(sql=sql, error=f"unsafe SQL ({reason})", category="unsafe_sql")
            return update
        # Clear prior retry error so summarize doesn't think we're still failing.
        update.update(sql=sql, error=None, category="")
        return update

    def execute_node(state: State) -> dict:
        if state.get("sql") is None:
            return {}
        try:
            cols, rows, _ = execute_sql(conn, state["sql"], max_rows=10_000)
            return {"result_columns": cols, "result_rows": rows, "error": None, "category": ""}
        except psycopg.Error as e:
            conn.rollback()
            msg = str(e).strip().splitlines()[0][:200]
            return {
                "error": msg, "category": "sql_error",
                "prev_sql": state["sql"],
                "retry_count": state.get("retry_count", 0) + 1,
            }

    def summarize_node(state: State) -> dict:
        if state.get("error") and state.get("result_rows") is None:
            ctx = f"The SQL failed or no SQL was produced. Error: {state['error']}"
        else:
            rows = state.get("result_rows") or []
            cols = state.get("result_columns") or []
            shown = rows[:50]
            ctx = (
                "QUERY RESULT:\n"
                + ", ".join(cols) + "\n"
                + "\n".join(str(r) for r in shown)
            )
            if len(rows) > 50:
                ctx += f"\n... ({len(rows) - 50} more rows)"
        user = f"QUESTION: {state['question']}\n\n{ctx}"
        text, ti, to, c = _call(client, HAIKU, SUMMARIZE_SYSTEM, user, max_tokens=400)
        return {"final_answer": text.strip(), "input_tokens": ti, "output_tokens": to, "cost": c}

    def route_after_synthesize(state: State) -> str:
        if state.get("category") in ("no_sql", "unsafe_sql"):
            return "summarize"
        return "execute"

    def route_after_execute(state: State) -> str:
        if state.get("category") == "sql_error" and state.get("retry_count", 0) <= MAX_RETRIES:
            return "synthesize"
        return "summarize"

    g = StateGraph(State)
    g.add_node("plan", plan_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("execute", execute_node)
    g.add_node("summarize", summarize_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "synthesize")
    g.add_conditional_edges("synthesize", route_after_synthesize,
                            {"execute": "execute", "summarize": "summarize"})
    g.add_conditional_edges("execute", route_after_execute,
                            {"synthesize": "synthesize", "summarize": "summarize"})
    g.add_edge("summarize", END)

    return g.compile()


class MultiAgent:
    name = "multi"

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        *,
        use_rag: bool = False,
        rag_k: int = 8,
    ):
        self.client = client or anthropic.Anthropic()
        self.use_rag = use_rag
        self.rag_k = rag_k

    def _build_schema_for(self, question: str, conn: psycopg.Connection) -> str:
        """Return the schema string to put in the prompt.

        With use_rag=False: full schema dump via load_schema() — the v1 behavior.
        With use_rag=True:  top-K relevant column chunks retrieved via pgvector.
        """
        if self.use_rag:
            from agent.retrieval import retrieve_relevant_columns
            return retrieve_relevant_columns(question, conn, k=self.rag_k)
        return load_schema(conn)

    def stream(self, question: str, conn: psycopg.Connection):
        """Yield events as each LangGraph node completes.

        Each event is a dict:
            {"node": "<name>", "data": {state delta from that node}}
        Followed at the very end by:
            {"type": "done", "total_cost": float, "total_latency_sec": float,
             "input_tokens": int, "output_tokens": int, "retry_count": int}

        On retry, the synthesize/execute pair re-fires — clients should treat
        each event as a replacement for the previous of the same node.
        """
        graph = _build_graph(self.client, conn)
        initial: State = {
            "question": question,
            "schema": self._build_schema_for(question, conn),
            "input_tokens": 0, "output_tokens": 0, "cost": 0.0,
            "retry_count": 0,
        }
        t0 = time.perf_counter()
        merged: dict = {}

        for chunk in graph.stream(initial, stream_mode="updates"):
            # chunk = {"node_name": {state_delta}}; our graph fires one node per step.
            for node_name, delta in chunk.items():
                if not isinstance(delta, dict):
                    continue
                # Merge for "done" totals at the end. Accumulators add; scalars replace.
                for k, v in delta.items():
                    if k in ("input_tokens", "output_tokens", "cost"):
                        merged[k] = merged.get(k, 0) + (v or 0)
                    else:
                        merged[k] = v
                yield {"node": node_name, "data": delta}

        yield {
            "type": "done",
            "total_cost": round(merged.get("cost", 0.0), 6),
            "total_latency_sec": round(time.perf_counter() - t0, 3),
            "input_tokens": merged.get("input_tokens", 0),
            "output_tokens": merged.get("output_tokens", 0),
            "retry_count": merged.get("retry_count", 0),
            "category": merged.get("category", ""),
            "error": merged.get("error"),
        }

    def answer(self, question: str, conn: psycopg.Connection) -> AgentResult:
        graph = _build_graph(self.client, conn)
        initial: State = {
            "question": question,
            "schema": self._build_schema_for(question, conn),
            "input_tokens": 0, "output_tokens": 0, "cost": 0.0,
            "retry_count": 0,
        }
        t0 = time.perf_counter()
        final = graph.invoke(initial)
        latency = round(time.perf_counter() - t0, 3)

        return AgentResult(
            sql=final.get("sql"),
            result_rows=final.get("result_rows"),
            result_columns=final.get("result_columns"),
            final_answer=final.get("final_answer"),
            error=final.get("error") if final.get("category") else None,
            category=final.get("category", ""),
            input_tokens=final.get("input_tokens", 0),
            output_tokens=final.get("output_tokens", 0),
            cost=round(final.get("cost", 0.0), 6),
            latency_sec=latency,
            extra={"plan": final.get("plan"), "retry_count": final.get("retry_count", 0)},
        )
