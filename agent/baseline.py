"""
Day 2 baseline: single-call text-to-SQL agent.

Pipeline:
    user question
    -> stuff every schema card + raw column listing into the prompt
    -> Claude Sonnet 4.5
    -> extract SQL (```sql fence)
    -> reject anything that isn't read-only
    -> execute on Postgres
    -> print rows + cost

Intentionally dumb. Establishes the measurable floor that Day 4+'s
multi-agent system has to beat.

Usage:
    uv run python agent/baseline.py "What was the peak hourly demand in ERCOT in 2024?"
    uv run python agent/baseline.py --interactive
"""

import argparse
import os
import re
import sys
from pathlib import Path
from textwrap import dedent

import psycopg
from dotenv import load_dotenv

# Load .env BEFORE importing anthropic. override=True is critical:
# Claude Code injects its own ANTHROPIC_API_KEY into the shell via launchctl,
# and without override that wins over the value in .env.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)
os.environ.pop("ANTHROPIC_BASE_URL", None)

import anthropic  # noqa: E402  (must come after dotenv)


MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 1024

# Sonnet 4.5 pricing as of 2026 ($/Mtok input/output)
PRICE_IN, PRICE_OUT = 3.0, 15.0

SAFE_PREFIXES = ("SELECT", "WITH")
BANNED_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "MERGE", "CALL", "VACUUM",
)

SYSTEM_PROMPT = dedent("""\
    You are a Postgres SQL expert. Given a user question and a schema, return ONE
    SQL query that answers it.

    Rules:
    - Only SELECT or WITH ... SELECT. No INSERT/UPDATE/DELETE/DDL.
    - All times in eia.demand are TIMESTAMPTZ in UTC.
    - Weather observations in noaa.daily_weather are local-station DATE values.
      To join with demand: cast d.period::date = w.obs_date AND pick the right
      station via noaa.stations.nearest_eia_region = d.region.
    - Available regions: ERCO (Texas), CISO (California), PJM (Mid-Atlantic), NYIS (NY).
    - Wrap your SQL in a ```sql ...``` fence.
    - After the SQL, add ONE short sentence explaining what the query does.
""")

SQL_FENCE_RE = re.compile(r"```sql\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def load_schema(conn: psycopg.Connection) -> str:
    """Pull schema cards + raw column listing into a single prompt block."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT schema_name, table_name, description, columns_summary, sample_values
            FROM docs.schema_cards
            ORDER BY schema_name, table_name
        """)
        cards = cur.fetchall()

        cur.execute("""
            SELECT table_schema, table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema IN ('eia', 'noaa')
            ORDER BY table_schema, table_name, ordinal_position
        """)
        columns = cur.fetchall()

    by_table: dict[tuple[str, str], list[str]] = {}
    for schema, table, col, dtype in columns:
        by_table.setdefault((schema, table), []).append(f"{col} {dtype}")

    parts = []
    for schema_name, table_name, description, columns_summary, sample_values in cards:
        cols = by_table.get((schema_name, table_name), [])
        parts.append(dedent(f"""
            ### {schema_name}.{table_name}
            {description}

            Columns ({len(cols)}):
              {", ".join(cols)}

            Notes: {columns_summary}
            Sample values: {sample_values}
        """).strip())
    return "\n\n".join(parts)


def call_claude(client: anthropic.Anthropic, question: str, schema: str):
    user_msg = f"SCHEMA:\n\n{schema}\n\n---\n\nQUESTION: {question}"
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "\n".join(block.text for block in resp.content if block.type == "text")
    return text, resp.usage


def extract_sql(text: str) -> str | None:
    m = SQL_FENCE_RE.search(text)
    return m.group(1).strip() if m else None


def is_safe_sql(sql: str) -> tuple[bool, str]:
    upper = sql.upper().lstrip()
    if not any(upper.startswith(p) for p in SAFE_PREFIXES):
        return False, "must start with SELECT or WITH"
    for kw in BANNED_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return False, f"contains banned keyword {kw}"
    return True, ""


def execute_sql(conn: psycopg.Connection, sql: str, max_rows: int = 100):
    """Execute SQL and fetch up to max_rows. Also returns total row count
    (so we can warn when truncating)."""
    with conn.cursor() as cur:
        cur.execute(sql)
        all_rows = cur.fetchall()  # day-1 datasets are small; safe to pull all
        cols = [d.name for d in cur.description] if cur.description else []
    return cols, all_rows[:max_rows], len(all_rows)


def print_rows(cols: list[str], rows: list[tuple]) -> None:
    if not rows:
        print("  (no rows returned)")
        return
    widths = [max(len(c), *(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    print("  " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print("  " + " | ".join(str(v).ljust(w) for v, w in zip(r, widths)))


def run_one(question: str, client: anthropic.Anthropic, conn: psycopg.Connection) -> None:
    print(f"\n{'=' * 70}")
    print(f"Q: {question}")
    print("=" * 70)

    schema = load_schema(conn)

    raw, usage = call_claude(client, question, schema)
    cost = (usage.input_tokens * PRICE_IN + usage.output_tokens * PRICE_OUT) / 1_000_000
    print(f"[LLM: in={usage.input_tokens} tok, out={usage.output_tokens} tok, ${cost:.4f}]")

    sql = extract_sql(raw)
    if not sql:
        print("\n✗ no ```sql fence in response. raw output:\n")
        print(raw)
        return

    safe, reason = is_safe_sql(sql)
    if not safe:
        print(f"\n✗ SQL rejected by safety check ({reason}):\n\n{sql}")
        return

    print(f"\nSQL:\n{sql}\n")
    try:
        cols, rows, total = execute_sql(conn, sql)
        if total > len(rows):
            print(f"✓ executed — showing {len(rows)} of {total} rows:\n")
        else:
            print(f"✓ executed — {total} row(s):\n")
        print_rows(cols, rows)
    except psycopg.Error as e:
        print(f"✗ Postgres error: {e}")
        # roll back the failed transaction so the connection stays usable
        conn.rollback()


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 2 baseline agent")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--interactive", action="store_true", help="REPL mode")
    args = parser.parse_args()

    if not args.question and not args.interactive:
        parser.error("provide a question or use --interactive")

    db_url = os.environ["DATABASE_URL"]
    client = anthropic.Anthropic()

    with psycopg.connect(db_url) as conn:
        if args.interactive:
            print("Baseline agent. Type 'exit' to quit.")
            while True:
                try:
                    q = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not q or q in ("exit", "quit"):
                    break
                run_one(q, client, conn)
        else:
            run_one(args.question, client, conn)


if __name__ == "__main__":
    main()
