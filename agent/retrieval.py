"""
Vector-routed schema retrieval (v2).

Replaces (or supplements) load_schema() — instead of stuffing every
column from every table into the prompt, retrieve only the top-K
columns most semantically relevant to the user's question.

Pattern used by Databricks Genie, Snowflake Cortex Analyst, Vanna AI,
LangChain SQLDatabase toolkit, etc. — at warehouse scale (1000s of
columns) it's the only way to keep prompts tractable.

Usage:
    from agent.retrieval import retrieve_relevant_columns
    schema_str = retrieve_relevant_columns("peak demand in 2024", conn, k=6)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-large-en-v1.5"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Load the embedding model lazily and cache it for the process lifetime."""
    return SentenceTransformer(MODEL_NAME)


def _format_rows(rows: Iterable[tuple]) -> str:
    """Group retrieved columns by their parent table, render as a mini-schema."""
    by_table: dict[str, list[tuple]] = {}
    for sn, tn, cn, dt, desc, sv, sim in rows:
        by_table.setdefault(f"{sn}.{tn}", []).append((cn, dt, desc, sv, sim))

    sections: list[str] = []
    for table_key, cols in by_table.items():
        col_lines = [
            f"  - {cn} ({dt}): {desc}\n    Sample: {sv}"
            for cn, dt, desc, sv, _sim in cols
        ]
        sections.append(f"### {table_key}\n" + "\n".join(col_lines))
    return "\n\n".join(sections)


def retrieve_relevant_columns(
    question: str,
    conn: psycopg.Connection,
    k: int = 8,
    *,
    return_debug: bool = False,
) -> str | tuple[str, list[tuple]]:
    """
    Embed the question, fetch the top-K schema_chunks rows by cosine similarity,
    and return a formatted mini-schema string for inclusion in the agent prompt.

    Set return_debug=True to also get the raw retrieval rows (with similarity
    scores) — useful for logging which columns the agent saw for a given query.
    """
    embedding = _model().encode([question], normalize_embeddings=True)[0]

    register_vector(conn)  # idempotent
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT schema_name, table_name, column_name, data_type,
                   description, sample_values,
                   1 - (embedding <=> %s) AS similarity
            FROM docs.schema_chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (embedding, embedding, k),
        )
        rows = cur.fetchall()

    schema_text = _format_rows(rows)
    if return_debug:
        return schema_text, rows
    return schema_text


if __name__ == "__main__":
    # Quick smoke test: print what gets retrieved for a few representative
    # questions, with similarity scores.
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)
    db_url = os.environ["DATABASE_URL"]

    test_questions = [
        "What was the peak hourly ERCOT demand in 2024?",
        "Houston's coldest day in the dataset.",
        "Correlation between Houston temperature and ERCOT demand.",
    ]
    with psycopg.connect(db_url) as conn:
        for q in test_questions:
            text, debug = retrieve_relevant_columns(q, conn, k=5, return_debug=True)
            print("\n" + "=" * 70)
            print(f"Q: {q}")
            print("Retrieved:")
            for sn, tn, cn, _dt, _desc, _sv, sim in debug:
                print(f"  {sim:.3f}  {sn}.{tn}.{cn}")
