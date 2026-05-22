"""
Run an agent against eval/dataset.jsonl and print a scoreboard.

Usage:
    uv run python eval/run.py                       # run baseline against full dataset
    uv run python eval/run.py --limit 3             # smoke test, first 3 questions
    uv run python eval/run.py --ids q09 q10         # specific questions
    uv run python eval/run.py --save                # save run to eval/runs/<ts>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(dotenv_path=ROOT / ".env", override=True)
os.environ.pop("ANTHROPIC_BASE_URL", None)

import anthropic  # noqa: E402

from agent.baseline import (  # noqa: E402
    PRICE_IN, PRICE_OUT,
    call_claude, execute_sql, extract_sql, is_safe_sql, load_schema,
)
from eval.scorer import score  # noqa: E402

DATASET_PATH = ROOT / "eval" / "dataset.jsonl"
RUNS_DIR = ROOT / "eval" / "runs"


def load_dataset(limit: int | None = None, ids: list[str] | None = None) -> list[dict]:
    rows = [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]
    if ids:
        rows = [r for r in rows if r["id"] in ids]
    if limit:
        rows = rows[:limit]
    return rows


def run_gold(conn: psycopg.Connection, sql: str) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def run_agent(question: str, conn: psycopg.Connection, client: anthropic.Anthropic, schema: str) -> dict:
    """Run the baseline pipeline. Returns a dict with sql, result, cost, latency, error."""
    out: dict = {"sql": None, "result": None, "error": None,
                 "input_tokens": 0, "output_tokens": 0, "cost": 0.0, "latency_sec": 0.0,
                 "category": ""}
    t0 = time.perf_counter()
    try:
        raw, usage = call_claude(client, question, schema)
        out["input_tokens"] = usage.input_tokens
        out["output_tokens"] = usage.output_tokens
        out["cost"] = (usage.input_tokens * PRICE_IN + usage.output_tokens * PRICE_OUT) / 1_000_000

        sql = extract_sql(raw)
        if not sql:
            out["error"] = "no SQL fence in response"
            out["category"] = "no_sql"
            return out
        out["sql"] = sql

        safe, reason = is_safe_sql(sql)
        if not safe:
            out["error"] = f"unsafe SQL ({reason})"
            out["category"] = "unsafe_sql"
            return out

        try:
            _, rows, _ = execute_sql(conn, sql, max_rows=10_000)
            out["result"] = rows
        except psycopg.Error as e:
            conn.rollback()
            out["error"] = str(e).strip().splitlines()[0][:200]
            out["category"] = "sql_error"
    finally:
        out["latency_sec"] = round(time.perf_counter() - t0, 3)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", nargs="+", default=None)
    parser.add_argument("--save", action="store_true", help="Save run results to eval/runs/")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(limit=args.limit, ids=args.ids)
    print(f"Running {len(dataset)} questions against baseline agent...\n")

    db_url = os.environ["DATABASE_URL"]
    client = anthropic.Anthropic()
    results: list[dict] = []

    with psycopg.connect(db_url) as conn:
        schema = load_schema(conn)
        for item in dataset:
            qid, question, gold_sql = item["id"], item["question"], item["gold_sql"]
            print(f"[{qid}] {question[:80]}{'...' if len(question) > 80 else ''}")
            try:
                gold_rows = run_gold(conn, gold_sql)
            except psycopg.Error as e:
                conn.rollback()
                print(f"   ⚠ gold SQL failed — fix the dataset entry: {e}")
                continue

            agent = run_agent(question, conn, client, schema)
            s = score(gold_rows, agent["result"])
            if s.correct:
                category = "correct"
                mark = "✓"
            else:
                category = agent["category"] or "wrong_result"
                mark = "✗"

            print(f"   {mark} {category:14s}  gold={len(gold_rows)}r  agent={s.agent_rows}r  "
                  f"${agent['cost']:.4f}  {agent['latency_sec']:.2f}s")
            if not s.correct and args.verbose:
                print(f"      reason: {s.reason}")
                if agent["error"]:
                    print(f"      err: {agent['error']}")
                if agent["sql"]:
                    print(f"      sql: {agent['sql'][:200]}")

            results.append({
                "id": qid, "question": question, "tags": item.get("tags", []),
                "category": category, "correct": s.correct, "reason": s.reason,
                "gold_rows": s.gold_rows, "agent_rows": s.agent_rows,
                "gold_cols": s.gold_cols, "agent_cols": s.agent_cols,
                "cost": agent["cost"], "latency_sec": agent["latency_sec"],
                "input_tokens": agent["input_tokens"], "output_tokens": agent["output_tokens"],
                "agent_sql": agent["sql"], "agent_error": agent["error"],
            })

    print("\n" + "=" * 70)
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    counts = Counter(r["category"] for r in results)
    total_cost = sum(r["cost"] for r in results)
    avg_lat = sum(r["latency_sec"] for r in results) / n if n else 0

    print(f"Baseline — {correct}/{n} correct ({100*correct/n:.0f}%)")
    print("Breakdown:")
    for cat, c in counts.most_common():
        print(f"  {cat:14s} {c}")
    print(f"Total cost:   ${total_cost:.4f}")
    print(f"Avg latency:  {avg_lat:.2f}s")

    if args.save:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RUNS_DIR / f"baseline-{ts}.json"
        out_path.write_text(json.dumps({
            "agent": "baseline",
            "model": "claude-sonnet-4-5",
            "timestamp": ts,
            "summary": {"n": n, "correct": correct, "accuracy": correct / n if n else 0,
                        "total_cost": total_cost, "avg_latency_sec": avg_lat,
                        "category_counts": dict(counts)},
            "results": results,
        }, indent=2, default=str))
        print(f"\nSaved to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
