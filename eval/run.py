"""
Run an agent against eval/dataset.jsonl and print a scoreboard.

Usage:
    uv run python eval/run.py --agent baseline                  # default
    uv run python eval/run.py --agent multi
    uv run python eval/run.py --agent multi --limit 3 -v
    uv run python eval/run.py --agent multi --ids q09 q10 --save
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

from agent.base import Agent, AgentResult  # noqa: E402
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


def make_agent(name: str, client: anthropic.Anthropic, use_rag: bool = False, rag_k: int = 8) -> Agent:
    if name == "baseline":
        from agent.baseline import BaselineAgent
        return BaselineAgent(client)
    if name == "multi":
        from agent.multi import MultiAgent
        return MultiAgent(client, use_rag=use_rag, rag_k=rag_k)
    raise ValueError(f"unknown agent: {name}")


def run_gold(conn: psycopg.Connection, sql: str) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["baseline", "multi"], default="baseline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", nargs="+", default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--rag", action="store_true",
                        help="(multi only) use vector-routed schema retrieval instead of full schema")
    parser.add_argument("--rag-k", type=int, default=8,
                        help="(multi only) number of column chunks to retrieve when --rag is set")
    args = parser.parse_args()

    dataset = load_dataset(limit=args.limit, ids=args.ids)
    mode = f"{args.agent}{' +rag(k=' + str(args.rag_k) + ')' if args.rag else ''}"
    print(f"Running {len(dataset)} questions against agent='{mode}'...\n")

    db_url = os.environ["DATABASE_URL"]
    client = anthropic.Anthropic()
    agent = make_agent(args.agent, client, use_rag=args.rag, rag_k=args.rag_k)

    results: list[dict] = []

    with psycopg.connect(db_url) as conn:
        for item in dataset:
            qid, question, gold_sql = item["id"], item["question"], item["gold_sql"]
            print(f"[{qid}] {question[:80]}{'...' if len(question) > 80 else ''}")
            try:
                gold_rows = run_gold(conn, gold_sql)
            except psycopg.Error as e:
                conn.rollback()
                print(f"   ⚠ gold SQL failed (fix dataset entry): {e}")
                continue

            r: AgentResult = agent.answer(question, conn)
            s = score(gold_rows, r.result_rows)

            if s.correct:
                category, mark = "correct", "✓"
            else:
                category, mark = r.category or "wrong_result", "✗"

            print(f"   {mark} {category:14s}  gold={len(gold_rows)}r  agent={s.agent_rows}r  "
                  f"${r.cost:.4f}  {r.latency_sec:.2f}s")

            if not s.correct and args.verbose:
                print(f"      reason: {s.reason}")
                if r.error:
                    print(f"      err: {r.error}")
                if r.sql:
                    print(f"      sql: {r.sql[:200]}")
            if args.verbose and r.final_answer:
                print(f"      answer: {r.final_answer[:200]}")

            results.append({
                "id": qid, "question": question, "tags": item.get("tags", []),
                "category": category, "correct": s.correct, "reason": s.reason,
                "gold_rows": s.gold_rows, "agent_rows": s.agent_rows,
                "gold_cols": s.gold_cols, "agent_cols": s.agent_cols,
                "cost": r.cost, "latency_sec": r.latency_sec,
                "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                "agent_sql": r.sql, "agent_error": r.error,
                "final_answer": r.final_answer,
                "extra": r.extra,
            })

    print("\n" + "=" * 70)
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    counts = Counter(r["category"] for r in results)
    total_cost = sum(r["cost"] for r in results)
    avg_lat = sum(r["latency_sec"] for r in results) / n if n else 0

    print(f"{args.agent} — {correct}/{n} correct ({100*correct/n:.0f}%)")
    print("Breakdown:")
    for cat, c in counts.most_common():
        print(f"  {cat:14s} {c}")
    print(f"Total cost:   ${total_cost:.4f}")
    print(f"Avg latency:  {avg_lat:.2f}s")

    if args.save:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RUNS_DIR / f"{args.agent}-{ts}.json"
        out_path.write_text(json.dumps({
            "agent": args.agent,
            "timestamp": ts,
            "summary": {
                "n": n, "correct": correct,
                "accuracy": correct / n if n else 0,
                "total_cost": total_cost, "avg_latency_sec": avg_lat,
                "category_counts": dict(counts),
            },
            "results": results,
        }, indent=2, default=str))
        print(f"\nSaved to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
