"""
Day 5a: FastAPI wrapper around MultiAgent.

Endpoints:
    GET  /                 → single-page HTML demo
    POST /api/ask          → run multi-agent, return JSON result
    GET  /api/health       → cheap liveness check

Run:
    uv run uvicorn server.main:app --reload --port 8000
    open http://localhost:8000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import psycopg
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT / ".env", override=True)
os.environ.pop("ANTHROPIC_BASE_URL", None)

import anthropic  # noqa: E402

from agent.multi import MultiAgent  # noqa: E402

DB_URL = os.environ["DATABASE_URL"]
STATIC_DIR = ROOT / "static"
INDEX_HTML = STATIC_DIR / "index.html"

app = FastAPI(title="Energy Text-to-SQL")

# One Anthropic client + one MultiAgent instance shared across requests.
# max_retries handles transient 429/5xx (including the 529 "overloaded" that
# Anthropic returns under load) with built-in exponential backoff.
# Postgres connections are per-request (cheap, avoids long-lived connection issues).
_client = anthropic.Anthropic(max_retries=3)
_agent = MultiAgent(_client)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ask")
def ask(payload: Annotated[dict, Body()]) -> JSONResponse:
    question = (payload or {}).get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="question too long (max 1000 chars)")

    try:
        with psycopg.connect(DB_URL) as conn:
            result = _agent.answer(question, conn)
    except psycopg.OperationalError as e:
        raise HTTPException(status_code=503, detail=f"database unavailable: {e}")
    except anthropic.APIStatusError as e:
        # 429 (rate limit), 529 (overloaded), 5xx (other) — Anthropic-side, not our bug
        raise HTTPException(
            status_code=503,
            detail=f"upstream LLM error ({e.status_code}): {getattr(e, 'message', str(e))[:200]}. "
                   "Try again in a few seconds — Anthropic is rate-limited or overloaded."
        )
    except anthropic.APIConnectionError as e:
        raise HTTPException(status_code=502, detail=f"could not reach Anthropic: {e}")

    return JSONResponse({
        "question": question,
        "plan": result.extra.get("plan"),
        "sql": result.sql,
        "result": {
            "columns": result.result_columns or [],
            "rows": [list(r) for r in (result.result_rows or [])][:200],
            "total_rows": len(result.result_rows) if result.result_rows is not None else 0,
        },
        "answer": result.final_answer,
        "error": result.error,
        "category": result.category,
        "cost_usd": result.cost,
        "latency_sec": result.latency_sec,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "retry_count": result.extra.get("retry_count", 0),
    })


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)
