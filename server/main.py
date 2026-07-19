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

import asyncio
import datetime as dt
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import psycopg
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT / ".env", override=True)
os.environ.pop("ANTHROPIC_BASE_URL", None)

import anthropic  # noqa: E402

from agent.multi import MultiAgent  # noqa: E402

DB_URL = os.environ["DATABASE_URL"]
STATIC_DIR = ROOT / "static"
INDEX_HTML = STATIC_DIR / "index.html"

app = FastAPI(title="Energy Text-to-SQL")


def _client_ip(request: Request) -> str:
    """Real client IP, honoring Render's X-Forwarded-For proxy header."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return get_remote_address(request)


# Rate limiting: this is a public demo backed by a metered LLM API, so cap
# each client to protect the (spend-capped) Anthropic key from abuse.
limiter = Limiter(key_func=_client_ip, default_limits=[])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@limiter.limit("10/minute;100/day")
def ask(request: Request, payload: Annotated[dict, Body()]) -> JSONResponse:
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


class _JSONEncoder(json.JSONEncoder):
    """Handles Decimal/date/datetime that pop out of Postgres rows."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (dt.datetime, dt.date)):
            return obj.isoformat()
        return super().default(obj)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, cls=_JSONEncoder)}\n\n"


@app.get("/api/ask/stream")
@limiter.limit("10/minute;100/day")
async def ask_stream(request: Request, question: Annotated[str, Query(min_length=1, max_length=1000)]):
    """SSE endpoint — emits one event per LangGraph node completion.

    Client should use EventSource. Each event is either:
      {"node": "plan|synthesize|execute|summarize", "data": {...}}
      {"type": "done", ...totals}
      {"type": "error", "message": "..."}
    """
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    async def generator():
        conn: psycopg.Connection | None = None
        try:
            conn = await asyncio.to_thread(psycopg.connect, DB_URL)
            stream = _agent.stream(question, conn)
            while True:
                try:
                    event = await asyncio.to_thread(next, stream)
                except StopIteration:
                    break
                except anthropic.APIStatusError as e:
                    yield _sse({"type": "error",
                                "message": f"upstream LLM error ({e.status_code}). "
                                           "Anthropic may be rate-limited or overloaded. Try again."})
                    return
                except anthropic.APIConnectionError:
                    yield _sse({"type": "error", "message": "could not reach Anthropic"})
                    return
                except psycopg.Error as e:
                    yield _sse({"type": "error", "message": f"database error: {str(e)[:200]}"})
                    return
                yield _sse(event)
        except psycopg.OperationalError as e:
            yield _sse({"type": "error", "message": f"database unavailable: {e}"})
        finally:
            if conn is not None:
                await asyncio.to_thread(conn.close)

    return StreamingResponse(generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # disable proxy buffering if behind nginx later
    })


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)
