#!/usr/bin/env bash
#
# scripts/demo.sh — one-command demo launcher.
#
# Starts the FastAPI server, waits for /api/health, opens the browser,
# and blocks until you hit Ctrl+C (cleanly killing the server on exit).
#
# Useful before recording a demo, but also for normal local poking around.

set -euo pipefail

PORT=8000
URL="http://127.0.0.1:${PORT}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Kill any stray uvicorn from a previous run on the same port.
pkill -f "uvicorn server.main" 2>/dev/null || true

echo "Starting server on port $PORT..."
uv run uvicorn server.main:app --host 127.0.0.1 --port "$PORT" --log-level warning &
SERVER_PID=$!

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    echo ""
    echo "Server stopped."
}
trap cleanup EXIT INT TERM

echo -n "Waiting for health..."
for _ in $(seq 1 30); do
    if curl -sf "$URL/api/health" >/dev/null 2>&1; then
        echo " ready."
        break
    fi
    echo -n "."
    sleep 0.5
done

if ! curl -sf "$URL/api/health" >/dev/null 2>&1; then
    echo " failed."
    echo "Server didn't become healthy in 15s. Check the output above for the real error."
    exit 1
fi

if command -v open >/dev/null 2>&1; then
    open "$URL"
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL"
else
    echo "Open $URL in your browser."
fi

echo ""
echo "Demo server: $URL"
echo "Press Ctrl+C to stop."
echo ""

wait "$SERVER_PID"
